#!/usr/bin/env python3
"""Materialize the locked V36 confirmation only after interface parameter freeze."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import (
    ENTITY_ALIASES, atom_key, canonical_json, deterministic_shuffle, opaque_id,
    positive_candidate_statement, predicate_specs, sha256_text,
)
from v32_language import compile_truth
from v36_language import (
    COLLISION_POLICY, GENERATOR_SEED, NORMALIZATION_VERSION, construction_hash,
    render_evidence, validate_registry,
)


def make_entities(count: int, token: str) -> list[dict[str, str]]:
    aliases = deterministic_shuffle(ENTITY_ALIASES, f"v36|{GENERATOR_SEED}|{token}|aliases")
    types = ["unit", "unit", "hub"]
    while len(types) < count:
        types.append("hub" if len(types) % 2 == 0 else "unit")
    return deterministic_shuffle([
        {"id": aliases[index], "entity_type": entity_type}
        for index, entity_type in enumerate(types)
    ], f"v36|{GENERATOR_SEED}|{token}|entity-order")


def base_facts(entities: list[dict[str, str]]) -> list[tuple[str, list[str]]]:
    units = [row["id"] for row in entities if row["entity_type"] == "unit"]
    hubs = [row["id"] for row in entities if row["entity_type"] == "hub"]
    return [
        ("stable", [units[0]]), ("charged", [units[1]]), ("online", [hubs[0]]),
        ("linked", [units[0], units[1]]), ("feeds", [hubs[0], units[0]]),
    ]


def structural_pairs(
    family: str, sign: str, operation: str, base: int, variant: str,
    predicate: str, fact_index: int,
) -> list[dict[str, str]]:
    prefix = f"v36|{family}|{sign}|{operation}|base{base}|fact{fact_index}|{predicate}"
    pairs = []
    if variant in ("direct_clean", "direct_distractor"):
        pairs.append({
            "kind": "distractor", "id": opaque_id("pair", f"{prefix}|distractor"),
            "role": "clean" if variant == "direct_clean" else "distractor",
        })
    if predicate in ("linked", "feeds") and variant in ("direct_clean", "inverse_clean"):
        pairs.append({
            "kind": "inverse", "id": opaque_id("pair", f"{prefix}|inverse"),
            "role": "direct" if variant == "direct_clean" else "inverse",
        })
    if predicate == "linked" and variant in ("direct_clean", "reversed_link_arguments_clean"):
        pairs.append({
            "kind": "argument_reversal", "id": opaque_id("pair", f"{prefix}|reverse"),
            "role": "forward" if variant == "direct_clean" else "reversed",
        })
    return pairs


def cross_pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metadata, target = row["oracle_metadata"], row["target"]
    return (
        metadata["surface_name"], metadata["base_scene_index"],
        metadata["scene_variant"], metadata["fact_index"], target["predicate"],
        tuple(target["arguments"]), metadata["relation_orientation"], metadata["distractor"],
    )


def attach_cross_pairs(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[cross_pair_key(row)].append(row)
    for key, members in grouped.items():
        by_cell = {
            (row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"]): row
            for row in members
        }
        for operation, kind in (("assert", "lexical_sign_assert"), ("unresolved", "unresolved_sign_invariance")):
            left, right = by_cell.get((operation, "positive")), by_cell.get((operation, "negative"))
            if left is not None and right is not None:
                identifier = opaque_id("pair", f"v36|{kind}|{key}")
                left["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": "positive"})
                right["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": "negative"})
        for sign in ("positive", "negative"):
            assertion = by_cell.get(("assert", sign))
            if assertion is None:
                continue
            for other, kind in (
                ("deny", "scope_assert_deny"),
                ("double_deny", "scope_assert_double_deny"),
                ("contrast_select", "scope_assert_contrast"),
            ):
                comparison = by_cell.get((other, sign))
                if comparison is not None:
                    identifier = opaque_id("pair", f"v36|{kind}|{sign}|{key}")
                    assertion["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": "assert"})
                    comparison["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": other})


def build_records(config: dict[str, Any], v32_config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_registry(config)
    suite = config["confirmationSuite"]
    records = []
    for operation in suite["outerOperations"]:
        for surface_index, surface_name in enumerate(suite["newSurfaceNamesPerOperation"]):
            family = f"{operation}.{surface_name}"
            for lexical_sign in suite["lexicalSignsPerOperation"]:
                for base_index in range(suite["baseScenesPerSurfaceFamilyCell"]):
                    counts = suite["entityCounts"]
                    count = counts[(surface_index + base_index) % len(counts)]
                    entity_token = f"{surface_name}|base{base_index}"
                    entities = make_entities(count, entity_token)
                    facts = base_facts(entities)
                    for variant in suite["sceneVariants"]:
                        if variant == "direct_clean":
                            selected, orientation, distractor = list(enumerate(facts)), "direct", False
                        elif variant == "inverse_clean":
                            selected = [(index, fact) for index, fact in enumerate(facts) if fact[0] in suite["factsInInverseScenes"]]
                            orientation, distractor = "inverse", False
                        elif variant == "direct_distractor":
                            selected, orientation, distractor = list(enumerate(facts)), "direct", True
                        elif variant == "reversed_link_arguments_clean":
                            index = next(index for index, fact in enumerate(facts) if fact[0] == "linked")
                            predicate, arguments = facts[index]
                            selected, orientation, distractor = [(index, (predicate, list(reversed(arguments))))], "direct", False
                        else:
                            raise ValueError(f"Unknown V36 scene variant: {variant}")
                        scene_id = opaque_id("scene", f"v36|{GENERATOR_SEED}|{entity_token}|{operation}|{lexical_sign}|{variant}")
                        for fact_index, (predicate, arguments) in selected:
                            evidence, length = render_evidence(
                                predicate, arguments, lexical_sign, operation, surface_name,
                                orientation, distractor, v32_config,
                            )
                            truth = compile_truth(lexical_sign, operation, v32_config)
                            kind = predicate_specs(v32_config)[predicate]["kind"]
                            record_id = opaque_id("clause", f"{scene_id}|{fact_index}|{predicate}|{'|'.join(arguments)}")
                            records.append({
                                "id": record_id, "schema_version": 36, "split": suite["split"], "scene_id": scene_id,
                                "agent_input": {
                                    "entities": entities,
                                    "predicate_ontology": {
                                        "entity_types": v32_config["ontology"]["entityTypes"],
                                        "unary_predicates": [{"id": row["id"], "entity_type": row["entityType"]} for row in v32_config["ontology"]["unaryPredicates"]],
                                        "relations": [{"id": row["id"], "source_type": row["sourceType"], "target_type": row["targetType"]} for row in v32_config["ontology"]["relations"]],
                                    },
                                    "evidence_text": evidence,
                                },
                                "target": {
                                    "predicate_kind": kind, "predicate": predicate, "arguments": arguments,
                                    "truth_status": truth, "atom": atom_key(predicate, arguments, v32_config),
                                    "candidate_statement": positive_candidate_statement(predicate, arguments, v32_config),
                                    "factorization": {"lexical_sign": lexical_sign, "outer_operation": operation},
                                },
                                "oracle_metadata": {
                                    "evaluation_stratum": "independent_confirmation", "surface_name": surface_name,
                                    "surface_family": family, "construction_hash": construction_hash(operation, surface_name),
                                    "combination_seen_in_v32_fit": lexical_sign in v32_config["factorization"]["fitCells"].get(operation, []),
                                    "sentence_length_stratum": length, "scene_variant": variant,
                                    "relation_orientation": orientation if kind == "relation" else None,
                                    "distractor": distractor, "entity_count": count, "base_scene_index": base_index,
                                    "fact_index": fact_index,
                                    "pairs": structural_pairs(family, lexical_sign, operation, base_index, variant, predicate, fact_index),
                                },
                            })
    attach_cross_pairs(records)
    return records


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-lock", default="configs/v36-interface-lock.json")
    args = parser.parse_args()
    interface_path = (PROJECT_ROOT / args.interface_lock).resolve()
    interface = json.loads(interface_path.read_text())
    implementation_path = PROJECT_ROOT / interface["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    if interface["implementation_lock_sha256"] != file_sha256(implementation_path):
        raise RuntimeError("V36 interface lock does not bind the implementation lock")
    if not interface["authorization"]["construct_confirmation"]:
        raise RuntimeError("V36 interface lock does not authorize confirmation construction")
    config, v32_config = implementation["config_payload"], implementation["v32_config_payload"]
    output = PROJECT_ROOT / config["confirmationSuite"]["outputDir"]
    if output.exists():
        raise RuntimeError("V36 confirmation corpus already exists")
    rows = build_records(config, v32_config)
    output.mkdir(parents=True)
    artifact = output / f"{config['confirmationSuite']['split']}.jsonl"
    artifact.write_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))
    manifest = {
        "schema_version": 36, "experiment": config["experiment"],
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)), "implementation_lock_sha256": file_sha256(implementation_path),
        "interface_lock": str(interface_path.relative_to(PROJECT_ROOT)), "interface_lock_sha256": file_sha256(interface_path),
        "generator_seed": GENERATOR_SEED, "normalization_version": NORMALIZATION_VERSION,
        "collision_policy": COLLISION_POLICY, "records": len(rows), "scenes": len({row["scene_id"] for row in rows}),
        "surface_families": len({row["oracle_metadata"]["surface_family"] for row in rows}),
        "cell_counts": dict(sorted(Counter(
            f"{row['target']['factorization']['outer_operation']}|{row['target']['factorization']['lexical_sign']}"
            for row in rows
        ).items())),
        "corpus_sha256": corpus_hash(rows), "artifact": str(artifact.relative_to(PROJECT_ROOT)),
        "artifact_sha256": file_sha256(artifact), "generator_sha256": file_sha256(Path(__file__)),
        "data_access": {"confirmation_records_constructed": len(rows), "model_forward_passes": 0, "interface_fit_runs": 0, "v32_evaluation_records_read": 0},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
