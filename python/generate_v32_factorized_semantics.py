#!/usr/bin/env python3
"""Generate the fresh V32 paraphrase and compositional-holdout corpus."""

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
from v32_language import compile_truth, construction_hash, render_evidence


def make_entities(count: int, token: str) -> list[dict[str, str]]:
    aliases = deterministic_shuffle(ENTITY_ALIASES, f"v32|{token}|aliases")
    types = ["unit", "unit", "hub"]
    while len(types) < count:
        types.append("hub" if len(types) % 2 == 0 else "unit")
    return deterministic_shuffle([
        {"id": aliases[index], "entity_type": entity_type}
        for index, entity_type in enumerate(types)
    ], f"v32|{token}|entity-order")


def base_facts(entities: list[dict[str, str]]) -> list[tuple[str, list[str]]]:
    units = [row["id"] for row in entities if row["entity_type"] == "unit"]
    hubs = [row["id"] for row in entities if row["entity_type"] == "hub"]
    return [
        ("stable", [units[0]]), ("charged", [units[1]]), ("online", [hubs[0]]),
        ("linked", [units[0], units[1]]), ("feeds", [hubs[0], units[0]]),
    ]


def structural_pairs(
    split: str, family: str, sign: str, operation: str, base: int, variant: str,
    predicate: str, fact_index: int,
) -> list[dict[str, str]]:
    prefix = f"v32|{split}|{family}|{sign}|{operation}|base{base}|fact{fact_index}|{predicate}"
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
        row["split"], metadata["surface_name"], metadata["base_scene_index"],
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
                identifier = opaque_id("pair", f"v32|{kind}|{key}")
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
                if comparison is None:
                    continue
                identifier = opaque_id("pair", f"v32|{kind}|{sign}|{key}")
                assertion["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": "assert"})
                comparison["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": other})


def build_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for split, split_config in config["splits"].items():
        cells = config["factorization"][split_config["cells"]]
        stratum = (
            "fit" if split == "factor_fit" else "calibration" if split == "factor_calibration"
            else "paraphrase" if split == "factor_evaluation_paraphrase" else "composition"
        )
        for operation, signs in cells.items():
            for surface_index, surface_name in enumerate(split_config["surfaceNames"]):
                surface_family = f"{operation}.{surface_name}"
                for lexical_sign in signs:
                    for base_index in range(config["construction"]["baseScenesPerSurfaceFamilyCell"]):
                        counts = config["construction"]["entityCounts"]
                        count = counts[(surface_index + base_index) % len(counts)]
                        entity_token = f"{config['seed']}|{split}|{surface_name}|base{base_index}"
                        entities = make_entities(count, entity_token)
                        facts = base_facts(entities)
                        for variant in config["construction"]["sceneVariants"]:
                            if variant == "direct_clean":
                                selected, orientation, distractor = list(enumerate(facts)), "direct", False
                            elif variant == "inverse_clean":
                                selected = [(i, fact) for i, fact in enumerate(facts) if fact[0] in config["construction"]["factsInInverseScenes"]]
                                orientation, distractor = "inverse", False
                            elif variant == "direct_distractor":
                                selected, orientation, distractor = list(enumerate(facts)), "direct", True
                            elif variant == "reversed_link_arguments_clean":
                                i = next(i for i, fact in enumerate(facts) if fact[0] == "linked")
                                predicate, arguments = facts[i]
                                selected, orientation, distractor = [(i, (predicate, list(reversed(arguments))))], "direct", False
                            else:
                                raise ValueError(f"Unknown V32 variant: {variant}")
                            scene_id = opaque_id(
                                "scene",
                                f"v32|{entity_token}|{operation}|{lexical_sign}|{variant}",
                            )
                            for fact_index, (predicate, arguments) in selected:
                                evidence, length = render_evidence(
                                    predicate, arguments, lexical_sign, operation, surface_name,
                                    orientation, distractor, config,
                                )
                                truth = compile_truth(lexical_sign, operation, config)
                                kind = predicate_specs(config)[predicate]["kind"]
                                record_id = opaque_id("clause", f"{scene_id}|{fact_index}|{predicate}|{'|'.join(arguments)}")
                                records.append({
                                    "id": record_id, "schema_version": 32, "split": split,
                                    "scene_id": scene_id,
                                    "agent_input": {
                                        "entities": entities,
                                        "predicate_ontology": {
                                            "entity_types": config["ontology"]["entityTypes"],
                                            "unary_predicates": [{"id": row["id"], "entity_type": row["entityType"]} for row in config["ontology"]["unaryPredicates"]],
                                            "relations": [{"id": row["id"], "source_type": row["sourceType"], "target_type": row["targetType"]} for row in config["ontology"]["relations"]],
                                        },
                                        "evidence_text": evidence,
                                    },
                                    "target": {
                                        "predicate_kind": kind, "predicate": predicate,
                                        "arguments": arguments, "truth_status": truth,
                                        "atom": atom_key(predicate, arguments, config),
                                        "candidate_statement": positive_candidate_statement(predicate, arguments, config),
                                        "factorization": {"lexical_sign": lexical_sign, "outer_operation": operation},
                                    },
                                    "oracle_metadata": {
                                        "evaluation_stratum": stratum, "surface_name": surface_name,
                                        "surface_family": surface_family,
                                        "construction_hash": construction_hash(operation, surface_name),
                                        "combination_seen_in_fit": lexical_sign in config["factorization"]["fitCells"].get(operation, []),
                                        "sentence_length_stratum": length, "scene_variant": variant,
                                        "relation_orientation": orientation if kind == "relation" else None,
                                        "distractor": distractor, "entity_count": count,
                                        "base_scene_index": base_index, "fact_index": fact_index,
                                        "pairs": structural_pairs(
                                            split, surface_family, lexical_sign, operation, base_index,
                                            variant, predicate, fact_index,
                                        ),
                                    },
                                })
    attach_cross_pairs(records)
    return records


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v32-factorized-semantics.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    output = PROJECT_ROOT / config["outputDir"]
    if output.exists():
        raise RuntimeError("V32 corpus already exists")
    rows = build_records(config)
    output.mkdir(parents=True)
    artifact_hashes = {}
    for split in config["splits"]:
        path = output / f"{split}.jsonl"
        selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["id"])
        path.write_text("".join(canonical_json(row) + "\n" for row in selected))
        artifact_hashes[path.name] = file_sha256(path)
    manifest = {
        "schema_version": 32, "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path), "records": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "surface_family_counts_by_split": {
            split: dict(sorted(Counter(
                row["oracle_metadata"]["surface_family"] for row in rows if row["split"] == split
            ).items())) for split in config["splits"]
        },
        "corpus_sha256": corpus_hash(rows), "artifact_sha256": artifact_hashes,
        "generator_sha256": file_sha256(Path(__file__)),
        "data_access": {"model_forward_passes": 0, "training_runs": 0, "evaluation_predictions": 0},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
