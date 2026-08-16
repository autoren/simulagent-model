#!/usr/bin/env python3
"""Construct the fresh V31 family-disjoint signed-fact corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import atom_key, canonical_json, deterministic_shuffle, opaque_id, positive_candidate_statement, predicate_specs, sha256_text
from v31_language import ENTITY_ALIASES, SURFACE_TEMPLATES, construction_hash, render_evidence


def make_entities(count: int, token: str) -> list[dict[str, str]]:
    aliases = deterministic_shuffle(ENTITY_ALIASES, f"v31|{token}|aliases")
    types = ["unit", "unit", "hub"]
    while len(types) < count:
        types.append("hub" if len(types) % 2 == 0 else "unit")
    return deterministic_shuffle([
        {"id": aliases[index], "entity_type": entity_type}
        for index, entity_type in enumerate(types)
    ], f"v31|{token}|entity-order")


def base_facts(entities: list[dict[str, str]]) -> list[tuple[str, list[str]]]:
    units = [row["id"] for row in entities if row["entity_type"] == "unit"]
    hubs = [row["id"] for row in entities if row["entity_type"] == "hub"]
    return [
        ("stable", [units[0]]), ("charged", [units[1]]), ("online", [hubs[0]]),
        ("linked", [units[0], units[1]]), ("feeds", [hubs[0], units[0]]),
    ]


def truth_for(operator: str, base_index: int, fact_index: int) -> str:
    if operator == "explicit_unknown":
        return "unknown"
    return "true" if (base_index + fact_index) % 2 == 0 else "false"


def pairs_for(
    split: str, surface_name: str, operator: str, base_index: int, variant: str,
    predicate: str, fact_index: int, truth_status: str,
) -> list[dict[str, str]]:
    prefix = f"v31|{split}|{surface_name}|base{base_index}|fact{fact_index}|{predicate}"
    pairs = []
    if variant in ("direct_clean", "direct_distractor"):
        pairs.append({
            "kind": "distractor", "id": opaque_id("pair", f"{operator}|{prefix}|distractor"),
            "role": "clean" if variant == "direct_clean" else "distractor",
        })
    if predicate in ("linked", "feeds") and variant in ("direct_clean", "inverse_clean"):
        pairs.append({
            "kind": "inverse", "id": opaque_id("pair", f"{operator}|{prefix}|inverse"),
            "role": "direct" if variant == "direct_clean" else "inverse",
        })
    if predicate == "linked" and variant in ("direct_clean", "reversed_link_arguments_clean"):
        pairs.append({
            "kind": "argument_reversal", "id": opaque_id("pair", f"{operator}|{prefix}|reverse"),
            "role": "forward" if variant == "direct_clean" else "reversed",
        })
    if operator in ("affirmative_gold", "negated_opposite"):
        pairs.append({
            "kind": "affirmative_negated", "id": opaque_id("pair", f"{prefix}|{variant}|affirmative-negated"),
            "role": operator,
        })
    if operator in ("affirmative_gold", "double_negation"):
        pairs.append({
            "kind": "affirmative_double_negation", "id": opaque_id("pair", f"{prefix}|{variant}|affirmative-double"),
            "role": operator,
        })
    affirmative_truth = "true" if (base_index + fact_index) % 2 == 0 else "false"
    if (
        (operator == "affirmative_gold" and truth_status == "false")
        or (operator == "explicit_unknown" and affirmative_truth == "false")
    ):
        pairs.append({
            "kind": "false_unknown", "id": opaque_id("pair", f"{prefix}|{variant}|false-unknown"),
            "role": "false" if operator == "affirmative_gold" else "unknown",
        })
    return pairs


def build_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for split, split_config in config["splits"].items():
        for operator in config["semanticOperators"]:
            for surface_index, surface_name in enumerate(split_config["surfaceFamiliesPerOperator"]):
                surface_family = f"{operator}.{surface_name}"
                family_hash = construction_hash(operator, surface_name)
                for base_index in range(config["construction"]["baseScenesPerSurfaceFamily"]):
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
                            raise ValueError(f"Unknown V31 scene variant {variant}")
                        scene_id = opaque_id("scene", f"v31|{config['seed']}|{surface_family}|base{base_index}|{variant}")
                        for fact_index, (predicate, arguments) in selected:
                            truth = truth_for(operator, base_index, fact_index)
                            evidence, length = render_evidence(
                                predicate, arguments, truth, operator, surface_name,
                                orientation, distractor, config,
                            )
                            kind = predicate_specs(config)[predicate]["kind"]
                            record_id = opaque_id("clause", f"{scene_id}|{fact_index}|{predicate}|{'|'.join(arguments)}")
                            records.append({
                                "id": record_id, "schema_version": 31, "split": split,
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
                                },
                                "oracle_metadata": {
                                    "semantic_operator": operator, "surface_name": surface_name,
                                    "surface_family": surface_family,
                                    "construction_hash": family_hash,
                                    "sentence_length_stratum": length,
                                    "scene_variant": variant,
                                    "relation_orientation": orientation if kind == "relation" else None,
                                    "distractor": distractor, "entity_count": count,
                                    "base_scene_index": base_index, "fact_index": fact_index,
                                    "pairs": pairs_for(
                                        split, surface_name, operator, base_index, variant,
                                        predicate, fact_index, truth,
                                    ),
                                },
                            })
    return records


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v31-signed-fact-adaptation.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    output = PROJECT_ROOT / config["outputDir"]
    if output.exists():
        raise RuntimeError(f"V31 corpus already exists: {output}")
    rows = build_records(config)
    output.mkdir(parents=True)
    artifact_hashes = {}
    for split in config["splits"]:
        path = output / f"{split}.jsonl"
        selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["id"])
        path.write_text("".join(canonical_json(row) + "\n" for row in selected))
        artifact_hashes[path.name] = file_sha256(path)
    manifest = {
        "schema_version": 31, "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path), "records": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "surface_family_counts": dict(sorted(Counter(row["oracle_metadata"]["surface_family"] for row in rows).items())),
        "construction_hashes": sorted({row["oracle_metadata"]["construction_hash"] for row in rows}),
        "corpus_sha256": corpus_hash(rows), "artifact_sha256": artifact_hashes,
        "generator_sha256": file_sha256(Path(__file__)),
        "data_access": {
            "benchmark_constructions": 1, "model_forward_passes": 0,
            "evaluation_features_read": 0, "evaluation_predictions_read": 0,
            "training_runs": 0,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
