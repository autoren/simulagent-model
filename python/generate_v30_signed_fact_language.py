#!/usr/bin/env python3
"""Materialize the fresh V30 surface-family-sealed language benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import (
    ENTITY_ALIASES, SURFACE_TEMPLATES, atom_key, canonical_json, deterministic_shuffle,
    opaque_id, positive_candidate_statement, predicate_specs, render_evidence, sha256_text,
)


def make_entities(count: int, token: str) -> list[dict[str, str]]:
    aliases = deterministic_shuffle(ENTITY_ALIASES, f"{token}|aliases")
    types = ["unit", "unit", "hub"]
    while len(types) < count:
        types.append("unit" if len(types) % 2 else "hub")
    entities = [
        {"id": aliases[index], "entity_type": entity_type}
        for index, entity_type in enumerate(types)
    ]
    return deterministic_shuffle(entities, f"{token}|entity-order")


def base_facts(entities: list[dict[str, str]]) -> list[tuple[str, list[str]]]:
    units = [row["id"] for row in entities if row["entity_type"] == "unit"]
    hubs = [row["id"] for row in entities if row["entity_type"] == "hub"]
    return [
        ("stable", [units[0]]),
        ("charged", [units[1]]),
        ("online", [hubs[0]]),
        ("linked", [units[0], units[1]]),
        ("feeds", [hubs[0], units[0]]),
    ]


def truth_for(operator: str, base_index: int, fact_index: int) -> str:
    if operator == "explicit_unknown":
        return "unknown"
    return "true" if (base_index + fact_index) % 2 == 0 else "false"


def pair_rows(
    split: str, surface_name: str, operator: str, base_index: int,
    variant: str, predicate: str, fact_index: int, truth_status: str,
) -> list[dict[str, str]]:
    prefix = f"{split}|{surface_name}|base{base_index}|fact{fact_index}|{predicate}"
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
            "kind": "argument_reversal",
            "id": opaque_id("pair", f"{operator}|{prefix}|argument-reversal"),
            "role": "forward" if variant == "direct_clean" else "reversed",
        })
    if operator in ("affirmative_gold", "negated_opposite"):
        pairs.append({
            "kind": "affirmative_negated",
            "id": opaque_id("pair", f"{prefix}|{variant}|affirmative-negated"),
            "role": operator,
        })
    affirmative_truth = "true" if (base_index + fact_index) % 2 == 0 else "false"
    if (
        (operator == "affirmative_gold" and truth_status == "false")
        or (operator == "explicit_unknown" and affirmative_truth == "false")
    ):
        pairs.append({
            "kind": "false_unknown",
            "id": opaque_id("pair", f"{prefix}|{variant}|false-unknown"),
            "role": "false" if operator == "affirmative_gold" else "unknown",
        })
    return pairs


def build_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    surface_ordinal = 0
    for split, split_config in config["splits"].items():
        for operator in config["semanticOperators"]:
            for surface_name in split_config["surfaceFamiliesPerOperator"]:
                if surface_name not in SURFACE_TEMPLATES[operator]:
                    raise ValueError(f"Missing template for {operator}.{surface_name}")
                surface_family = f"{operator}.{surface_name}"
                for base_index in range(config["construction"]["baseScenesPerSurfaceFamily"]):
                    counts = config["construction"]["entityCounts"]
                    surface_index = split_config["surfaceFamiliesPerOperator"].index(surface_name)
                    entity_count = counts[(surface_index + base_index) % len(counts)]
                    entity_token = f"{config['seed']}|{split}|{surface_name}|base{base_index}"
                    entities = make_entities(entity_count, entity_token)
                    facts = base_facts(entities)
                    for variant in config["construction"]["sceneVariants"]:
                        if variant == "direct_clean":
                            selected = list(enumerate(facts))
                            orientation, distractor = "direct", False
                        elif variant == "inverse_clean":
                            selected = [
                                (index, fact) for index, fact in enumerate(facts)
                                if fact[0] in config["construction"]["factsInInverseScenes"]
                            ]
                            orientation, distractor = "inverse", False
                        elif variant == "direct_distractor":
                            selected = list(enumerate(facts))
                            orientation, distractor = "direct", True
                        elif variant == "reversed_link_arguments_clean":
                            linked_index = next(
                                index for index, fact in enumerate(facts) if fact[0] == "linked"
                            )
                            predicate, arguments = facts[linked_index]
                            selected = [(linked_index, (predicate, list(reversed(arguments))))]
                            orientation, distractor = "direct", False
                        else:
                            raise ValueError(f"Unknown scene variant {variant}")
                        scene_id = opaque_id(
                            "scene", f"{config['seed']}|{surface_family}|base{base_index}|{variant}"
                        )
                        for fact_index, (predicate, arguments) in selected:
                            truth_status = truth_for(operator, base_index, fact_index)
                            evidence, length = render_evidence(
                                predicate, arguments, truth_status, operator, surface_name,
                                orientation, distractor, config,
                            )
                            record_id = opaque_id(
                                "clause", f"{scene_id}|fact{fact_index}|{predicate}|{'|'.join(arguments)}"
                            )
                            kind = predicate_specs(config)[predicate]["kind"]
                            target = {
                                "predicate_kind": kind,
                                "predicate": predicate,
                                "arguments": arguments,
                                "truth_status": truth_status,
                                "atom": atom_key(predicate, arguments, config),
                                "candidate_statement": positive_candidate_statement(
                                    predicate, arguments, config
                                ),
                            }
                            records.append({
                                "id": record_id,
                                "schema_version": 30,
                                "split": split,
                                "scene_id": scene_id,
                                "agent_input": {
                                    "entities": entities,
                                    "predicate_ontology": {
                                        "entity_types": config["ontology"]["entityTypes"],
                                        "unary_predicates": [
                                            {"id": row["id"], "entity_type": row["entityType"]}
                                            for row in config["ontology"]["unaryPredicates"]
                                        ],
                                        "relations": [
                                            {
                                                "id": row["id"],
                                                "source_type": row["sourceType"],
                                                "target_type": row["targetType"],
                                            }
                                            for row in config["ontology"]["relations"]
                                        ],
                                    },
                                    "evidence_text": evidence,
                                },
                                "target": target,
                                "oracle_metadata": {
                                    "semantic_operator": operator,
                                    "surface_name": surface_name,
                                    "surface_family": surface_family,
                                    "surface_ordinal": surface_ordinal,
                                    "sentence_length_stratum": length,
                                    "scene_variant": variant,
                                    "relation_orientation": orientation if kind == "relation" else None,
                                    "distractor": distractor,
                                    "entity_count": entity_count,
                                    "base_scene_index": base_index,
                                    "fact_index": fact_index,
                                    "pairs": pair_rows(
                                        split, surface_name, operator, base_index, variant,
                                        predicate, fact_index, truth_status,
                                    ),
                                },
                            })
                surface_ordinal += 1
    return records


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: row["id"])
    return sha256_text("".join(canonical_json(row) + "\n" for row in ordered))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v30-signed-fact-language.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    output = PROJECT_ROOT / config["outputDir"]
    if output.exists():
        raise RuntimeError(f"V30 benchmark already exists: {output}")
    rows = build_records(config)
    output.mkdir(parents=True)
    artifact_hashes = {}
    for split in config["splits"]:
        path = output / f"{split}.jsonl"
        selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["id"])
        path.write_text("".join(canonical_json(row) + "\n" for row in selected))
        artifact_hashes[path.name] = file_sha256(path)
    manifest = {
        "schema_version": 30,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "records": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "surface_family_counts": dict(sorted(Counter(
            row["oracle_metadata"]["surface_family"] for row in rows
        ).items())),
        "semantic_operator_counts": dict(sorted(Counter(
            row["oracle_metadata"]["semantic_operator"] for row in rows
        ).items())),
        "corpus_sha256": corpus_hash(rows),
        "artifact_sha256": artifact_hashes,
        "generator_sha256": file_sha256(Path(__file__)),
        "data_access": {
            "benchmark_constructions": 1,
            "model_forward_passes": 0,
            "evaluation_predictions_read": 0,
            "head_fits": 0,
            "threshold_fits": 0,
            "adapter_training_runs": 0,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
