#!/usr/bin/env python3
"""Materialize the locked V37 fit sample and fresh development validation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import (
    ENTITY_ALIASES,
    atom_key,
    canonical_json,
    deterministic_shuffle,
    opaque_id,
    positive_candidate_statement,
    predicate_specs,
    sha256_text,
)
from v32_language import compile_truth
from v37_language import construction_hash, render_evidence, validate_registry


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_order(rows: list[dict[str, Any]], seed: int, token: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"v37|{seed}|{token}|{row['id']}".encode()).hexdigest(),
    )


def take_balanced_distractor(
    rows: list[dict[str, Any]], count: int, seed: int, token: str
) -> list[dict[str, Any]]:
    false_rows = stable_order(
        [row for row in rows if not row["oracle_metadata"]["distractor"]], seed, f"{token}|clean"
    )
    true_rows = stable_order(
        [row for row in rows if row["oracle_metadata"]["distractor"]], seed, f"{token}|distractor"
    )
    half = count // 2
    if len(false_rows) < half or len(true_rows) < count - half:
        raise ValueError(f"Insufficient distractor balance for {token}")
    return false_rows[:half] + true_rows[: count - half]


def build_fit_sample(config: dict[str, Any], v32_config: dict[str, Any]) -> list[dict[str, Any]]:
    spec = config["developmentFit"]
    sources: dict[str, list[dict[str, Any]]] = {}
    for source in config["allowedTrainingSources"]:
        sources[source["source"]] = read_jsonl(PROJECT_ROOT / source["corpus"])
    records: list[dict[str, Any]] = []
    available = v32_config["factorization"]["fitCells"]
    for operation in config["interfaces"]["outerOperationClasses"]:
        for sign in config["interfaces"]["lexicalSignClasses"]:
            cell_available = sign in available.get(operation, [])
            quotas = {
                "v32_factor_fit": spec["recordsFromV32WhenCellAvailable"] if cell_available else 0,
                "v36_exposed_confirmation": (
                    spec["recordsFromV36WhenV32CellAvailable"]
                    if cell_available else spec["recordsFromV36WhenV32CellUnavailable"]
                ),
            }
            if sum(quotas.values()) != spec["recordsPerOperationSignCell"]:
                raise ValueError("V37 cell quotas do not sum to the registered population")
            for source_name, quota in quotas.items():
                candidates = [
                    row for row in sources[source_name]
                    if row["target"]["factorization"] == {
                        "lexical_sign": sign, "outer_operation": operation
                    }
                ]
                selected = take_balanced_distractor(
                    candidates, quota, spec["samplingSeed"], f"{source_name}|{operation}|{sign}"
                ) if quota else []
                for source_row in selected:
                    surface = source_row["oracle_metadata"]["surface_name"]
                    copied = json.loads(json.dumps(source_row))
                    copied["id"] = opaque_id("v37fit", f"{source_name}|{source_row['id']}")
                    copied["schema_version"] = 37
                    copied["split"] = "semantic_invariance_fit"
                    copied["oracle_metadata"].update({
                        "v37_source": source_name,
                        "v37_source_record_id": source_row["id"],
                        "v37_selection_group": f"{source_name}|{operation}|{surface}",
                        "evaluation_stratum": "semantic_invariance_fit",
                    })
                    records.append(copied)
    if len(records) != spec["expectedRecords"] or len({row["id"] for row in records}) != len(records):
        raise ValueError("V37 fit sample does not match the registered population")
    return sorted(records, key=lambda row: row["id"])


def make_entities(count: int, token: str, seed: int) -> list[dict[str, str]]:
    aliases = deterministic_shuffle(ENTITY_ALIASES, f"v37|{seed}|{token}|aliases")
    types = ["unit", "unit", "hub"]
    while len(types) < count:
        types.append("hub" if len(types) % 2 == 0 else "unit")
    return deterministic_shuffle(
        [{"id": aliases[index], "entity_type": entity_type} for index, entity_type in enumerate(types)],
        f"v37|{seed}|{token}|entity-order",
    )


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


def validation_pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metadata, target = row["oracle_metadata"], row["target"]
    return (
        metadata["surface_name"], metadata["scene_variant"], metadata["fact_index"],
        target["predicate"], tuple(target["arguments"]), metadata["relation_orientation"],
    )


def attach_validation_pairs(rows: list[dict[str, Any]]) -> None:
    # Clean/prefix/suffix controls share operation, sign, surface, and fact.
    distraction: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata, target = row["oracle_metadata"], row["target"]
        if metadata["scene_variant"] in (
            "direct_clean", "direct_distractor_prefix", "direct_distractor_suffix"
        ):
            key = (
                metadata["surface_family"], target["factorization"]["lexical_sign"],
                metadata["fact_index"], target["predicate"], tuple(target["arguments"]),
            )
            distraction[key].append(row)
    for key, members in distraction.items():
        if len(members) == 3:
            identifier = opaque_id("pair", f"v37|distractor_position|{key}")
            for row in members:
                row["oracle_metadata"]["pairs"].append({
                    "kind": "distractor_position", "id": identifier,
                    "role": row["oracle_metadata"]["distractor_placement"],
                })

    # Sign and scope controls compare aligned surfaces and variants.
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata, target = row["oracle_metadata"], row["target"]
        key = (
            metadata["surface_name"], metadata["scene_variant"], metadata["fact_index"],
            target["predicate"], tuple(target["arguments"]), metadata["relation_orientation"],
        )
        grouped[key].append(row)
    for key, members in grouped.items():
        by_cell = {
            (row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"]): row
            for row in members
        }
        for operation in ("assert", "unresolved"):
            left, right = by_cell.get((operation, "positive")), by_cell.get((operation, "negative"))
            if left and right:
                kind = "lexical_sign_assert" if operation == "assert" else "unresolved_sign_invariance"
                identifier = opaque_id("pair", f"v37|{kind}|{key}")
                left["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": "positive"})
                right["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": "negative"})
        for sign in ("positive", "negative"):
            assertion = by_cell.get(("assert", sign))
            if not assertion:
                continue
            for operation in ("deny", "double_deny", "contrast_select"):
                other = by_cell.get((operation, sign))
                if other:
                    kind = f"scope_assert_{operation}"
                    identifier = opaque_id("pair", f"v37|{kind}|{sign}|{key}")
                    assertion["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": "assert"})
                    other["oracle_metadata"]["pairs"].append({"kind": kind, "id": identifier, "role": operation})


def build_validation(config: dict[str, Any], v32_config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_registry(config)
    spec = config["freshValidation"]
    seed = spec["generatorSeed"]
    records: list[dict[str, Any]] = []
    for operation in config["interfaces"]["outerOperationClasses"]:
        for surface_index, surface_name in enumerate(spec["surfaceNamesPerOperation"]):
            family = f"{operation}.{surface_name}"
            for sign in spec["lexicalSignsPerOperation"]:
                for variant_index, variant in enumerate(spec["sceneVariants"]):
                    variant_group = (
                        "direct"
                        if variant in ("direct_clean", "direct_distractor_prefix", "direct_distractor_suffix")
                        else variant
                    )
                    group_index = {"direct": 0, "inverse_clean": 1, "reversed_link_arguments_clean": 2}[variant_group]
                    count = spec["entityCounts"][(surface_index + group_index) % len(spec["entityCounts"])]
                    entity_token = f"surface{surface_index}|variant-group{variant_group}|count{count}"
                    entities = make_entities(count, entity_token, seed)
                    facts = base_facts(entities)
                    if variant == "direct_clean":
                        selected, orientation, placement = list(enumerate(facts)), "direct", "none"
                    elif variant == "inverse_clean":
                        selected = [(index, fact) for index, fact in enumerate(facts) if fact[0] in ("linked", "feeds")]
                        orientation, placement = "inverse", "none"
                    elif variant == "direct_distractor_prefix":
                        selected, orientation, placement = list(enumerate(facts)), "direct", "prefix"
                    elif variant == "direct_distractor_suffix":
                        selected, orientation, placement = list(enumerate(facts)), "direct", "suffix"
                    elif variant == "reversed_link_arguments_clean":
                        fact_index = next(index for index, fact in enumerate(facts) if fact[0] == "linked")
                        predicate, arguments = facts[fact_index]
                        selected = [(fact_index, (predicate, list(reversed(arguments))))]
                        orientation, placement = "direct", "none"
                    else:
                        raise ValueError(f"Unknown V37 validation variant: {variant}")
                    scene_id = opaque_id(
                        "scene", f"v37|{seed}|{operation}|{surface_name}|{sign}|{variant}|{entity_token}"
                    )
                    for fact_index, (predicate, arguments) in selected:
                        evidence, length = render_evidence(
                            predicate, arguments, sign, operation, surface_name,
                            orientation, placement, v32_config,
                        )
                        kind = predicate_specs(v32_config)[predicate]["kind"]
                        record_id = opaque_id(
                            "v37val", f"{scene_id}|{fact_index}|{predicate}|{'|'.join(arguments)}"
                        )
                        records.append({
                            "id": record_id,
                            "schema_version": 37,
                            "split": spec["split"],
                            "scene_id": scene_id,
                            "agent_input": {
                                "entities": entities,
                                "predicate_ontology": {
                                    "entity_types": v32_config["ontology"]["entityTypes"],
                                    "unary_predicates": [
                                        {"id": row["id"], "entity_type": row["entityType"]}
                                        for row in v32_config["ontology"]["unaryPredicates"]
                                    ],
                                    "relations": [
                                        {"id": row["id"], "source_type": row["sourceType"], "target_type": row["targetType"]}
                                        for row in v32_config["ontology"]["relations"]
                                    ],
                                },
                                "evidence_text": evidence,
                            },
                            "target": {
                                "predicate_kind": kind,
                                "predicate": predicate,
                                "arguments": arguments,
                                "truth_status": compile_truth(sign, operation, v32_config),
                                "atom": atom_key(predicate, arguments, v32_config),
                                "candidate_statement": positive_candidate_statement(predicate, arguments, v32_config),
                                "factorization": {"lexical_sign": sign, "outer_operation": operation},
                            },
                            "oracle_metadata": {
                                "evaluation_stratum": "semantic_invariance_validation",
                                "surface_name": surface_name,
                                "surface_family": family,
                                "construction_hash": construction_hash(operation, surface_name),
                                "sentence_length_stratum": length,
                                "scene_variant": variant,
                                "relation_orientation": orientation if kind == "relation" else None,
                                "distractor": placement != "none",
                                "distractor_placement": placement,
                                "entity_count": count,
                                "base_scene_index": 0,
                                "fact_index": fact_index,
                                "pairs": [],
                            },
                        })
    attach_validation_pairs(records)
    return sorted(records, key=lambda row: row["id"])


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v37-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_corpus"]:
        raise RuntimeError("V37 implementation lock does not authorize corpus construction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V37 locked implementation changed: {path}")
    config, v32_config = lock["config_payload"], lock["v32_config_payload"]
    output = PROJECT_ROOT / "data/v37-semantic-invariance"
    if output.exists():
        raise RuntimeError("V37 corpus already exists")
    fit_rows = build_fit_sample(config, v32_config)
    validation_rows = build_validation(config, v32_config)
    expected = lock["expected_corpora"]
    if corpus_hash(fit_rows) != expected["fit_corpus_sha256"] or corpus_hash(validation_rows) != expected["validation_corpus_sha256"]:
        raise RuntimeError("V37 constructed corpus differs from implementation lock")
    output.mkdir(parents=True)
    artifacts = {}
    for name, rows in (("semantic_invariance_fit", fit_rows), ("semantic_invariance_validation", validation_rows)):
        path = output / f"{name}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        artifacts[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "records": len(rows),
            "artifact_sha256": file_sha256(path),
            "corpus_sha256": corpus_hash(rows),
        }
    manifest = {
        "schema_version": 37,
        "experiment": config["experiment"],
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "fit_cell_counts": dict(sorted(Counter(
            f"{row['target']['factorization']['outer_operation']}|{row['target']['factorization']['lexical_sign']}"
            for row in fit_rows
        ).items())),
        "fit_source_counts": dict(sorted(Counter(row["oracle_metadata"]["v37_source"] for row in fit_rows).items())),
        "validation_surface_families": len({row["oracle_metadata"]["surface_family"] for row in validation_rows}),
        "validation_scenes": len({row["scene_id"] for row in validation_rows}),
        "data_access": {
            "fit_records_materialized": len(fit_rows),
            "validation_records_constructed": len(validation_rows),
            "backbone_forward_passes": 0,
            "v32_evaluation_records_read": 0,
            "v28_runs": 0,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
