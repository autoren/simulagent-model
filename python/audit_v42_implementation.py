#!/usr/bin/env python3
"""Audit V42 implementation before development-population construction."""

from __future__ import annotations

import argparse
from collections import Counter
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v42_sequential import build_population, corpus_hash
from v42_stateful import canonical_program, mechanic_registry, program_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v42-design-lock.json")
    parser.add_argument("--output", default="outputs/v42-sequential-state-foundation/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors = []
    if not design["authorization"]["write_oracle_implementation"]:
        errors.append("V42 design does not authorize implementation")
    registry = mechanic_registry()
    if len(registry) != 40 or len({row["key"] for row in registry}) != 40:
        errors.append("V42 mechanic registry is not exactly 40 unique programs")
    if any(canonical_program(row["program"]) != row["program"] or program_key(row["program"]) != row["key"] for row in registry):
        errors.append("V42 registry programs are not canonical")
    rows = build_population(config)
    family_counts = Counter(row["construction_family"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    support_counts = [len(row["agent_input"]["support_sequences"]) for row in rows]
    queries = [query for row in rows for query in row["agent_input"]["queries"]]
    sequence_lengths = Counter(query["sequence_length"] for query in queries)
    entity_counts = Counter(query["entity_count"] for query in queries)
    if len(rows) != config["population"]["mechanics"] or set(family_counts.values()) != {10}:
        errors.append("V42 population or family quota mismatch")
    if split_counts != Counter({"development_fit": 24, "development_evaluation": 16}):
        errors.append("V42 split quota mismatch")
    if max(support_counts) > config["population"]["supportSequencesPerMechanicMaximum"]:
        errors.append("V42 support budget exceeded")
    if len(queries) != 960 or set(sequence_lengths) != {2, 3, 4} or set(sequence_lengths.values()) != {320}:
        errors.append("V42 query sequence-length balance failed")
    if set(entity_counts) != {2, 3, 4, 5} or set(entity_counts.values()) != {240}:
        errors.append("V42 entity-count balance failed")
    overlaps = 0
    duplicate_query_ids = 0
    for row in rows:
        support_keys = {value["structural_key"] for value in row["agent_input"]["support_sequences"]}
        query_keys = {value["structural_key"] for value in row["agent_input"]["queries"]}
        overlaps += len(support_keys & query_keys)
        query_ids = [value["id"] for value in row["agent_input"]["queries"]]
        duplicate_query_ids += len(query_ids) - len(set(query_ids))
        if set(query_ids) != {value["id"] for value in row["oracle_queries"]}:
            errors.append(f"V42 query/oracle ID mismatch: {row['id']}")
            break
    if overlaps or duplicate_query_ids:
        errors.append("V42 support/query overlap or duplicate query IDs")
    if any("target" in query for query in queries):
        errors.append("V42 query target exposed in agent input")
    causal_pairs = sum(row["oracle_metadata"]["causal_order_pairs"] for row in rows)
    if causal_pairs <= 0 or any(
        row["oracle_metadata"]["causal_order_pairs"] < 1
        for row in rows if row["construction_family"] == "order_sensitive_composition"
    ):
        errors.append("V42 order-sensitive mechanics lack causal order pairs")
    forbidden = (
        PROJECT_ROOT / "configs/v42-implementation-lock.json",
        PROJECT_ROOT / "data/v42-sequential-state-foundation",
        PROJECT_ROOT / "configs/v42-corpus-seal.json",
        PROJECT_ROOT / "outputs/v42-sequential-state-foundation/development",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V42 downstream artifact exists before implementation lock")
    audit = {
        "schema_version": 42,
        "experiment": "v42_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v42_implementation_lock" if not errors else "repair_v42_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "dry_run": {
            "mechanics": len(rows),
            "registry_programs": len(registry),
            "family_counts": dict(sorted(family_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "support_sequences": sum(support_counts),
            "maximum_support_sequences": max(support_counts),
            "query_sequences": len(queries),
            "sequence_length_counts": {str(key): value for key, value in sorted(sequence_lengths.items())},
            "entity_count_counts": {str(key): value for key, value in sorted(entity_counts.items())},
            "partial_queries": sum(query["partial_initial_state"] for query in queries),
            "causal_order_pairs": causal_pairs,
            "support_query_structural_overlap": overlaps,
            "expected_corpus_sha256": corpus_hash(rows),
            "oracle_development_runs": 0,
            "development_evaluation_predictions": 0,
        },
        "data_access": {
            "oracle_development_runs": 0,
            "language_model_forward_passes": 0,
            "adapter_training_runs": 0,
            "v41_records_read": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
