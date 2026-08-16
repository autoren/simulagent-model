#!/usr/bin/env python3
"""Structurally audit the V42 materialized population without scoring it."""

from __future__ import annotations

import argparse
from collections import Counter
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v42_sequential import corpus_hash


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v42-implementation-lock.json")
    parser.add_argument("--output", default="outputs/v42-sequential-state-foundation/corpus-audit.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    base = PROJECT_ROOT / "data/v42-sequential-state-foundation"
    errors = []
    artifacts = {}
    rows = []
    expected_split_counts = {"development_fit": 24, "development_evaluation": 16}
    for split, count in expected_split_counts.items():
        path = base / f"{split}.jsonl"
        selected = read(path)
        rows.extend(selected)
        artifacts[split] = {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(selected), "sha256": file_sha256(path)}
        if len(selected) != count:
            errors.append(f"V42 {split} count differs from lock")
    if corpus_hash(rows) != lock["expected_corpus_sha256"]:
        errors.append("V42 corpus differs from implementation lock")
    counts = {
        "mechanics": len(rows),
        "support_sequences": sum(len(row["agent_input"]["support_sequences"]) for row in rows),
        "query_sequences": sum(len(row["agent_input"]["queries"]) for row in rows),
        "partial_queries": sum(query["partial_initial_state"] for row in rows for query in row["agent_input"]["queries"]),
        "causal_order_pairs": sum(row["oracle_metadata"]["causal_order_pairs"] for row in rows),
    }
    if counts != lock["expected_counts"]:
        errors.append("V42 corpus counts differ from implementation lock")
    if any("target" in query for row in rows for query in row["agent_input"]["queries"]):
        errors.append("V42 query target exposed in agent input")
    if len({row["target"]["program_key"] for row in rows}) != 40:
        errors.append("V42 target program keys are not unique")
    audit = {
        "schema_version": 42,
        "experiment": "v42_corpus_audit",
        "passed": not errors,
        "decision": "authorize_v42_corpus_seal" if not errors else "reject_v42_corpus",
        "errors": errors,
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "structural_checks": {
            **counts,
            "family_counts": dict(Counter(row["construction_family"] for row in rows)),
            "oracle_development_runs": 0,
        },
        "data_access": {"oracle_development_runs": 0, "language_model_forward_passes": 0, "adapter_training_runs": 0, "v41_records_read": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
