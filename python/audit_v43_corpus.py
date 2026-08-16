#!/usr/bin/env python3
"""Structurally audit the V43 paired language corpus without scoring it."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v43_language import corpus_hash, population_counts, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v43-implementation-lock.json")
    parser.add_argument("--output", default="outputs/v43-sequential-language-grounding/corpus-audit.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    base = PROJECT_ROOT / "data/v43-sequential-language-grounding"
    errors, artifacts, rows = [], {}, []
    for split, expected in (("development_fit", 24), ("development_evaluation", 16)):
        path = base / f"{split}.jsonl"
        selected = read(path)
        rows.extend(selected)
        artifacts[split] = {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(selected), "sha256": file_sha256(path)}
        if len(selected) != expected:
            errors.append(f"V43 {split} count differs from lock")
    if corpus_hash(rows) != lock["expected_corpus_sha256"]:
        errors.append("V43 corpus differs from implementation lock")
    counts = population_counts(rows)
    comparable = {key: counts[key] for key in lock["expected_counts"]}
    if comparable != lock["expected_counts"]:
        errors.append("V43 corpus counts differ from implementation lock")
    exposed_raw = any(
        any(key in sequence for key in ("initial_state", "actions", "observed_step_states"))
        for row in rows
        for section in (row["agent_input"]["support_sequences"], row["agent_input"]["queries"])
        for sequence in section
    )
    if exposed_raw:
        errors.append("V43 exposes raw symbolic sequence fields")
    if any("target" in query for row in rows for query in row["agent_input"]["queries"]):
        errors.append("V43 query target exposed in agent input")
    audit = {
        "schema_version": 43,
        "experiment": "v43_corpus_audit",
        "passed": not errors,
        "decision": "authorize_v43_corpus_seal" if not errors else "reject_v43_corpus",
        "errors": errors,
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "structural_checks": {**counts, "raw_symbolic_agent_input_fields": int(exposed_raw), "paired_development_runs": 0},
        "data_access": {
            "v42_records_read_during_audit": 0,
            "paired_development_runs": 0,
            "language_model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
