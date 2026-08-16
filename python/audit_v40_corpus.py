#!/usr/bin/env python3
"""Structural audit for the materialized V40 confirmation."""

from __future__ import annotations

import argparse
from collections import Counter
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v40_confirmation import PACK_NAMES, corpus_hash


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v40-implementation-lock.json")
    parser.add_argument("--output", default="outputs/v40-independent-compiler-confirmation/corpus-audit.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    base = PROJECT_ROOT / "data/v40-independent-compiler-confirmation"
    expected_counts = {"independent_confirmation": 1440, "independent_safety": 120}
    rows = {}
    artifacts = {}
    errors = []
    for name, count in expected_counts.items():
        path = base / f"{name}.jsonl"
        rows[name] = read(path)
        artifacts[name] = {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(rows[name]), "sha256": file_sha256(path)}
        if len(rows[name]) != count or corpus_hash(rows[name]) != lock["expected_corpus_sha256"][name]:
            errors.append(f"V40 {name} differs from implementation lock")
    pack_counts = Counter(row["ontology_pack"] for row in rows["independent_confirmation"])
    safety_counts = Counter(row["expected"]["condition"] for row in rows["independent_safety"])
    if set(pack_counts) != set(PACK_NAMES) or set(pack_counts.values()) != {120}:
        errors.append("V40 ontology pack balance failed")
    if set(safety_counts.values()) != {24} or len(safety_counts) != 5:
        errors.append("V40 safety condition balance failed")
    if any("target" in row["agent_input"] or "expected" in row["agent_input"] for values in rows.values() for row in values):
        errors.append("V40 target information exposed in agent input")
    if file_sha256(PROJECT_ROOT / lock["frozen_compiler"]) != lock["frozen_compiler_sha256"]:
        errors.append("Frozen V39 compiler changed before V40 seal")
    audit = {
        "schema_version": 40,
        "experiment": "v40_corpus_audit",
        "passed": not errors,
        "decision": "authorize_v40_corpus_seal" if not errors else "reject_v40_corpus",
        "errors": errors,
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "structural_checks": {"ontology_pack_counts": dict(sorted(pack_counts.items())), "safety_condition_counts": dict(sorted(safety_counts.items())), "confirmation_records_scored": 0},
        "data_access": {"confirmation_scoring_runs": 0, "model_forward_passes": 0, "v32_evaluation_records_read": 0, "v28_runs": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
