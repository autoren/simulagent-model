#!/usr/bin/env python3
"""Audit the materialized V39 corpus before any held-out scoring."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v39_compiler import corpus_hash


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v39-implementation-lock.json")
    parser.add_argument("--output", default="outputs/v39-declared-language-compiler/corpus-audit.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    base = PROJECT_ROOT / "data/v39-declared-language-compiler"
    errors: list[str] = []
    rows = {}
    artifacts = {}
    expected_counts = {"compiler_development": 360, "supported_evaluation": 360, "compiler_safety": 240, "novel_paraphrase_diagnostic": 50}
    for name, count in expected_counts.items():
        path = base / f"{name}.jsonl"
        rows[name] = read(path)
        artifacts[name] = {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(rows[name]), "sha256": file_sha256(path)}
        if len(rows[name]) != count or corpus_hash(rows[name]) != lock["expected_corpus_sha256"][name]:
            errors.append(f"V39 {name} differs from implementation lock")
    overlap = {
        row["agent_input"]["evidence_text"] for row in rows["compiler_development"]
    } & {
        row["agent_input"]["evidence_text"] for row in rows["supported_evaluation"]
    }
    if overlap:
        errors.append("V39 development/evaluation evidence overlap")
    if any("target" in row["agent_input"] or "expected" in row["agent_input"] for population in rows.values() for row in population):
        errors.append("V39 target or expected status exposed in agent input")
    eval_cells = {row["oracle_metadata"]["composition_cell"] for row in rows["supported_evaluation"]}
    if len(eval_cells) != 120 or any(sum(row["oracle_metadata"]["composition_cell"] == cell for row in rows["supported_evaluation"]) != 3 for cell in eval_cells):
        errors.append("V39 held-out composition cells are not balanced")
    challenge_counts = {}
    for row in rows["compiler_safety"]:
        kind = row["expected"]["challenge_kind"]
        challenge_counts[kind] = challenge_counts.get(kind, 0) + 1
    if set(challenge_counts.values()) != {60} or len(challenge_counts) != 4:
        errors.append("V39 safety challenges are not balanced")
    audit = {
        "schema_version": 39,
        "experiment": "v39_corpus_audit",
        "passed": not errors,
        "decision": "authorize_v39_corpus_seal" if not errors else "repair_v39_corpus",
        "errors": errors,
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "structural_checks": {
            "development_evaluation_exact_overlap": len(overlap),
            "held_out_composition_cells": len(eval_cells),
            "safety_challenge_counts": challenge_counts,
            "held_out_records_scored": 0,
        },
        "data_access": {"model_forward_passes": 0, "fit_runs": 0, "evaluation_scoring_runs": 0, "v32_evaluation_records_read": 0, "v28_runs": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
