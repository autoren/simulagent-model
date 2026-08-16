#!/usr/bin/env python3
"""Audit V39 implementation without scoring the held-out evaluation."""

from __future__ import annotations

import argparse
import copy
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v39_compiler import build_populations, corpus_hash
from v39_compiler import compile_agent_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v39-declared-language-compiler-lock.json")
    parser.add_argument("--output", default="outputs/v39-declared-language-compiler/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    v32_path = PROJECT_ROOT / "configs/v32-factorized-semantics.json"
    v32 = json.loads(v32_path.read_text())
    errors: list[str] = []
    if not design["authorization"]["write_implementation"]:
        errors.append("V39 design does not authorize implementation")
    v38_path = PROJECT_ROOT / config["sourceV38OutcomeLock"]
    if not v38_path.is_file():
        errors.append("V39 source V38 outcome lock is missing")
    populations = build_populations(config, v32)
    expected_counts = {
        "compiler_development": 360,
        "supported_evaluation": 360,
        "compiler_safety": 240,
        "novel_paraphrase_diagnostic": 50,
    }
    if {name: len(rows) for name, rows in populations.items()} != expected_counts:
        errors.append("V39 population count mismatch")
    development = populations["compiler_development"]
    evaluation = populations["supported_evaluation"]
    dev_evidence = {row["agent_input"]["evidence_text"] for row in development}
    eval_evidence = {row["agent_input"]["evidence_text"] for row in evaluation}
    if dev_evidence & eval_evidence:
        errors.append("V39 development/evaluation evidence overlap")
    dev_cells = {row["oracle_metadata"]["composition_cell"] for row in development}
    eval_cells = {row["oracle_metadata"]["composition_cell"] for row in evaluation}
    if dev_cells != eval_cells or len(eval_cells) != 120:
        errors.append("V39 macro composition cell coverage mismatch")
    for cell in eval_cells:
        dev_pairs = {
            (row["oracle_metadata"]["cue_index"], row["oracle_metadata"]["punctuation"])
            for row in development if row["oracle_metadata"]["composition_cell"] == cell
        }
        eval_pairs = {
            (row["oracle_metadata"]["cue_index"], row["oracle_metadata"]["punctuation"])
            for row in evaluation if row["oracle_metadata"]["composition_cell"] == cell
        }
        if len(dev_pairs) != 3 or len(eval_pairs) != 3 or dev_pairs & eval_pairs:
            errors.append(f"V39 cue/punctuation holdout failed for {cell}")
            break
    # Development-only semantic verification; held-out and safety targets remain unscored.
    development_exact = 0
    for row in development:
        result = compile_agent_input(row["agent_input"])
        development_exact += int(result.get("status") == "ok" and result.get("parse") == row["target"]["parse"])
    if development_exact != len(development):
        errors.append("V39 compiler failed development exactness")
    row = development[0]
    changed = copy.deepcopy(row)
    changed["target"] = {"sentinel": True}
    if compile_agent_input(row["agent_input"]) != compile_agent_input(changed["agent_input"]):
        errors.append("V39 compiler depends on target")
    forbidden = (
        PROJECT_ROOT / "configs/v39-implementation-lock.json",
        PROJECT_ROOT / "data/v39-declared-language-compiler",
        PROJECT_ROOT / "configs/v39-corpus-seal.json",
        PROJECT_ROOT / "outputs/v39-declared-language-compiler/evaluation",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V39 artifact exists before implementation lock")
    audit = {
        "schema_version": 39,
        "experiment": "v39_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v39_implementation_lock" if not errors else "repair_v39_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "source_v38_outcome_lock": str(v38_path.relative_to(PROJECT_ROOT)),
        "source_v38_outcome_lock_sha256": file_sha256(v38_path) if v38_path.is_file() else None,
        "dry_run": {
            "population_counts": expected_counts,
            "development_exact_parse": development_exact / len(development),
            "held_out_evaluation_records_scored": 0,
            "safety_records_scored": 0,
            "macro_composition_cells": len(eval_cells),
            "expected_corpus_sha256": {name: corpus_hash(rows) for name, rows in populations.items()},
        },
        "data_access": {
            "model_forward_passes": 0,
            "fit_runs": 0,
            "evaluation_scoring_runs": 0,
            "v32_evaluation_records_read": 0,
            "v28_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
