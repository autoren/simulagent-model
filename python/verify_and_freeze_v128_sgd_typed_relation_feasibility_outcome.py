#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from verify_and_freeze_v127_sgd_typed_constraint_feasibility_outcome import aggregate_checks


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v128-sgd-typed-relation-feasibility-lock.json"
    result_path = PROJECT_ROOT / "outputs/v128-sgd-typed-relation-feasibility/evaluation/result.json"
    doc_path = PROJECT_ROOT / "docs/v128-sgd-typed-relation-feasibility-results.md"
    audit_path = PROJECT_ROOT / "outputs/v128-sgd-typed-relation-feasibility/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v128-sgd-typed-relation-feasibility-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v128_sgd_typed_relation_feasibility_outcome.py"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V128 outcome already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V128 result document first")
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text())
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    independent = aggregate_checks(result, lock["config_payload"]); access = result["access"]
    checks = {
        "lock_and_dependencies_exact": payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies),
        "aggregate_gate_classification_exact": independent == result["summary"]["outcome_gates"] and result["passed"] == all(independent.values()),
        "deterministic_recomputation_recorded": result["deterministic_in_memory_recomputation_exact"],
        "oracle_boundary_and_zero_language_value_model_execution": access["source_archive_read_count"] == 1 and access["automatic_annotation_parse_pass_count"] == 1 and all(value == 0 for key, value in access.items() if key not in {"source_archive_read_count", "automatic_annotation_parse_pass_count"}) and lock["config_payload"]["authorityBoundary"]["annotationsAreUnavailableOracleEvidenceNotRuntimeInputs"] and result["summary"]["actual_execution_count"] == 0,
    }
    passed = all(checks.values())
    audit = {"schema_version": "128-sgd-typed-relation-feasibility-outcome-audit", "experiment": lock["config_payload"]["experiment"], "passed": passed, "checks": checks, "independent_aggregate_gate_checks": independent}
    write_json(audit_path, audit)
    if not passed: raise SystemExit(1)
    experimental_pass = bool(result["passed"])
    paths = {"analysis_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {
        "schema_version": "128-sgd-typed-relation-feasibility-outcome-lock", "experiment": "v128_sgd_typed_relation_feasibility_outcome_lock",
        "outcome": {"passed": True, "audit_pass": True, "experimental_pass": experimental_pass, "decision": result["decision"], "summary": result["summary"]},
        "authorization": {
            "modify_rerun_retune_or_mine_V128": False,
            "preregister_typed_parser_realization_design": experimental_pass,
            "close_annotation_signature_family": not experimental_pass,
            "run_language_model_or_protected": False, "begin_induction_or_richer_planning": False,
            "run_API_training_action_or_execution": False,
        },
    }
    for key, path in paths.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
