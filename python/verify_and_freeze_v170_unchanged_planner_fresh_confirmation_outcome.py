#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v170_unchanged_planner_fresh_confirmation import DEPENDENCY_KEYS, reconstruct
from v170_unchanged_planner_fresh_confirmation import evaluate_integrity_gates, evaluate_strong_thresholds


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v170-unchanged-planner-fresh-confirmation-lock.json"; result_path = PROJECT_ROOT / "outputs/v170-unchanged-planner-fresh-confirmation/scoring/result.json"
    doc_path = PROJECT_ROOT / "docs/v170-unchanged-planner-fresh-confirmation-results.md"; verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v170_unchanged_planner_fresh_confirmation_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v170-unchanged-planner-fresh-confirmation/outcome-audit.json"; outcome_path = PROJECT_ROOT / "configs/v170-unchanged-planner-fresh-confirmation-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V170 already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V170 results first")
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text()); evaluation = reconstruct(lock); config = lock["config_payload"]
    integrity = evaluate_integrity_gates(evaluation, result["access"], config); strong = evaluate_strong_thresholds(evaluation, config)
    expected = {"case_policy_results": {"cases": evaluation["cases"], "contains_language": False, "shadow_only": True}, "confirmation_summary": evaluation["summary"]}
    outputs_exact = all(json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == value and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"]) == result["output_integrity"][key]["sha256"] for key, value in expected.items())
    passed_integrity = all(integrity.values()); strong_passed = all(strong.values())
    expected_decision = config["decisionRule"]["ifIntegrityAndStrongThresholdsPass"] if passed_integrity and strong_passed else config["decisionRule"]["ifIntegrityPassesButStrongThresholdsFail"] if passed_integrity else config["decisionRule"]["otherwise"]
    checks = {
        "lock_and_dependencies_exact": bool(payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)),
        "cases_and_summary_reconstruct_exactly": bool(outputs_exact and result["summary"] == evaluation["summary"]),
        "integrity_thresholds_and_decision_reconstruct": bool(result["integrity_gates"] == integrity and result["strong_thresholds"] == strong and result["passed"] == passed_integrity and result["strong_confirmation"] == strong_passed and result["decision"] == expected_decision),
        "membership_and_unchanged_planner_boundary_hold": bool(evaluation["summary"]["case_count"] == 58 and result["access"]["formal_fresh_policy_score_count"] == 58),
        "model_authority_and_execution_zero": all(result["access"][key] == 0 for key in ("evaluation_record_count", "manual_judgment_count", "model_load_count", "model_generation_count", "API_call_count", "training_run_count", "ontology_registration_count", "trusted_state_mutation_count", "real_service_call_count", "external_side_effect_count", "actual_execution_count")),
    }
    passed = all(checks.values()); audit = {"schema_version": "170-unchanged-planner-fresh-confirmation-outcome-audit", "experiment": config["experiment"], "passed": passed, "scientific_integrity_passed": result["passed"], "strong_confirmation": result["strong_confirmation"], "decision": "freeze_verified_V170_outcome" if passed else "reject_V170_outcome", "checks": checks, "independent_summary": evaluation["summary"], "additional_access": {"model_load_count": 0, "actual_execution_count": 0}}
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"planner_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path, "parent_V169r1_outcome": PROJECT_ROOT / lock["parent_V169r1_outcome"]}
    for key, item in result["output_integrity"].items(): deps[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {"schema_version": "170-unchanged-planner-fresh-confirmation-outcome-lock", "experiment": config["experiment"], "outcome": {"passed": True, "scientific_integrity_passed": result["passed"], "strong_confirmation": result["strong_confirmation"], "decision": result["decision"], "summary": result["summary"]}, "authorization": {"modify_rerun_or_tune_V170": False, "retain_outcome_without_posthoc_selection": True, "advance_to_stateful_sandbox_confirmation": True, "design_cross_track_integration_now": False, "run_model_register_mutate_real_state_act_or_execute": False}}
    for key, path in deps.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
