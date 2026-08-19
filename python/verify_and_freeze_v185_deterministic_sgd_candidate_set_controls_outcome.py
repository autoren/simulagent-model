#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v185_deterministic_sgd_candidate_set_controls import DEPENDENCY_KEYS, reconstruct


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v185-deterministic-sgd-candidate-set-controls-lock.json"
    output_root = PROJECT_ROOT / "outputs/v185-deterministic-sgd-candidate-set-controls/evaluation"
    result_path = output_root / "result.json"
    doc_path = PROJECT_ROOT / "docs/v185-deterministic-sgd-candidate-set-controls-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v185_deterministic_sgd_candidate_set_controls_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v185-deterministic-sgd-candidate-set-controls/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v185-deterministic-sgd-candidate-set-controls-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V185 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V185 results before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    controls, controls_audit = reconstruct(lock)
    expected = {
        "split_manifest": controls["split"],
        "shadow_predictions": {"predictions": controls["predictions"], "contains_language_or_authoritative_state": False},
        "residual_identifiers": {"record_ids": controls["residual_ids"], "membership_uses_predictions_only": True},
        "evaluation_summary": controls["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == payload
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, payload in expected.items()
    )
    config = lock["config_payload"]
    expected_decision = (
        config["decisionRule"]["ifEveryEvaluationSafetySelectivityCostAndResidualGatePasses"]
        if controls_audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    zero_keys = (
        "protected_language_read_count", "manual_language_inspection_count", "model_load_count",
        "model_generation_count", "API_call_count", "training_run_count", "ontology_registration_count",
        "trusted_state_mutation_count", "service_call_count", "external_side_effect_count", "actual_execution_count",
    )
    checks = {
        "lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "all_evaluation_outputs_reconstruct_exactly": outputs_exact,
        "summary_gates_and_decision_reconstruct_exactly": bool(
            result["summary"] == controls["summary"]
            and result["evaluation_gates"] == controls_audit["checks"]
            and result["passed"] == controls_audit["passed"]
            and result["decision"] == expected_decision
        ),
        "trusted_answer_and_prediction_only_residual_boundaries_hold": bool(
            controls["summary"]["comparators"]["same_singleton_consensus_then_safe_fallback"]["final_exactness_after_trusted_answers"] == 1.0
            and controls["summary"]["comparators"]["same_singleton_consensus_then_safe_fallback"]["authoritative_hypothesis_retention_rate"] == 1.0
            and controls["summary"]["residual_membership_uses_predictions_only"]
        ),
        "protected_model_authority_and_effect_access_is_zero": all(result["access"][key] == 0 for key in zero_keys),
    }
    verified = all(checks.values())
    outcome_audit = {
        "schema_version": "185-deterministic-SGD-candidate-set-controls-outcome-audit",
        "experiment": config["experiment"],
        "passed": verified,
        "scientific_evaluation_gates_passed": controls_audit["passed"],
        "decision": "freeze_verified_V185_evaluation" if verified else "reject_V185_outcome",
        "checks": checks,
        "independent_summary": controls["summary"],
        "additional_access": {
            "development_language_read_count": 1,
            "protected_language_read_count": 0,
            "manual_language_inspection_count": 0,
            "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, outcome_audit)
    if not verified:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "evaluation_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V184_outcome": PROJECT_ROOT / lock["parent_V184_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "185-deterministic-SGD-candidate-set-controls-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_evaluation_gates_passed": controls_audit["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rerun_recalibrate_or_redefine_V185": False,
            "preregister_one_local_residual_protocol": bool(controls_audit["passed"]),
            "run_model_without_separate_lock": False,
            "read_protected_language": False,
            "use_predictions_as_terminal_authority_register_mutate_call_service_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
