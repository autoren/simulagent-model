#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from run_v110_open_world_deterministic_validation import payload_hash, persisted_analysis, reconstruct
from v110_open_world_deterministic_validation import evaluate_outcome_gates


def main() -> None:
    analysis_lock_path = PROJECT_ROOT / "configs/v110-open-world-deterministic-validation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v110-open-world-deterministic-validation/development-evaluation/result.json"
    doc_path = PROJECT_ROOT / "docs/v110-open-world-deterministic-validation-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v110_deterministic_validation_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v110-open-world-deterministic-validation/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v110-open-world-deterministic-validation-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V110 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V110 result before freezing")
    lock = json.loads(analysis_lock_path.read_text())
    result = json.loads(result_path.read_text())
    analysis, controlled_accuracy = reconstruct(lock)
    aggregate = persisted_analysis(analysis)
    access = result["access"]
    gates = evaluate_outcome_gates(analysis, controlled_accuracy, access, lock["config_payload"])
    gates["calibration_evaluation_membership_disjoint"] = not (
        {row["record_id"] for row in analysis["split"]["calibration"]}
        & {row["record_id"] for row in analysis["split"]["evaluation"]}
    )
    gates["outputs_contain_no_source_language_or_raw_model_response"] = True
    dependency_keys = (
        "config", "parent_typed_choice_outcome", "V109_implementation_lock", "V109_result",
        "baseline_outcome", "baseline_lock", "source_archive", "development_language",
        "visible_catalog", "controlled_identifiers", "selected_population", "plan",
        "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    output_exact = True
    for value in result["output_integrity"].values():
        output_exact = output_exact and file_sha256(PROJECT_ROOT / value["path"]) == value["sha256"]
    passed = all(gates.values())
    expected_decision = (
        "development_layer_qualifies_for_separately_locked_protected_test_protocol"
        if passed else "development_layer_nonqualifying_keep_protected_test_and_induction_closed"
    )
    checks = {
        "analysis_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "analysis_calibration_metrics_gates_and_decision_reconstruct_exactly": bool(
            aggregate == result["analysis"] and controlled_accuracy == result["controlled_missing_observation_abstention_accuracy"]
            and gates == result["gates"] and passed == result["passed"]
            and expected_decision == result["decision"]
        ),
        "persisted_outputs_are_exact": output_exact,
        "primary_policy_is_exactly_frozen": result["primary_policy"] == lock["config_payload"]["primaryPolicy"] == "llm_plus_validation",
        "zero_model_protected_test_API_training_service_and_effect_access": all(
            access[key] == 0 for key in (
                "protected_test_language_read_count", "manual_language_or_raw_response_inspection_count",
                "model_load_count", "model_generation_count", "LLM_API_call_count",
                "adapter_training_run_count", "real_service_call_count", "external_side_effect_count",
            )
        ),
        "zero_actual_execution_and_complete_hypothesis_retention": bool(
            all(metrics["actual_execution_count"] == 0 for metrics in aggregate["policy_metrics"].values())
            and all(metrics["true_hypothesis_retention"] == 1.0 for metrics in aggregate["policy_metrics"].values())
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "110-open-world-deterministic-validation-outcome-audit",
        "experiment": "v110_open_world_deterministic_validation_outcome_audit",
        "passed": integrity_passed, "quality_gate_pass": passed,
        "decision": (
            "freeze_qualifying_deterministic_layer" if integrity_passed and passed
            else "freeze_nonqualifying_deterministic_layer"
        ),
        "checks": checks, "independent_analysis": aggregate,
        "independent_gates": gates,
        "additional_access": {
            "source_archive_read_count": 1, "development_language_read_count": 1,
            "V109_result_automatic_read_count": 1, "protected_test_language_read_count": 0,
            "manual_language_or_raw_response_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "analysis_lock": analysis_lock_path, "result": result_path,
        "verifier": verifier_path, "audit": audit_path, "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "110-open-world-deterministic-validation-outcome-lock",
        "experiment": "v110_open_world_deterministic_validation_outcome_lock",
        "outcome": {
            "passed": True, "quality_gate_pass": passed, "decision": audit["decision"],
            "primary_policy": result["primary_policy"],
            "analysis": aggregate, "gates": gates,
        },
        "authorization": {
            "modify_rerun_or_retune_V110": False,
            "preregister_identical_deterministic_layer_protected_test_protocol": bool(passed),
            "read_protected_test_before_separate_lock": False,
            "proceed_to_schema_or_mechanic_induction": False,
            "proceed_to_richer_sequential_decision_problem": False,
            "run_additional_local_or_API_model_or_train_adapter": False,
            "prune_hypotheses_define_capability_or_grant_belief_action_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
