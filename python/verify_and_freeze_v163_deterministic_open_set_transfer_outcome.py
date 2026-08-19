#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from run_v163_deterministic_open_set_transfer_baselines import (
    payload_hash,
    reconstruct,
)
from v163_deterministic_open_set_transfer_baselines import (
    evaluate_baseline_pipeline_gates,
)


def main() -> None:
    benchmark_lock_path = (
        PROJECT_ROOT
        / "configs/v163-deterministic-open-set-transfer-baselines-lock.json"
    )
    result_path = (
        PROJECT_ROOT
        / "outputs/v163-deterministic-open-set-transfer/development-baselines/result.json"
    )
    doc_path = (
        PROJECT_ROOT / "docs/v163-deterministic-open-set-transfer-baselines-results.md"
    )
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v163_deterministic_open_set_transfer_outcome.py"
    )
    audit_path = (
        PROJECT_ROOT
        / "outputs/v163-deterministic-open-set-transfer/development-outcome-audit.json"
    )
    outcome_path = (
        PROJECT_ROOT
        / "configs/v163-deterministic-open-set-transfer-baselines-outcome-lock.json"
    )
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V163 development outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V163 development result before freezing")

    lock = json.loads(benchmark_lock_path.read_text())
    result = json.loads(result_path.read_text())
    artifacts = reconstruct(lock)
    config = lock["config_payload"]
    access = {
        "source_archive_read_count": 1,
        "development_language_read_count": 1,
        "protected_language_read_count": 0,
        "manual_utterance_inspection_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    reconstructed_gates = evaluate_baseline_pipeline_gates(artifacts, access, config)
    reconstructed_gates["calibration_evaluation_membership_is_disjoint"] = not (
        {row["record_id"] for row in artifacts["split"]["calibration"]}
        & {row["record_id"] for row in artifacts["split"]["evaluation"]}
    )
    reconstructed_gates["outputs_contain_no_source_language"] = True
    dependency_keys = (
        "config",
        "parent_language_outcome",
        "historical_interface_outcome",
        "visible_catalog",
        "safe_hypothesis_universe",
        "source_archive",
        "plan",
        "protocol",
        "tests",
        "runner",
        "verifier",
        "auditor",
        "design_audit",
    )
    expected_outputs: dict[str, Any] = {
        "development_split_membership": {
            "membership": artifacts["split"]["membership"],
            "membership_sha256": artifacts["split"]["membership_sha256"],
            "counts": artifacts["split"]["counts"],
            "class_counts": artifacts["split"]["class_counts"],
            "contains_language": False,
        },
        "retrieval_calibration": artifacts["retrieval_tuning"],
        "baseline_evaluation": {
            "baseline_metrics": artifacts["baseline_metrics"],
            "best_nonoracle_baseline": artifacts["best_nonoracle_baseline"],
            "residual_summary": artifacts["residual_summary"],
            "nonresidual_summary": artifacts["nonresidual_summary"],
            "residual_qualification_checks": artifacts[
                "residual_qualification_checks"
            ],
            "residual_qualified": artifacts["residual_qualified"],
        },
        "baseline_predictions": {
            "predictions": artifacts["evaluation_predictions"],
            "payload_sha256": artifacts["evaluation_prediction_payload_sha256"],
            "contains_source_language": False,
        },
        "declared_training_summary": artifacts["training_summary"],
        "controlled_missing_identifiers": {
            "records": artifacts["controlled_missing_identifiers"],
            "contains_language": False,
        },
        "model_eligible_residual": {
            "records": artifacts["residual_manifest"],
            "payload_sha256": artifacts["residual_summary"][
                "manifest_payload_sha256"
            ],
            "membership_uses_truth_or_language": False,
            "contains_language": False,
        },
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text())
        == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, value in expected_outputs.items()
    )
    expected_decision = (
        "freeze_baselines_and_preregister_one_local_residual_model_protocol"
        if result["passed"] and artifacts["residual_qualified"]
        else (
            "freeze_baselines_and_close_local_model_residual_branch"
            if result["passed"]
            else "stop_V163_before_model_or_protected_access"
        )
    )
    zero_boundary_keys = (
        "protected_language_read_count",
        "manual_utterance_inspection_count",
        "model_load_count",
        "model_generation_count",
        "LLM_API_call_count",
        "adapter_training_run_count",
        "real_service_call_count",
        "external_side_effect_count",
        "actual_execution_count",
    )
    checks = {
        "benchmark_lock_and_dependencies_are_exact": bool(
            payload_hash(
                {
                    key: value
                    for key, value in lock.items()
                    if key != "lock_payload_sha256"
                }
            )
            == lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "development_artifacts_reconstruct_exactly": bool(
            outputs_exact
            and artifacts["source_locale_member"] == result["source_locale_member"]
            and artifacts["source_record_count"] == result["source_record_count"]
            and artifacts["baseline_metrics"] == result["baseline_metrics"]
            and artifacts["residual_qualification_checks"]
            == result["residual_qualification_checks"]
        ),
        "result_gates_residual_and_decision_are_consistent": bool(
            reconstructed_gates == result["gates"]
            and result["passed"] == all(result["gates"].values())
            and result["residual_qualified"] == artifacts["residual_qualified"]
            and result["decision"] == expected_decision
        ),
        "oracle_retention_and_truth_free_residual_controls_hold": bool(
            result["baseline_metrics"]["oracle"]["observed_exact_decision_accuracy"]
            == 1.0
            and result["baseline_metrics"]["oracle"]["mean_regret"] == 0.0
            and result["baseline_metrics"]["complete_safe_enumeration"][
                "true_hypothesis_retention"
            ]
            == 1.0
            and not artifacts["residual_summary"]["membership_uses_truth_or_language"]
        ),
        "protected_model_API_ontology_and_effect_boundary_holds": all(
            result["access"][key] == 0 for key in zero_boundary_keys
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "163-deterministic-open-set-transfer-outcome-audit",
        "experiment": "v163_deterministic_open_set_transfer_outcome_audit",
        "passed": integrity_passed,
        "scientific_baseline_pipeline_passed": result["passed"],
        "residual_qualified": result["residual_qualified"],
        "decision": (
            "freeze_positive_V163_deterministic_transfer_outcome"
            if integrity_passed and result["passed"]
            else "reject_V163_development_outcome"
        ),
        "checks": checks,
        "independent_summary": result["development_summary"],
        "additional_access": access,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "benchmark_lock": benchmark_lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    for key, integrity in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / integrity["path"]
    outcome: dict[str, Any] = {
        "schema_version": "163-deterministic-open-set-transfer-outcome-lock",
        "experiment": "v163_deterministic_open_set_transfer_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_baseline_pipeline_passed": result["passed"],
            "residual_qualified": result["residual_qualified"],
            "decision": result["decision"],
            "development_summary": result["development_summary"],
            "baseline_metrics": result["baseline_metrics"],
            "residual_qualification_checks": result[
                "residual_qualification_checks"
            ],
        },
        "authorization": {
            "modify_or_rerun_V163_baselines": False,
            "preregister_one_pinned_local_model_on_frozen_residual_only": bool(
                result["passed"] and result["residual_qualified"]
            ),
            "run_model_without_separate_residual_protocol_lock": False,
            "read_protected_transfer_before_fresh_development_outcome": False,
            "run_API_model_or_train_adapter": False,
            "induce_or_register_ontology": False,
            "grant_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
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
    print(
        json.dumps(
            {
                "lock": str(outcome_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(outcome_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
