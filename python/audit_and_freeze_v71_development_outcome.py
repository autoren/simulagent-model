#!/usr/bin/env python3
"""Independently audit and freeze the one-shot V71 development outcome."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from evaluate_v71_sensor_codebook_development import aggregate_rows, read_jsonl
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    evaluator_lock_path = (
        PROJECT_ROOT / "configs/v71-sensor-codebook-development-evaluator-lock.json"
    )
    result_path = (
        PROJECT_ROOT / "outputs/v71-sensor-codebook/development-evaluation/result.json"
    )
    report_path = PROJECT_ROOT / "docs/v71-sensor-codebook-development-results.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v71_development_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v71-sensor-codebook/development-outcome-audit.json"
    lock_path = (
        PROJECT_ROOT / "configs/v71-sensor-codebook-development-outcome-lock.json"
    )
    if lock_path.exists():
        raise RuntimeError("V71 development outcome is already frozen")

    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    evaluator_payload = {
        key: value
        for key, value in evaluator_lock.items()
        if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    evaluator_ok = bool(
        payload_hash(evaluator_payload) == evaluator_lock["lock_payload_sha256"]
        and evaluator_lock["authorization"]["run_development_outcomes_once"]
        and not evaluator_lock["authorization"][
            "read_protected_confirmation_histories_or_outcomes"
        ]
        and evaluator_lock["expected_records"] == 21
        and evaluator_lock["expected_protected_confirmation_policy_value_count"] == 0
    )
    if not evaluator_ok:
        errors.append("V71 evaluator lock or one-shot development authorization failed")

    seal_path = PROJECT_ROOT / evaluator_lock["development_census_seal"]
    seal = json.loads(seal_path.read_text())
    census_path = PROJECT_ROOT / seal["census"]
    census = read_jsonl(census_path)
    result = json.loads(result_path.read_text())
    attempt_path = PROJECT_ROOT / result["attempt"]
    rows_path = PROJECT_ROOT / result["record_results"]
    attempt = json.loads(attempt_path.read_text())
    rows = read_jsonl(rows_path)
    artifacts_ok = bool(
        file_sha256(seal_path) == evaluator_lock["development_census_seal_sha256"]
        and file_sha256(census_path) == seal["census_sha256"]
        and file_sha256(attempt_path) == result["attempt_sha256"]
        and file_sha256(rows_path) == result["record_results_sha256"]
        and attempt["attempt_number"] == 1
        and attempt["evaluator_lock_sha256"] == file_sha256(evaluator_lock_path)
        and attempt["census_sha256"] == file_sha256(census_path)
        and attempt["protected_confirmation_policy_value_count"] == 0
        and len(census) == len(rows) == result["records"] == 21
        and {row["record_id"] for row in census}
        == {row["record_id"] for row in rows}
    )
    if not artifacts_ok:
        errors.append("V71 attempt, census, rows, result, or one-shot binding failed")

    numerical_ok = bool(
        all(row["all_metrics_finite"] for row in rows)
        and all(row["joint_belief_sum_error"] <= 1e-12 for row in rows)
        and all(row["point_models_on_support"] for row in rows)
        and all(row["point_model_fallback_count"] == 0 for row in rows)
        and all(
            row["exact_bayes_adaptive"]["value"] + 1e-10
            >= control["value"]
            for row in rows
            for control in (
                row["map"],
                row["posterior_sampling"],
                row["open_loop"],
                row["myopic_one_step"],
            )
        )
        and all(
            math.isfinite(value)
            for row in rows
            for value in row["normalized_regrets"].values()
        )
    )
    if not numerical_ok:
        errors.append("V71 exact dominance, normalization, support, or finiteness failed")

    source_lock_path = PROJECT_ROOT / seal["source_lock"]
    source_lock = json.loads(source_lock_path.read_text())
    config = source_lock["config_payload"]
    recomputed = aggregate_rows(
        rows,
        config,
        result["source_validation"],
        expected_records=evaluator_lock["expected_records"],
    )
    aggregate_ok = bool(
        recomputed["passed"] == result["passed"] is False
        and recomputed["decision"] == result["decision"]
        and recomputed["metrics"] == result["metrics"]
        and recomputed["gate_results"] == result["gate_results"]
        and recomputed["by_model"] == result["by_model"]
        and recomputed["full_census_normalized_regret"]
        == result["full_census_normalized_regret"]
    )
    if not aggregate_ok:
        errors.append("independent V71 aggregation does not reproduce the result")

    expected_failed = {
        "minimumModelsWithExactBAMAPRootActionDisagreement",
        "minimumModelsWithMaterialMAPRegret",
        "minimumModelsWithMaterialPosteriorSamplingRegret",
        "minimumMaximumNormalizedMAPRegret",
    }
    failed = {key for key, passed in result["gate_results"].items() if not passed}
    decision_ok = bool(
        failed == expected_failed
        and result["metrics"]["models_with_exact_BA_MAP_root_action_disagreement"]
        == 0
        and result["metrics"]["models_with_material_MAP_regret"] == 0
        and result["metrics"]["models_with_material_posterior_sampling_regret"]
        == 0
        and result["metrics"]["maximum_normalized_MAP_regret"] == 0.0
        and result["decision"]
        == "stop_v71_before_any_protected_confirmation_history_or_outcome"
    )
    if not decision_ok:
        errors.append("V71 failed-gate set or mandatory stop decision drifted")

    access = result["access"]
    firewall_ok = bool(
        access["development_records_evaluated"] == 21
        and access["records_selected_rejected_or_replaced"] == 0
        and access["protected_confirmation_policy_value_count"] == 0
        and access["SMC2_runs"] == 0
        and access["human_records"] == 0
        and access["model_forward_passes"] == 0
        and access["adapter_training_runs"] == 0
        and not (
            PROJECT_ROOT
            / "outputs/v71-sensor-codebook/protected-confirmation-evaluation"
        ).exists()
    )
    if not firewall_ok:
        errors.append("V71 access or protected-confirmation firewall failed")

    report_ok = bool(
        report_path.exists()
        and "stops before protected confirmation" in report_path.read_text()
        and "maximum normalized MAP regret: `0.0`" in report_path.read_text()
        and "must not be opened" in report_path.read_text()
    )
    if not report_ok:
        errors.append("V71 negative-result report does not preserve the stop boundary")

    checks = {
        "locked_one_shot_development_authorization": evaluator_ok,
        "attempt_census_rows_and_result_binding": artifacts_ok,
        "exact_dominance_normalization_finiteness_and_shared_support": numerical_ok,
        "independent_aggregate_and_gate_reproduction": aggregate_ok,
        "exact_four_gate_failure_and_stop_decision": decision_ok,
        "zero_selection_protected_SMC2_human_model_or_adapter_access": firewall_ok,
        "negative_report_and_protected_stop_boundary": report_ok,
    }
    audit = {
        "schema_version": "71-sensor-codebook-development",
        "experiment": "v71_development_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_negative_development_result_and_close_v71_before_confirmation"
            if not errors
            else "reject_v71_development_outcome"
        ),
        "errors": errors,
        "checks": checks,
        "failed_development_gates": sorted(failed),
        "access": access,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "71-sensor-codebook-development",
        "experiment": "v71_sensor_codebook_development_outcome_lock",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "record_results": str(rows_path.relative_to(PROJECT_ROOT)),
        "record_results_sha256": file_sha256(rows_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "outcome_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "outcome_auditor_sha256": file_sha256(auditor_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "report_sha256": file_sha256(report_path),
        "outcome": {
            "passed_development_gates": False,
            "development_models": 3,
            "development_records": 21,
            "models_with_exact_BA_MAP_root_action_disagreement": 0,
            "models_with_material_MAP_regret": 0,
            "models_with_material_posterior_sampling_regret": 0,
            "maximum_normalized_MAP_regret": 0.0,
            "protected_confirmation_policy_value_count": 0,
        },
        "authorization": {
            "modify_or_rerun_V71": False,
            "revise_V71_family_reliability_horizon_models_controls_normalization_or_gates": False,
            "read_V71_protected_confirmation_histories_or_outcomes": False,
            "report_and_synthesize_negative_development_result": True,
            "begin_new_family_only_after_fresh_source_and_preregistration": True,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
