#!/usr/bin/env python3
"""Audit and freeze the completed positive V69 development outcome."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import evaluate_v68_development_screen as base

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    evaluator_lock_path = PROJECT_ROOT / "configs/v69-development-evaluator-lock.json"
    attempt_path = PROJECT_ROOT / "outputs/v69-development-screening/evaluation/attempt.json"
    rows_path = PROJECT_ROOT / "outputs/v69-development-screening/evaluation/record-results.jsonl"
    result_path = PROJECT_ROOT / "outputs/v69-development-screening/evaluation/result.json"
    audit_path = PROJECT_ROOT / "outputs/v69-development-screening/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v69-development-outcome-lock.json"
    if lock_path.exists():
        raise RuntimeError("V69 development outcome already frozen")
    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    lock_payload = {
        key: value
        for key, value in evaluator_lock.items()
        if key != "lock_payload_sha256"
    }
    attempt = json.loads(attempt_path.read_text())
    rows = base.read_jsonl(rows_path)
    result = json.loads(result_path.read_text())
    errors: list[str] = []

    evaluator_ok = bool(
        payload_hash(lock_payload) == evaluator_lock["lock_payload_sha256"]
        and evaluator_lock["authorization"]["run_development_screen_once"]
        and not evaluator_lock["authorization"]["score_confirmatory_models"]
        and file_sha256(PROJECT_ROOT / evaluator_lock["evaluator"])
        == evaluator_lock["evaluator_sha256"]
    )
    if not evaluator_ok:
        errors.append("V69 evaluator lock or one-shot authorization failed")

    files_ok = bool(
        attempt["attempt_number"] == evaluator_lock["expected_attempt_number"] == 1
        and attempt["evaluator_lock_sha256"] == file_sha256(evaluator_lock_path)
        and len(rows) == evaluator_lock["expected_records"] == result["records"] == 59
        and result["attempt_sha256"] == file_sha256(attempt_path)
        and result["record_results_sha256"] == file_sha256(rows_path)
    )
    if not files_ok:
        errors.append("V69 attempt, row count, or result hash binding failed")

    census_seal = json.loads(
        (PROJECT_ROOT / evaluator_lock["development_census_seal"]).read_text()
    )
    design_lock = json.loads(
        (PROJECT_ROOT / census_seal["development_design_lock"]).read_text()
    )
    config = design_lock["config_payload"]
    recomputed = base.aggregate_rows(
        rows,
        config,
        expected_record_count=59,
        confirmatory_models_scored=0,
    )
    aggregate_ok = bool(
        recomputed["passed"] == result["passed"]
        and recomputed["decision"] == result["decision"]
        and recomputed["metrics"] == result["metrics"]
        and recomputed["gate_results"] == result["gate_results"]
        and recomputed["by_model"] == result["by_model"]
        and recomputed["full_census_normalized_regret"]
        == result["full_census_normalized_regret"]
    )
    if not aggregate_ok:
        errors.append("V69 independently recomputed aggregate differs")

    positive_ok = bool(
        result["passed"]
        and result["decision"]
        == "authorize_preregistration_of_confirmatory_multi_environment_design_only"
        and all(result["gate_results"].values())
        and len(result["gate_results"]) == 18
        and result["metrics"]["exact_BA_MAP_root_action_disagreement_records"] == 8
        and result["metrics"]["material_regret_record_counts"]["map"] == 8
        and result["metrics"]["material_regret_record_counts"]["posterior_sampling"] == 16
        and result["metrics"]["maximum_normalized_MAP_regret"] >= 0.01
    )
    if not positive_ok:
        errors.append("V69 positive decision or frozen gate result differs")

    firewall_ok = bool(
        result["access"]["confirmatory_models_scored"] == 0
        and result["access"]["records_selected_rejected_or_replaced"] == 0
        and result["access"]["SMC2_runs"] == 0
        and result["access"]["human_records"] == 0
        and attempt["confirmatory_models_scored"] == 0
        and census_seal["confirmatory_models_scored"] == 0
    )
    if not firewall_ok:
        errors.append("V69 holdout or access firewall failed")

    diagnostics_ok = all(
        result["off_support_totalization"][name]["epsilon_smoothing_count"] == 0
        and result["off_support_totalization"][name][
            "model_reselection_or_resampling_count"
        ]
        == 0
        for name in ("map", "posterior_sampling")
    )
    if not diagnostics_ok:
        errors.append("V69 point-control totalization diagnostics failed")

    checks = {
        "evaluator_lock_and_one_shot_authorization": evaluator_ok,
        "attempt_rows_and_result_hash_binding": files_ok,
        "independent_aggregate_recomputation": aggregate_ok,
        "all_frozen_gates_pass_and_positive_decision": positive_ok,
        "zero_holdout_and_access_firewall": firewall_ok,
        "point_control_totalization_semantics": diagnostics_ok,
    }
    audit = {
        "schema_version": "69-development-screening",
        "experiment": "v69_development_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_positive_development_result_and_authorize_confirmatory_preregistration_only"
            if not errors
            else "reject_v69_development_outcome"
        ),
        "errors": errors,
        "checks": checks,
        "access": result["access"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "69-development-screening",
        "experiment": "v69_development_outcome_lock",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "record_results": str(rows_path.relative_to(PROJECT_ROOT)),
        "record_results_sha256": file_sha256(rows_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome": {
            "passed": True,
            "decision": result["decision"],
            "records": 59,
            "confirmatory_models_scored": 0,
            "root_action_disagreements": result["metrics"][
                "exact_BA_MAP_root_action_disagreement_records"
            ],
            "material_MAP_regret_records": result["metrics"][
                "material_regret_record_counts"
            ]["map"],
            "maximum_normalized_MAP_regret": result["metrics"][
                "maximum_normalized_MAP_regret"
            ],
        },
        "authorization": {
            "modify_or_rerun_V69": False,
            "preregister_confirmatory_multi_environment_design": True,
            "construct_confirmatory_census_before_design_lock": False,
            "score_confirmatory_models_before_all_new_locks": False,
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
