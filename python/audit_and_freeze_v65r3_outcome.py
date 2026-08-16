#!/usr/bin/env python3
"""Independently re-aggregate and freeze the successful V65r3 outcome."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluate_v65r1_eig import aggregate_evaluation, read_jsonl
from evaluate_v65r3_eig import aggregate_implementation_audit
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    evaluator_path = PROJECT_ROOT / "configs/v65r3-evaluation-implementation-lock.json"
    evaluation_dir = PROJECT_ROOT / "outputs/v65r3-synthetic-only-implementation/evaluation"
    attempt_path = evaluation_dir / "attempt.json"
    result_path = evaluation_dir / "result.json"
    raw_path = evaluation_dir / "record-budget-cells.jsonl"
    failure_path = evaluation_dir / "failure.json"
    audit_path = PROJECT_ROOT / "outputs/v65r3-synthetic-only-implementation/outcome-audit.json"
    output_path = PROJECT_ROOT / "configs/v65r3-outcome-lock.json"
    if output_path.exists():
        raise RuntimeError("V65r3 outcome already frozen")
    evaluator = json.loads(evaluator_path.read_text())
    implementation_path = PROJECT_ROOT / evaluator["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design_path = PROJECT_ROOT / implementation["design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    result = json.loads(result_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    rows = read_jsonl(raw_path)
    errors: list[str] = []

    evaluator_payload = {
        key: value for key, value in evaluator.items() if key != "lock_payload_sha256"
    }
    evaluator_ok = bool(
        hashlib.sha256(
            json.dumps(evaluator_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == evaluator["lock_payload_sha256"]
        and evaluator["authorization"]["run_one_immutable_evaluation"]
        and not evaluator["authorization"]["run_additional_evaluation"]
        and not evaluator["authorization"]["reward_planning"]
        and file_sha256(implementation_path) == evaluator["implementation_lock_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator["subset_seal"])
        == evaluator["subset_seal_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator["evaluator_audit"])
        == evaluator["evaluator_audit_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / relative) == digest
            for relative, digest in evaluator["source_sha256"].items()
        )
    )
    if not evaluator_ok:
        errors.append("V65r3 evaluator lock or one-shot authorization binding failed")

    artifact_ok = bool(
        attempt["logical_evaluation_attempt"] == 1
        and attempt["one_shot_authorization_consumed"]
        and attempt["evaluation_implementation_lock_sha256"] == file_sha256(evaluator_path)
        and result["bindings"]["attempt_sha256"] == file_sha256(attempt_path)
        and result["bindings"]["evaluation_implementation_lock_sha256"]
        == file_sha256(evaluator_path)
        and result["record_budget_cells_sha256"] == file_sha256(raw_path)
        and not failure_path.exists()
    )
    if not artifact_ok:
        errors.append("V65r3 attempt, result, raw-cell, or absent-failure binding failed")

    implementation_audit_raw = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    implementation_audit = aggregate_implementation_audit(implementation_audit_raw)
    recomputed = aggregate_evaluation(
        rows, config, implementation_audit, result["access"]
    )
    original_result_checks = dict(result["gate_checks"])
    zero_branch_gate = original_result_checks.pop(
        "registered_exact_zero_identity_branch_count", None
    )
    reaggregation_ok = bool(
        recomputed["passed"]
        and recomputed["failed_gates"] == []
        and recomputed["gate_checks"] == original_result_checks
        and recomputed["by_budget"] == result["by_budget"]
        and recomputed["controls"] == result["controls"]
        and zero_branch_gate is True
    )
    if not reaggregation_ok:
        errors.append("independent raw-cell reaggregation differs from frozen result")

    record_ids = {row["record_id"] for row in rows}
    prefix_counts = Counter(
        (row["record_id"], row["prefix_length"]) for row in rows if row["budget"] == 31
    )
    prefix_records = Counter(prefix for _, prefix in prefix_counts)
    grid_ok = bool(
        len(rows) == 144
        and len(record_ids) == 48
        and prefix_records == Counter({prefix: 8 for prefix in range(6)})
        and len(
            {
                (row["record_id"], row["budget"], repeat["repeat"])
                for row in rows
                for repeat in row["repeat_diagnostics"]
            }
        )
        == 432
    )
    support_count = sum(
        diagnostic["exact_zero_identity_count"]
        for row in rows
        for diagnostic in row["support_diagnostics"]
    )
    support_ok = bool(
        support_count == 9
        and result["repair_diagnostics"]
        == {
            "exact_zero_identity_branch_count": 9,
            "expected_exact_zero_identity_branch_count": 9,
            "positive_support_particle_extinction_count": 0,
        }
    )
    if not grid_ok or not support_ok:
        errors.append("V65r3 record grid, repeat grid, or zero-identity diagnostics failed")

    primary = result["by_budget"]["509"]
    gates_ok = bool(
        result["schema_version"] == "65r3"
        and result["passed"]
        and result["decision"]
        == "authorize_preregistration_of_external_Bayes_adaptive_reward_decisions"
        and result["failed_gates"] == []
        and all(result["gate_checks"].values())
        and primary["absolute_eig_vector_error"]["mean"]
        <= config["gates"]["maximumPrimaryMeanAbsoluteEigVectorErrorNats"]
        and primary["absolute_eig_vector_error"]["q95"]
        <= config["gates"]["maximumPrimaryQ95AbsoluteEigVectorErrorNats"]
        and primary["strict_optimal_membership_rate"]
        >= config["gates"]["minimumPrimaryStrictOptimalSetMembershipRate"]
        and primary["epsilon_optimal_membership_rate"]
        >= config["gates"]["minimumPrimaryEpsilonOptimalMembershipRate"]
        and result["controls"]["detected_or_dominated"]
        >= config["gates"]["minimumControlsDetectedOrDominated"]
    )
    if not gates_ok:
        errors.append("V65r3 result or original noncompensatory gates did not pass")

    access = result["access"]
    access_ok = bool(
        access["logical_evaluation_attempts"] == 1
        and access["subset_public_records_loaded"] == 48
        and access["v64_source_public_records_loaded_during_evaluation"] == 0
        and access["v64_selection_audit_records_loaded"] == 0
        and access["v64_evaluation_records_loaded"] == 0
        and access["truth_field_access_count"] == 0
        and access["realized_outcome_access_before_selection_count"] == 0
        and access["candidate_omission_count"] == 0
        and access["tie_break_violation_count"] == 0
        and access["random_stream_collision_count"] == 0
        and access["human_record_access_count"] == 0
        and access["model_forward_pass_count"] == 0
        and access["adapter_training_run_count"] == 0
        and access["V65r1_evaluation_reruns"] == 0
        and access["V65r2_evaluation_attempts"] == 0
    )
    if not access_ok:
        errors.append("V65r3 one-shot, selection, truth, random-stream, or external access failed")

    checks = {
        "frozen_evaluator_and_one_shot_authorization_binding": evaluator_ok,
        "attempt_result_raw_and_absent_failure_binding": artifact_ok,
        "independent_raw_cell_reaggregation": reaggregation_ok,
        "complete_record_budget_repeat_grid": grid_ok,
        "registered_exact_zero_identity_diagnostics": support_ok,
        "all_original_noncompensatory_gates": gates_ok,
        "one_shot_selection_truth_stream_and_external_access": access_ok,
    }
    audit = {
        "schema_version": "65r3",
        "experiment": "v65r3_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_successful_v65r3_and_authorize_Bayes_adaptive_reward_decision_preregistration_only"
            if not errors and all(checks.values())
            else "reject_v65r3_outcome"
        ),
        "errors": errors,
        "checks": checks,
        "primary_summary": primary,
        "controls_detected_or_dominated": result["controls"]["detected_or_dominated"],
        "repair_diagnostics": result["repair_diagnostics"],
        "runtime_seconds": result["runtime_seconds"],
        "access": access,
        "claim_boundary": {
            "paired_repair_result_not_independent_replication": True,
            "Rao_Blackwellized_known_state_acquisition": True,
            "SMC2_static_identity_theta_posterior": True,
            "pure_nested_particle_predictive_qualified": False,
            "sequential_Bayes_adaptive_reward_decisions_tested": False,
            "formal_policy_verification_tested": False,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "65r3",
        "experiment": "v65r3_successful_outcome_lock",
        "evaluation_implementation_lock": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluator_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "record_budget_cells": str(raw_path.relative_to(PROJECT_ROOT)),
        "record_budget_cells_sha256": file_sha256(raw_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome_auditor": "python/audit_and_freeze_v65r3_outcome.py",
        "outcome_auditor_sha256": file_sha256(Path(__file__).resolve()),
        "decision": "authorize_preregistration_of_external_Bayes_adaptive_reward_decisions",
        "authorization": {
            "modify_or_rerun_v65r1": False,
            "modify_or_continue_v65r2": False,
            "modify_or_rerun_v65r3": False,
            "preregister_external_Bayes_adaptive_reward_decisions": True,
            "run_reward_decision_evaluation_before_preregistration_and_locks": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "decision": lock["decision"],
                "primary_mean_EIG_error": primary["absolute_eig_vector_error"]["mean"],
                "primary_strict_membership": primary["strict_optimal_membership_rate"],
                "primary_mean_regret": primary["selected_eig_regret"]["mean"],
                "controls_detected_or_dominated": audit["controls_detected_or_dominated"],
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
