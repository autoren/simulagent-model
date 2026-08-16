#!/usr/bin/env python3
"""Independently reaggregate and freeze the successful V66 outcome."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_v66_reward import aggregate_evaluation, read_jsonl
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    evaluator_path = PROJECT_ROOT / "configs/v66-evaluation-implementation-lock.json"
    evaluation_dir = PROJECT_ROOT / "outputs/v66-external-bayes-adaptive-reward/evaluation"
    attempt_path = evaluation_dir / "attempt.json"
    result_path = evaluation_dir / "result.json"
    raw_path = evaluation_dir / "record-cells.jsonl"
    failure_path = evaluation_dir / "failure.json"
    audit_path = PROJECT_ROOT / "outputs/v66-external-bayes-adaptive-reward/outcome-audit.json"
    output_path = PROJECT_ROOT / "configs/v66-outcome-lock.json"
    if output_path.exists():
        raise RuntimeError("V66 outcome already frozen")

    evaluator = json.loads(evaluator_path.read_text())
    implementation_path = PROJECT_ROOT / evaluator["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design_path = PROJECT_ROOT / implementation["design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    smc_implementation_path = PROJECT_ROOT / evaluator[
        "source_v65r3_implementation_lock"
    ]
    smc_implementation = json.loads(smc_implementation_path.read_text())
    smc_audit = json.loads(
        (PROJECT_ROOT / smc_implementation["implementation_audit"]).read_text()
    )
    implementation_audit["inherited_smc_shared_stream_detected"] = bool(
        smc_audit["mutation_audit"]["checks"]["share_inner_random_streams"]
    )
    result = json.loads(result_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    rows = read_jsonl(raw_path)
    errors: list[str] = []

    evaluator_payload = {
        key: value for key, value in evaluator.items() if key != "lock_payload_sha256"
    }
    evaluator_ok = bool(
        payload_hash(evaluator_payload) == evaluator["lock_payload_sha256"]
        and evaluator["authorization"]["run_one_immutable_evaluation"]
        and not evaluator["authorization"]["run_additional_evaluation"]
        and not evaluator["authorization"]["formal_verification"]
        and file_sha256(implementation_path) == evaluator["implementation_lock_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator["subset_seal"])
        == evaluator["subset_seal_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator["evaluator_audit"])
        == evaluator["evaluator_audit_sha256"]
        and file_sha256(smc_implementation_path)
        == evaluator["source_v65r3_implementation_lock_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / relative) == digest
            for relative, digest in evaluator["source_sha256"].items()
        )
    )
    if not evaluator_ok:
        errors.append("V66 evaluator lock or one-shot authorization binding failed")

    artifact_ok = bool(
        attempt["logical_evaluation_attempt"] == 1
        and attempt["one_shot_authorization_consumed"]
        and attempt["evaluation_implementation_lock_sha256"]
        == file_sha256(evaluator_path)
        and result["bindings"]["attempt_sha256"] == file_sha256(attempt_path)
        and result["bindings"]["evaluation_implementation_lock_sha256"]
        == file_sha256(evaluator_path)
        and result["record_cells_sha256"] == file_sha256(raw_path)
        and not failure_path.exists()
    )
    if not artifact_ok:
        errors.append("V66 attempt, result, raw-cell, or absent-failure binding failed")

    recomputed = aggregate_evaluation(rows, config, implementation_audit, result["access"])
    reaggregation_ok = bool(
        recomputed["passed"]
        and recomputed["failed_gates"] == []
        and recomputed["decision"]
        == "authorize_independent_bounded_policy_verification_only"
        and recomputed["gate_checks"] == result["gate_checks"]
        and recomputed["primary"] == result["primary"]
        and recomputed["strategy_value_summaries"]
        == result["strategy_value_summaries"]
        and recomputed["controls"] == result["controls"]
        and recomputed["integrity"] == result["integrity"]
        and recomputed["by_prefix_length"] == result["by_prefix_length"]
    )
    if not reaggregation_ok:
        errors.append("independent V66 raw-cell reaggregation differs from frozen result")

    record_ids = [row["record_id"] for row in rows]
    prefix_counts = Counter(int(row["prefix_length"]) for row in rows)
    grid_ok = bool(
        len(rows) == 48
        and len(set(record_ids)) == 48
        and prefix_counts == Counter({prefix: 8 for prefix in range(6)})
        and all(len(row["repeat_diagnostics"]) == 3 for row in rows)
        and sum(len(row["repeat_diagnostics"]) for row in rows) == 144
    )
    if not grid_ok:
        errors.append("V66 record, prefix, or repeat grid is incomplete")

    policy_archive_ok = bool(
        all(
            row["exact_policy"]["horizon"] == 3
            and row["pooled_SMC2_policy"]["horizon"] == 3
            and row["exact_policy"]["selected_action_name"] in ("n", "e", "s", "w")
            and row["pooled_SMC2_policy"]["selected_action_name"]
            in ("n", "e", "s", "w")
            and len(row["exact_policy"]["q_values"]) == 4
            and len(row["pooled_SMC2_policy"]["q_values"]) == 4
            for row in rows
        )
        and all(
            row["persistent_mixture"]["primary_points"] == 32
            and row["persistent_mixture"]["sensitivity_points"] == 64
            and row["persistent_mixture"]["sampled_model_persists_for_full_policy"]
            for row in rows
        )
    )
    if not policy_archive_ok:
        errors.append("V66 frozen policy trees or mixture semantics are incomplete")

    primary = result["primary"]
    scientific_ok = bool(
        result["passed"]
        and result["decision"]
        == "authorize_independent_bounded_policy_verification_only"
        and result["failed_gates"] == []
        and all(result["gate_checks"].values())
        and primary["policy_value_regret"]["mean"]
        <= config["gates"]["maximumMeanSMC2PolicyValueRegret"]
        and primary["policy_value_regret"]["q95"]
        <= config["gates"]["maximumQ95SMC2PolicyValueRegret"]
        and primary["policy_value_regret"]["max"]
        <= config["gates"]["maximumSMC2PolicyValueRegret"]
        and primary["strict_root_optimal_membership_rate"]
        >= config["gates"]["minimumStrictExactRootOptimalSetMembershipRate"]
        and primary["epsilon_root_optimal_membership_rate"]
        >= config["gates"]["minimumEpsilonOptimalRootMembershipRate"]
        and result["controls"]["detected_or_dominated"]
        >= config["gates"]["minimumControlsDetectedOrDominated"]
    )
    if not scientific_ok:
        errors.append("V66 result or original noncompensatory gates did not pass")

    nuance_ok = bool(
        not result["controls"]["detected"]["MAP"]
        and not result["controls"]["detected"]["firstRepeatOnly"]
        and not result["controls"]["detected"][
            "persistentPosteriorSamplingMixture"
        ]
        and result["controls"]["detected"]["myopicExpectedReward"]
        and result["controls"]["detected"]["informationOnlyEIG"]
        and result["controls"]["detected"]["invalidMeanTransition"]
        and result["controls"]["detected"]["sharedRandomStream"]
        and result["controls"]["detected"]["outcomeLeakage"]
    )
    if not nuance_ok:
        errors.append("V66 control interpretation differs from the frozen result")

    access = result["access"]
    access_ok = bool(
        access["logical_evaluation_attempts"] == 1
        and access["subset_public_records_loaded"] == 48
        and access["V64_or_V65_evaluation_result_record_access"] == 0
        and access["truth_field_access_count"] == 0
        and access["candidate_omission_count"] == 0
        and access["tie_break_violation_count"] == 0
        and access["random_stream_collision_count"] == 0
        and access["human_record_access_count"] == 0
        and access["model_forward_pass_count"] == 0
        and access["adapter_training_run_count"] == 0
        and access["V65r3_evaluation_reruns"] == 0
    )
    if not access_ok:
        errors.append("V66 one-shot, truth, random-stream, or external access failed")

    checks = {
        "frozen_evaluator_and_one_shot_authorization_binding": evaluator_ok,
        "attempt_result_raw_and_absent_failure_binding": artifact_ok,
        "independent_raw_cell_reaggregation": reaggregation_ok,
        "complete_record_prefix_and_repeat_grid": grid_ok,
        "frozen_policy_tree_and_persistent_mixture_archive": policy_archive_ok,
        "all_original_noncompensatory_gates_pass": scientific_ok,
        "control_nuance_preserved": nuance_ok,
        "one_shot_truth_random_stream_and_external_access": access_ok,
    }
    audit = {
        "schema_version": "66",
        "experiment": "v66_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_successful_v66_outcome_and_authorize_independent_policy_verification_only"
            if not errors and all(checks.values())
            else "reject_v66_outcome_freeze"
        ),
        "errors": errors,
        "checks": checks,
        "primary": primary,
        "controls": result["controls"],
        "runtime_seconds": result["runtime_seconds"],
        "access": access,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "66",
        "experiment": "v66_successful_outcome_lock",
        "decision": "authorize_independent_bounded_policy_verification_only",
        "evaluation_implementation_lock": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluator_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "record_cells": str(raw_path.relative_to(PROJECT_ROOT)),
        "record_cells_sha256": file_sha256(raw_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome_auditor": "python/audit_and_freeze_v66_outcome.py",
        "outcome_auditor_sha256": file_sha256(
            PROJECT_ROOT / "python/audit_and_freeze_v66_outcome.py"
        ),
        "authorization": {
            "modify_or_rerun_v65r3": False,
            "modify_or_rerun_v66": False,
            "preregister_independent_bounded_policy_verification": True,
            "run_verification_before_preregistration_and_locks": False,
            "infinite_horizon_claim": False,
            "safety_claim": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "checks": checks,
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
                "primary": primary,
                "controls": result["controls"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
