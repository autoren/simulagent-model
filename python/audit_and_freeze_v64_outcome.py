#!/usr/bin/env python3
"""Audit and freeze the passing V64 exact active-identification outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default="outputs/v64-external-multi-action-eig/evaluation/result.json"
    )
    parser.add_argument("--summary", default="docs/v64-results.md")
    parser.add_argument(
        "--audit", default="outputs/v64-external-multi-action-eig/post-result-audit.json"
    )
    parser.add_argument("--output", default="configs/v64-outcome-lock.json")
    args = parser.parse_args()
    result_path = (PROJECT_ROOT / args.result).resolve()
    summary_path = (PROJECT_ROOT / args.summary).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V64 outcome already frozen")
    result = json.loads(result_path.read_text())
    evaluator_path = (PROJECT_ROOT / result["bindings"]["evaluation_implementation_lock"]).resolve()
    evaluator = json.loads(evaluator_path.read_text())
    seal_path = (PROJECT_ROOT / evaluator["population_seal"]).resolve()
    seal = json.loads(seal_path.read_text())
    implementation_path = (PROJECT_ROOT / seal["implementation_lock"]).resolve()
    errors: list[str] = []

    bindings_ok = bool(
        file_sha256(evaluator_path)
        == result["bindings"]["evaluation_implementation_lock_sha256"]
        and file_sha256(seal_path) == result["bindings"]["population_seal_sha256"]
        and file_sha256(seal_path) == evaluator["population_seal_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator["evaluator_audit"])
        == evaluator["evaluator_audit_sha256"]
        and file_sha256(implementation_path) == seal["implementation_lock_sha256"]
        and file_sha256(PROJECT_ROOT / seal["population_audit"])
        == seal["population_audit_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / relative) == digest
            for relative, digest in evaluator["source_sha256"].items()
        )
        and all(
            file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"]
            for row in seal["files"].values()
        )
    )
    if not bindings_ok:
        errors.append("V64 result, evaluator, population, or exact implementation binding failed")

    gates_ok = bool(
        result["passed"]
        and all(result["gate_checks"].values())
        and not result["failed_gates"]
        and result["decision"]
        == "authorize_preregistration_of_pooled_three_repeat_SMC2_EIG_stage"
    )
    if not gates_ok:
        errors.append("V64 did not pass every frozen noncompensatory gate")

    selection = result["selection_benchmark"]
    selection_ok = bool(
        selection["records"] == 192
        and selection["candidate_action_comparisons"] == 768
        and selection["optimal_set_membership_rate"] == 1.0
        and selection["maximum_absolute_candidate_eig_error"] <= 1e-10
        and selection["maximum_selected_eig_regret"] <= 1e-10
        and len(selection["distinct_strictly_optimal_actions"]) == 4
        and selection["informative_record_fraction"] >= 0.5
        and selection["mean_oracle_minus_uniform_random_eig"] >= 0.003
        and selection["mean_oracle_minus_fixed_cycle_eig"] >= 0.002
    )
    adaptive = result["paired_adaptive_information"]
    adaptive_ok = bool(
        adaptive["replications"] == 512
        and adaptive["all_trajectories_completed"]
        and adaptive["posterior_normalization_rate"] == 1.0
        and adaptive["budget_8_paired_differences"]["adaptive_minus_fixed"]["normal_lower_95"]
        >= 0.02
        and adaptive["budget_8_paired_differences"]["adaptive_minus_random"]["normal_lower_95"]
        >= 0.02
    )
    sbc = result["adaptive_simulation_based_calibration"]
    sbc_ok = bool(
        sbc["replications"] == 256
        and sbc["post_selection_normalization_rate"] == 1.0
        and sbc["minimum_rank_chi_square_p_value"] >= 0.001
        and sbc["maximum_absolute_rank_bin_z"] <= 4.75
        and sbc["maximum_absolute_coverage_z"] <= 4.75
    )
    if not selection_ok or not adaptive_ok or not sbc_ok:
        errors.append("V64 exact selection, adaptive information, or SBC evidence is inconsistent")

    access = result["access"]
    access_ok = bool(
        access["logical_evaluation_attempts"] == 1
        and access["selection_public_records"] == 192
        and access["selection_audit_records_loaded"] == 0
        and access["adaptive_public_records"] == 512
        and access["SBC_public_records"] == 256
        and all(
            access[key] == 0
            for key in (
                "truth_field_access_count",
                "realized_outcome_access_before_selection_count",
                "candidate_omission_count",
                "tie_break_violation_count",
                "random_stream_collision_count",
                "human_record_access_count",
                "simulated_human_record_count",
                "model_forward_pass_count",
                "adapter_training_run_count",
            )
        )
    )
    boundary = result["claim_boundary"]
    boundary_ok = bool(
        boundary["exact_benchmark_and_acquisition_reference_qualified"]
        and boundary["external_model_arrays_from_POBAX"]
        and boundary["unknown_actuator_family_project_authored"]
        and not boundary["approximate_particle_acquisition_tested"]
        and not boundary["reward_planning_tested"]
        and not boundary["formal_verification_tested"]
        and not boundary["human_or_model_access"]
    )
    if not access_ok or not boundary_ok:
        errors.append("V64 access firewall or claim boundary is invalid")

    controls_ok = bool(
        result["controls"]["detected_or_dominated"] == 6
        and all(
            row["detected_or_dominated"]
            for row in result["controls"]["controls"].values()
        )
    )
    if not controls_ok:
        errors.append("V64 did not detect all registered controls")

    audit = {
        "schema_version": 64,
        "experiment": "v64_post_result_audit",
        "passed": not errors,
        "decision": "freeze_v64_and_authorize_pooled_SMC2_EIG_preregistration_only" if not errors else "reject_v64_outcome",
        "errors": errors,
        "checks": {
            "result_evaluator_population_and_source_bindings": bindings_ok,
            "all_noncompensatory_gates": gates_ok,
            "exact_selection_adaptive_information_and_SBC": selection_ok and adaptive_ok and sbc_ok,
            "all_controls_detected": controls_ok,
            "truth_outcome_human_model_and_claim_firewalls": access_ok and boundary_ok,
        },
        "failed_gates": result["failed_gates"],
        "primary_metrics": {
            "maximum_candidate_eig_error": selection["maximum_absolute_candidate_eig_error"],
            "informative_record_fraction": selection["informative_record_fraction"],
            "mean_oracle_minus_random_eig": selection["mean_oracle_minus_uniform_random_eig"],
            "mean_oracle_minus_fixed_eig": selection["mean_oracle_minus_fixed_cycle_eig"],
            "budget8_adaptive_minus_fixed_lower95": adaptive["budget_8_paired_differences"]["adaptive_minus_fixed"]["normal_lower_95"],
            "budget8_adaptive_minus_random_lower95": adaptive["budget_8_paired_differences"]["adaptive_minus_random"]["normal_lower_95"],
            "SBC_minimum_p": sbc["minimum_rank_chi_square_p_value"],
        },
        "data_access": access,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": 64,
        "experiment": "v64_outcome_lock",
        "qualification_passed": True,
        "decision": result["decision"],
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "evaluation_implementation_lock": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluator_path),
        "authorization": {
            "modify_or_rerun_v64": False,
            "preregister_pooled_three_repeat_SMC2_EIG_stage": True,
            "construct_or_run_SMC2_EIG_population": False,
            "reward_planning": False,
            "formal_verification": False,
            "access_human_data": False,
            "simulate_human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
