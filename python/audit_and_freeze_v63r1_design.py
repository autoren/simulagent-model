#!/usr/bin/env python3
"""Audit and freeze the one-change V63r1 repeat-pooling repair."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", default="configs/v63r1-repeat-pooling-repair.json")
    parser.add_argument("--plan", default="docs/v63r1-repeat-pooling-repair-plan.md")
    parser.add_argument(
        "--audit", default="outputs/v63r1-repeat-pooling-repair/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v63r1-design-lock.json")
    args = parser.parse_args()
    repair_path = (PROJECT_ROOT / args.repair).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V63r1 repair design already frozen")
    repair = json.loads(repair_path.read_text())
    source_path = (PROJECT_ROOT / repair["sourceV63FailedOutcomeLock"]).resolve()
    source = json.loads(source_path.read_text())
    original_result_path = (PROJECT_ROOT / source["result"]).resolve()
    original_result = json.loads(original_result_path.read_text())
    errors = []
    source_ok = bool(
        not source["qualification_passed"]
        and source["failed_gates"] == ["primary_mean_joint_tv"]
        and source["authorization"]["write_and_audit_narrow_repeat_pooling_repair"]
        and not source["authorization"]["modify_or_rerun_original_v63"]
        and file_sha256(original_result_path) == source["result_sha256"]
    )
    if not source_ok:
        errors.append("V63 failed-outcome source or repair authorization is not intact")
    expected_changes = {
        "poolingRule": "equal_weight_posterior_mixture_of_all_three_frozen_repeats_before_every_exact_benchmark_accuracy_and_degeneracy_metric",
        "metricRows": "one_row_per_record_and_outer_budget_after_pooling",
        "logEvidence": "log_mean_exp_of_the_three_repeat_evidence_estimates",
        "diagnostics": "merge_stream_ESS_ancestry_and_rejuvenation_diagnostics_across_the_unpooled_repeats",
        "repairEvaluationScope": "rerun_only_the_32_exact_benchmark_records_with_the_identical_candidate_seeds_budgets_and_repeats",
    }
    changes_ok = repair["authorizedChanges"] == expected_changes
    if not changes_ok:
        errors.append("V63r1 changes exceed or omit the diagnosed pooling repair")
    original_integrity = bool(
        original_result["access"]["logical_evaluation_attempts"] == 1
        and [name for name, passed in original_result["gate_checks"].items() if not passed]
        == ["primary_mean_joint_tv"]
        and original_result["simulation_based_calibration"]["completed_fraction"] == 1.0
        and original_result["scale_stress"]["completed_fraction"] == 1.0
        and original_result["runtime_crosscheck"]["completed_fraction"] == 1.0
    )
    if not original_integrity:
        errors.append("V63 original reusable non-exact results are incomplete or not immutable")
    forbidden_ok = bool(
        repair["claimBoundary"]["originalV63RemainsFailed"]
        and repair["claimBoundary"]["measurementRepairNotIndependentReplication"]
        and not repair["claimBoundary"]["activeInterventionSelection"]
        and not repair["claimBoundary"]["modelAccess"]
        and "metrics_thresholds_or_decision_hierarchy" in repair["forbiddenChanges"]
        and "random_seeds" in repair["forbiddenChanges"]
        and "outer_or_inner_particle_budgets" in repair["forbiddenChanges"]
    )
    if not forbidden_ok:
        errors.append("V63r1 immutability or claim boundary is incomplete")
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v63r1-design-lock.json",
            "configs/v63r1-evaluation-implementation-lock.json",
            "configs/v63r1-outcome-lock.json",
            "outputs/v63r1-repeat-pooling-repair/evaluation/result.json",
        )
    )
    if not downstream_absent:
        errors.append("V63r1 evaluator or outcome already exists")
    audit = {
        "schema_version": "63r1",
        "experiment": "v63r1_repair_design_audit",
        "passed": not errors,
        "decision": "authorize_v63r1_repair_design_lock" if not errors else "reject_v63r1_repair_design",
        "errors": errors,
        "checks": {
            "failed_v63_source_and_authorization": source_ok,
            "one_diagnosed_pooling_change": changes_ok,
            "original_reusable_results_complete": original_integrity,
            "immutability_and_claim_boundary": forbidden_ok,
            "downstream_absent": downstream_absent,
        },
        "bindings": {
            "repair": str(repair_path.relative_to(PROJECT_ROOT)),
            "repair_sha256": file_sha256(repair_path),
            "plan": str(plan_path.relative_to(PROJECT_ROOT)),
            "plan_sha256": file_sha256(plan_path),
            "source_v63_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
            "source_v63_outcome_lock_sha256": file_sha256(source_path),
            "original_v63_result": str(original_result_path.relative_to(PROJECT_ROOT)),
            "original_v63_result_sha256": file_sha256(original_result_path),
        },
        "data_access": {
            "sealed_population_records_read": 0,
            "repair_candidate_runs": 0,
            "human_record_access_count": 0,
            "model_forward_pass_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "63r1",
        "experiment": "v63r1_design_lock",
        "repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_sha256": file_sha256(repair_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_v63_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v63_outcome_lock_sha256": file_sha256(source_path),
        "original_v63_result": str(original_result_path.relative_to(PROJECT_ROOT)),
        "original_v63_result_sha256": file_sha256(original_result_path),
        "authorization": {
            "modify_v63_or_v63r1_design": False,
            "write_and_audit_repair_evaluator": True,
            "run_one_repair_evaluation": False,
            "active_intervention_selection": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
