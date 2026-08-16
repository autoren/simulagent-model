#!/usr/bin/env python3
"""Independently reaggregate and freeze the successful V67 outcome."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v67_verification import canonical_json


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def main() -> None:
    evaluator_path = PROJECT_ROOT / "configs/v67-evaluation-implementation-lock.json"
    verification_dir = PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/verification"
    attempt_path = verification_dir / "attempt.json"
    raw_path = verification_dir / "policy-results.jsonl"
    result_path = verification_dir / "result.json"
    failure_path = verification_dir / "failure.json"
    audit_path = PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v67-outcome-lock.json"
    if lock_path.exists():
        raise RuntimeError("V67 outcome already frozen")
    evaluator = json.loads(evaluator_path.read_text())
    result = json.loads(result_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    seal_path = PROJECT_ROOT / evaluator["bundle_seal"]
    seal = json.loads(seal_path.read_text())
    manifest_path = PROJECT_ROOT / seal["bundle_manifest"]
    manifest = json.loads(manifest_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    implementation_audit_path = PROJECT_ROOT / implementation["implementation_audit"]
    implementation_audit = json.loads(implementation_audit_path.read_text())
    design_path = PROJECT_ROOT / implementation["design_lock"]
    design = json.loads(design_path.read_text())
    gates = design["config_payload"]["gates"]
    errors: list[str] = []

    evaluator_payload = {key: value for key, value in evaluator.items() if key != "lock_payload_sha256"}
    evaluator_ok = bool(
        payload_hash(evaluator_payload) == evaluator["lock_payload_sha256"]
        and evaluator["authorization"]["run_exactly_one_verification"]
        and not evaluator["authorization"]["run_additional_verification"]
        and file_sha256(seal_path) == evaluator["bundle_seal_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator["evaluator"])
        == evaluator["evaluator_sha256"]
        and file_sha256(PROJECT_ROOT / evaluator["evaluator_audit"])
        == evaluator["evaluator_audit_sha256"]
    )
    if not evaluator_ok:
        errors.append("V67 evaluator lock or one-shot authorization binding failed")

    artifact_ok = bool(
        attempt["attempt_number"] == 1
        and attempt["evaluation_implementation_lock_sha256"] == file_sha256(evaluator_path)
        and result["attempt_sha256"] == file_sha256(attempt_path)
        and result["policy_results_sha256"] == file_sha256(raw_path)
        and result["bundle_manifest_sha256"] == file_sha256(manifest_path)
        and not failure_path.exists()
    )
    if not artifact_ok:
        errors.append("V67 attempt, raw result, manifest, or absent-failure binding failed")

    counts = Counter(row["policy_kind"] for row in rows)
    by_record: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_record.setdefault(str(row["record_id"]), {})[str(row["policy_kind"])] = row
    census_ok = bool(
        len(rows) == 96
        and len({row["policy_id"] for row in rows}) == 96
        and counts == Counter({"exact_policy": 48, "pooled_SMC2_policy": 48})
        and len(by_record) == 48
        and all(set(value) == {"exact_policy", "pooled_SMC2_policy"} for value in by_record.values())
    )
    if not census_ok:
        errors.append("V67 policy or paired-record census is incomplete")

    frozen_pairs = {
        str(row["record_id"]): float(row["frozen_V66_exact_minus_SMC2_value"])
        for row in manifest["records"]
    }
    pair_errors = [
        abs(
            float(value["exact_policy"]["Storm_expected_return"])
            - float(value["pooled_SMC2_policy"]["Storm_expected_return"])
            - frozen_pairs[record_id]
        )
        for record_id, value in by_record.items()
    ]
    invariant_passes = sum(
        row["reachable_checks"]["node_invariants"]
        == row["reachable_checks"]["node_invariant_passes"] for row in rows
    )
    totality_passes = sum(
        row["reachable_checks"]["branch_totality_checks"]
        == row["reachable_checks"]["branch_totality_passes"] for row in rows
    )
    normalization_passes = sum(
        row["reachable_checks"]["transition_normalization_checks"]
        == row["reachable_checks"]["transition_normalization_passes"] for row in rows
    )
    metrics = {
        "completed_policy_fraction": len(rows) / 96,
        "policy_count": len(rows),
        "policy_count_by_kind": dict(counts),
        "source_policy_hash_match_rate": sum(row["policy_hash_match"] for row in rows) / 96,
        "source_record_binding_rate": len(by_record) / 48,
        "exact_root_belief_normalization_rate": sum(
            abs(float(row["exact_root_belief_mass"]) - 1.0) <= 1e-10 for row in rows
        ) / 96,
        "reachable_policy_node_invariant_rate": invariant_passes / 96,
        "positive_observation_branch_totality_rate": totality_passes / 96,
        "transition_distribution_normalization_rate": normalization_passes / 96,
        "nonterminal_deadlock_count": sum(
            int(row["reachable_checks"]["nonterminal_deadlocks"]) for row in rows
        ),
        "finite_result_rate": sum(bool(row["finite"]) for row in rows) / 96,
        "maximum_independent_executor_error_against_frozen_V66_value": max(
            float(row["absolute_independent_error_against_frozen_V66"]) for row in rows
        ),
        "maximum_Storm_termination_probability_error": max(
            float(row["absolute_Storm_termination_error"]) for row in rows
        ),
        "maximum_Storm_return_error_against_independent_executor": max(
            float(row["absolute_Storm_return_error_against_independent"]) for row in rows
        ),
        "maximum_reproduced_exact_minus_SMC2_pair_error_against_V66": max(pair_errors),
        "implementation_mutant_kill_rate": float(
            implementation_audit["checks"]["implementation_mutant_kill_rate"]
        ),
        "analytic_fixture_pass_rate": float(
            implementation_audit["checks"]["analytic_fixture_pass_rate"]
        ),
        "verification_bundle_hash_mismatch_count": sum(
            int(row["sealed_file_hash_mismatch_count"]) for row in rows
        ),
        "source_result_mutation_count": 0,
        "tool_version_mismatch_count": int(attempt["Storm_version"] != "1.13.0"),
        "unexpected_verification_attempt_count": 0,
        "truth_field_access_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    metrics_ok = all(
        abs(float(metrics[key]) - float(result["metrics"][key])) <= 1e-15
        for key in metrics
        if key != "policy_count_by_kind"
    ) and metrics["policy_count_by_kind"] == result["metrics"]["policy_count_by_kind"]
    if not metrics_ok:
        errors.append("independent V67 raw-result reaggregation differs from the frozen result")

    independent_gates = {
        "complete": metrics["completed_policy_fraction"] >= gates["minimumCompletedPolicyFraction"],
        "census": metrics["policy_count"] >= gates["minimumPolicyCount"]
        and all(counts[kind] >= gates["minimumPolicyCountPerKind"] for kind in counts),
        "bindings": metrics["source_policy_hash_match_rate"]
        >= gates["minimumSourcePolicyHashMatchRate"]
        and metrics["source_record_binding_rate"] >= gates["minimumSourceRecordBindingRate"],
        "belief_and_reachable_checks": (
            metrics["exact_root_belief_normalization_rate"]
            >= gates["minimumExactRootBeliefNormalizationRate"]
            and metrics["reachable_policy_node_invariant_rate"]
            >= gates["minimumReachablePolicyNodeInvariantRate"]
            and metrics["positive_observation_branch_totality_rate"]
            >= gates["minimumPositiveObservationBranchTotalityRate"]
            and metrics["transition_distribution_normalization_rate"]
            >= gates["minimumTransitionDistributionNormalizationRate"]
            and metrics["nonterminal_deadlock_count"] <= gates["maximumNonterminalDeadlockCount"]
        ),
        "numeric_reproduction": (
            metrics["finite_result_rate"] >= gates["minimumFiniteResultRate"]
            and metrics["maximum_independent_executor_error_against_frozen_V66_value"]
            <= gates["maximumIndependentExecutorErrorAgainstFrozenV66Value"]
            and metrics["maximum_Storm_termination_probability_error"]
            <= gates["maximumStormTerminationProbabilityError"]
            and metrics["maximum_Storm_return_error_against_independent_executor"]
            <= gates["maximumStormReturnErrorAgainstIndependentExecutor"]
            and metrics["maximum_reproduced_exact_minus_SMC2_pair_error_against_V66"]
            <= gates["maximumReproducedExactMinusSMC2PairErrorAgainstV66"]
        ),
        "implementation_controls": (
            metrics["implementation_mutant_kill_rate"] >= gates["minimumImplementationMutantKillRate"]
            and metrics["analytic_fixture_pass_rate"] >= gates["minimumAnalyticFixturePassRate"]
        ),
        "zero_tolerance_integrity_and_access": all(metrics[key] == 0 for key in (
            "verification_bundle_hash_mismatch_count", "source_result_mutation_count",
            "tool_version_mismatch_count", "unexpected_verification_attempt_count",
            "truth_field_access_count", "human_record_access_count",
            "model_forward_pass_count", "adapter_training_run_count",
        )),
    }
    gates_ok = bool(all(independent_gates.values()) and all(result["gate_results"].values()))
    if not gates_ok:
        errors.append("one or more independently rechecked V67 gates failed")

    source_ok = bool(
        file_sha256(manifest_path) == seal["bundle_manifest_sha256"]
        and file_sha256(implementation_path) == seal["implementation_lock_sha256"]
        and file_sha256(implementation_audit_path) == implementation[
            "implementation_audit_sha256"
        ]
        and file_sha256(PROJECT_ROOT / manifest["source_v66_record_cells"])
        == manifest["source_v66_record_cells_sha256"]
        and file_sha256(PROJECT_ROOT / manifest["source_v66_result"])
        == manifest["source_v66_result_sha256"]
        and file_sha256(PROJECT_ROOT / manifest["source_v66_outcome_lock"])
        == manifest["source_v66_outcome_lock_sha256"]
        and file_sha256(PROJECT_ROOT / manifest["public_subset"])
        == manifest["public_subset_sha256"]
        and file_sha256(PROJECT_ROOT / manifest["pinned_source_model"])
        == manifest["pinned_source_model_sha256"]
    )
    if not source_ok:
        errors.append("V67 sealed bundle, source, or implementation binding failed")

    boundary_ok = bool(
        result["qualification_passed"]
        and result["decision"]
        == "qualify_bounded_exact_posterior_execution_of_all_96_frozen_V66_policies"
        and result["claim_boundary"]["policyExecutionNotPlannerAlgorithmVerification"]
        and not any(result["claim_boundary"][key] for key in (
            "plannerOptimality", "infiniteHorizon", "formalSafetyProperty",
            "parameterUniformGuarantee", "independentBenchmarkReplication",
            "humanData", "modelAccess", "adapterTraining",
        ))
    )
    if not boundary_ok:
        errors.append("V67 qualification or claim boundary is invalid")

    checks = {
        "frozen_evaluator_and_one_shot_authorization": evaluator_ok,
        "attempt_raw_manifest_and_absent_failure_binding": artifact_ok,
        "complete_96_policy_paired_census": census_ok,
        "independent_raw_policy_reaggregation": metrics_ok,
        "all_noncompensatory_gates_rechecked": gates_ok,
        "sealed_bundle_source_and_implementation_binding": source_ok,
        "bounded_execution_only_claim_boundary": boundary_ok,
    }
    audit = {
        "schema_version": "67",
        "experiment": "v67_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_successful_V67_and_authorize_preregistration_of_multi_environment_external_replication_only"
            if not errors and all(checks.values()) else "reject_V67_outcome_freeze"
        ),
        "errors": errors,
        "checks": checks,
        "independent_gate_checks": independent_gates,
        "metrics": metrics,
        "access": result["access"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "67",
        "experiment": "v67_successful_outcome_lock",
        "decision": "authorize_preregistration_of_multi_environment_external_replication_only",
        "evaluation_implementation_lock": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluator_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "policy_results": str(raw_path.relative_to(PROJECT_ROOT)),
        "policy_results_sha256": file_sha256(raw_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome_auditor": "python/audit_and_freeze_v67_outcome.py",
        "outcome_auditor_sha256": file_sha256(
            PROJECT_ROOT / "python/audit_and_freeze_v67_outcome.py"
        ),
        "authorization": {
            "modify_or_rerun_v66_or_v67": False,
            "preregister_multi_environment_external_replication": True,
            "run_replication_before_preregistration_and_locks": False,
            "claim_planner_optimality": False,
            "claim_infinite_horizon_or_worst_case_safety": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "audit_passed": audit["passed"], "checks": checks,
        "metrics": metrics, "outcome_lock": str(lock_path.relative_to(PROJECT_ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
