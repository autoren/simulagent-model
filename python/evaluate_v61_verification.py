#!/usr/bin/env python3
"""Run the single sealed V61 external probabilistic verification."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v56_verification import run_storm_properties, tool_versions


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def explicit_rows_normalize(directory: Path, expected_states: int) -> bool:
    lines = (directory / "model.tra").read_text().splitlines()
    if not lines or lines[0] != "dtmc":
        return False
    grouped = defaultdict(float)
    states = set()
    for line in lines[1:]:
        source, target, probability = line.split()
        source, target, probability = int(source), int(target), float(probability)
        grouped[source] += probability
        states.update((source, target))
    return (
        states == set(range(expected_states))
        and set(grouped) == states
        and all(abs(value - 1.0) <= 1e-12 for value in grouped.values())
    )


def verify_bundle_hashes(bundle: Path, manifest: dict) -> int:
    mismatches = 0
    for policy in manifest["policies"]:
        for row in policy["files"]:
            path = bundle / row["path"]
            mismatches += int(not (
                path.exists() and path.stat().st_size == row["bytes"]
                and file_sha256(path) == row["sha256"]
            ))
    return mismatches


def source_result_mutation_count(bundle: Path, manifest: dict) -> int:
    bindings = set()
    for row in manifest["policies"]:
        meta = json.loads(
            (bundle / row["directory"] / "model.meta.json").read_text()
        )
        bindings.add((meta["source_result"], meta["source_result_sha256"]))
    return sum(
        file_sha256(PROJECT_ROOT / path) != digest for path, digest in bindings
    )


def evaluate_policy_model(directory: Path, manifest_row: dict) -> dict:
    meta = json.loads((directory / "model.meta.json").read_text())
    storm = run_storm_properties(directory)
    direct = meta["independent_executor"]
    symbolic = meta["symbolic"]
    finite = all(math.isfinite(value) for value in (
        *storm.values(), direct["success_probability"], direct["expected_return"],
        meta["v60_stored_monte_carlo_return"],
    ))
    mc_excess = max(
        0.0,
        abs(direct["expected_return"] - meta["v60_stored_monte_carlo_return"])
        - meta["v60_monte_carlo_simultaneous_radius"],
    )
    binding = meta["source_binding"]
    return {
        "id": manifest_row["id"],
        "task_id": manifest_row["task_id"],
        "record": manifest_row["record"],
        "replicate": manifest_row["replicate"],
        "horizon": manifest_row["horizon"],
        "states": manifest_row["states"],
        "transitions": manifest_row["transitions"],
        "tree_hash_match": binding["tree_hash_match"],
        "root_action_match": binding["root_action_match"],
        "search_metadata_match": binding["search_metadata_match"],
        "exact_belief_normalized": abs(meta["exact_belief_weight_sum"] - 1.0) <= 1e-10,
        "symbolic": symbolic,
        "storm": storm,
        "independent_executor": direct,
        "termination_probability_error": abs(storm["termination_probability"] - 1.0),
        "success_probability_error": abs(
            storm["success_probability"] - direct["success_probability"]
        ),
        "expected_return_error": abs(
            storm["expected_return"] - direct["expected_return"]
        ),
        "v60_monte_carlo_return": meta["v60_stored_monte_carlo_return"],
        "v60_monte_carlo_exact_error": abs(
            direct["expected_return"] - meta["v60_stored_monte_carlo_return"]
        ),
        "v60_monte_carlo_simultaneous_radius": meta[
            "v60_monte_carlo_simultaneous_radius"
        ],
        "v60_monte_carlo_within_simultaneous_bound": mc_excess == 0.0,
        "v60_monte_carlo_excess_over_simultaneous_bound": mc_excess,
        "transition_distribution_normalized": explicit_rows_normalize(
            directory, manifest_row["states"]
        ),
        "finite": finite,
        "storm_completed": True,
        "completed": True,
        "error": None,
    }


def failed_policy_record(manifest_row: dict, error: Exception) -> dict:
    return {
        "id": manifest_row["id"], "task_id": manifest_row["task_id"],
        "record": manifest_row["record"], "replicate": manifest_row["replicate"],
        "horizon": manifest_row["horizon"], "states": manifest_row["states"],
        "transitions": manifest_row["transitions"],
        "tree_hash_match": False, "root_action_match": False,
        "search_metadata_match": False, "exact_belief_normalized": False,
        "symbolic": {
            "invariant_checks": 0, "invariant_passes": 0,
            "support_checks": 0, "support_passes": 0,
            "probability_passes": 0, "maximum_probability_error": math.inf,
            "totality_checks": 0, "totality_passes": 0,
            "deployment_checks": 0, "deployment_passes": 0,
            "z3_unknown_count": 0, "nonterminal_deadlock_count": 1,
            "counterexamples": [],
        },
        "storm": None, "independent_executor": None,
        "termination_probability_error": math.inf,
        "success_probability_error": math.inf,
        "expected_return_error": math.inf,
        "v60_monte_carlo_return": None,
        "v60_monte_carlo_exact_error": math.inf,
        "v60_monte_carlo_simultaneous_radius": 0.0,
        "v60_monte_carlo_within_simultaneous_bound": False,
        "v60_monte_carlo_excess_over_simultaneous_bound": math.inf,
        "transition_distribution_normalized": False,
        "finite": False, "storm_completed": False, "completed": False,
        "error": f"{type(error).__name__}: {error}",
    }


def aggregate(records: list[dict], config: dict, integrity: dict, controls: dict) -> dict:
    gates = config["gates"]
    horizon_counts = {
        str(horizon): sum(row["horizon"] == horizon for row in records)
        for horizon in (3, 5, 7)
    }
    symbolic_checks = {
        field: sum(row["symbolic"][field] for row in records)
        for field in (
            "invariant_checks", "invariant_passes", "support_checks",
            "support_passes", "probability_passes", "totality_checks",
            "totality_passes", "deployment_checks", "deployment_passes",
            "z3_unknown_count", "nonterminal_deadlock_count",
        )
    }
    source = {
        "completed_policy_fraction": mean(float(row["completed"]) for row in records),
        "policy_count": len(records),
        "policy_count_by_horizon": horizon_counts,
        "reconstructed_tree_hash_match_rate": mean(
            float(row["tree_hash_match"]) for row in records
        ),
        "reconstructed_root_action_match_rate": mean(
            float(row["root_action_match"]) for row in records
        ),
        "reconstructed_search_metadata_match_rate": mean(
            float(row["search_metadata_match"]) for row in records
        ),
        "exact_root_belief_normalization_rate": mean(
            float(row["exact_belief_normalized"]) for row in records
        ),
    }
    totality_denominator = (
        symbolic_checks["totality_checks"] + symbolic_checks["deployment_checks"]
    )
    symbolic = {
        "reachable_state_invariant_proof_rate": (
            symbolic_checks["invariant_passes"] / symbolic_checks["invariant_checks"]
            if symbolic_checks["invariant_checks"] else 0.0
        ),
        "reachable_state_invariant_checks": symbolic_checks["invariant_checks"],
        "reachable_transition_support_equivalence_proof_rate": (
            min(symbolic_checks["support_passes"], symbolic_checks["probability_passes"])
            / symbolic_checks["support_checks"]
            if symbolic_checks["support_checks"] else 0.0
        ),
        "reachable_transition_support_checks": symbolic_checks["support_checks"],
        "policy_observation_totality_rate": (
            (symbolic_checks["totality_passes"] + symbolic_checks["deployment_passes"])
            / totality_denominator if totality_denominator else 0.0
        ),
        "policy_observation_totality_checks": totality_denominator,
        "nonterminal_deadlock_count": symbolic_checks["nonterminal_deadlock_count"],
        "z3_unknown_count": symbolic_checks["z3_unknown_count"],
    }
    probabilistic = {
        "storm_completed_model_fraction": mean(
            float(row["storm_completed"]) for row in records
        ),
        "maximum_termination_probability_error": max(
            (row["termination_probability_error"] for row in records), default=math.inf
        ),
        "maximum_success_probability_error_against_independent_executor": max(
            (row["success_probability_error"] for row in records), default=math.inf
        ),
        "maximum_expected_return_error_against_independent_executor": max(
            (row["expected_return_error"] for row in records), default=math.inf
        ),
        "v60_monte_carlo_return_within_simultaneous_bound_rate": mean(
            float(row["v60_monte_carlo_within_simultaneous_bound"])
            for row in records
        ),
        "maximum_v60_monte_carlo_return_excess_over_simultaneous_bound": max(
            (row["v60_monte_carlo_excess_over_simultaneous_bound"] for row in records),
            default=math.inf,
        ),
        "maximum_v60_monte_carlo_exact_error": max(
            (row["v60_monte_carlo_exact_error"] for row in records), default=math.inf
        ),
        "transition_distribution_normalization_rate": mean(
            float(row["transition_distribution_normalized"]) for row in records
        ),
        "finite_result_rate": mean(float(row["finite"]) for row in records),
        "verified_exact_return_by_horizon": {
            str(horizon): {
                "mean": mean(
                    row["independent_executor"]["expected_return"]
                    for row in records
                    if row["horizon"] == horizon and row["independent_executor"]
                ),
                "minimum": min(
                    (
                        row["independent_executor"]["expected_return"]
                        for row in records
                        if row["horizon"] == horizon and row["independent_executor"]
                    ), default=math.inf,
                ),
                "maximum": max(
                    (
                        row["independent_executor"]["expected_return"]
                        for row in records
                        if row["horizon"] == horizon and row["independent_executor"]
                    ), default=-math.inf,
                ),
            }
            for horizon in (3, 5, 7)
        },
    }
    metrics = {
        "source_binding": source,
        "symbolic": symbolic,
        "probabilistic": probabilistic,
        "integrity": integrity,
        "implementation_controls": {
            "implementation_mutant_kill_rate": controls["mutation_kill_rate"],
            "analytic_fixture_pass_rate": controls["analytic_fixture_pass_rate"],
        },
    }
    checks = {
        "completed_policy_fraction": source["completed_policy_fraction"]
        >= gates["minimumCompletedPolicyFraction"],
        "policy_count": source["policy_count"] >= gates["minimumPolicyCount"],
        "policy_count_per_horizon": min(horizon_counts.values())
        >= gates["minimumPolicyCountPerHorizon"],
        "reconstructed_tree_hash_match_rate": source[
            "reconstructed_tree_hash_match_rate"
        ] >= gates["minimumReconstructedTreeHashMatchRate"],
        "reconstructed_root_action_match_rate": source[
            "reconstructed_root_action_match_rate"
        ] >= gates["minimumReconstructedRootActionMatchRate"],
        "reconstructed_search_metadata_match_rate": source[
            "reconstructed_search_metadata_match_rate"
        ] >= gates["minimumReconstructedSearchMetadataMatchRate"],
        "exact_root_belief_normalization_rate": source[
            "exact_root_belief_normalization_rate"
        ] >= gates["minimumExactRootBeliefNormalizationRate"],
        "reachable_state_invariant_proof_rate": symbolic[
            "reachable_state_invariant_proof_rate"
        ] >= gates["minimumReachableStateInvariantProofRate"],
        "reachable_transition_support_equivalence_proof_rate": symbolic[
            "reachable_transition_support_equivalence_proof_rate"
        ] >= gates["minimumReachableTransitionSupportEquivalenceProofRate"],
        "policy_observation_totality_rate": symbolic[
            "policy_observation_totality_rate"
        ] >= gates["minimumPolicyObservationTotalityRate"],
        "nonterminal_deadlock_count": symbolic["nonterminal_deadlock_count"]
        <= gates["maximumNonterminalDeadlockCount"],
        "z3_unknown_count": symbolic["z3_unknown_count"]
        <= gates["maximumZ3UnknownCount"],
        "storm_completed_model_fraction": probabilistic[
            "storm_completed_model_fraction"
        ] >= gates["minimumStormCompletedModelFraction"],
        "termination_probability_error": probabilistic[
            "maximum_termination_probability_error"
        ] <= gates["maximumTerminationProbabilityError"],
        "success_probability_error": probabilistic[
            "maximum_success_probability_error_against_independent_executor"
        ] <= gates["maximumSuccessProbabilityErrorAgainstIndependentExecutor"],
        "expected_return_error": probabilistic[
            "maximum_expected_return_error_against_independent_executor"
        ] <= gates["maximumExpectedReturnErrorAgainstIndependentExecutor"],
        "v60_monte_carlo_return_within_simultaneous_bound_rate": probabilistic[
            "v60_monte_carlo_return_within_simultaneous_bound_rate"
        ] >= gates["minimumV60MonteCarloReturnWithinSimultaneousBoundRate"],
        "v60_monte_carlo_return_excess_over_simultaneous_bound": probabilistic[
            "maximum_v60_monte_carlo_return_excess_over_simultaneous_bound"
        ] <= gates["maximumV60MonteCarloReturnExcessOverSimultaneousBound"],
        "transition_distribution_normalization_rate": probabilistic[
            "transition_distribution_normalization_rate"
        ] >= gates["minimumTransitionDistributionNormalizationRate"],
        "finite_result_rate": probabilistic["finite_result_rate"]
        >= gates["minimumFiniteResultRate"],
        "implementation_mutant_kill_rate": controls["mutation_kill_rate"]
        >= gates["minimumImplementationMutantKillRate"],
        "analytic_fixture_pass_rate": controls["analytic_fixture_pass_rate"]
        >= gates["minimumAnalyticFixturePassRate"],
        "truth_field_access_count": integrity["truth_field_access_count"]
        <= gates["maximumTruthFieldAccessCount"],
        "source_result_mutation_count": integrity["source_result_mutation_count"]
        <= gates["maximumSourceResultMutationCount"],
        "verification_bundle_hash_mismatch_count": integrity[
            "verification_bundle_hash_mismatch_count"
        ] <= gates["maximumVerificationBundleHashMismatchCount"],
        "tool_version_mismatch_count": integrity["tool_version_mismatch_count"]
        <= gates["maximumToolVersionMismatchCount"],
        "unexpected_verification_attempt_count": integrity[
            "unexpected_verification_attempt_count"
        ] <= gates["maximumUnexpectedVerificationAttemptCount"],
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-lock", default="configs/v61-evaluation-implementation-lock.json"
    )
    parser.add_argument(
        "--output-dir", default="outputs/v61-long-horizon-policy-verification/verification"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.evaluation_lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "verification-attempt.json"
    if attempt_path.exists() or output_dir.exists():
        raise RuntimeError("V61 permits exactly one sealed verification attempt")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_v61_candidate_verification"]:
        raise RuntimeError("V61 evaluation lock does not authorize the run")
    for section in ("evaluation_files_sha256", "frozen_dependencies_sha256"):
        for relative, digest in lock[section].items():
            if file_sha256(PROJECT_ROOT / relative) != digest:
                raise RuntimeError(f"V61 frozen verification input changed: {relative}")
    seal_path = PROJECT_ROOT / lock["verification_bundle_seal"]
    if file_sha256(seal_path) != lock["verification_bundle_seal_sha256"]:
        raise RuntimeError("V61 verification bundle seal changed")
    seal = json.loads(seal_path.read_text())
    manifest_path = PROJECT_ROOT / seal["manifest"]
    if file_sha256(manifest_path) != seal["manifest_sha256"]:
        raise RuntimeError("V61 sealed manifest changed")
    manifest = json.loads(manifest_path.read_text())
    bundle = PROJECT_ROOT / seal["bundle"]
    preflight_mismatches = verify_bundle_hashes(bundle, manifest)
    if preflight_mismatches:
        raise RuntimeError(f"V61 sealed bundle has {preflight_mismatches} hash mismatches")
    implementation = json.loads(
        (PROJECT_ROOT / lock["implementation_lock"]).read_text()
    )
    design = json.loads(
        (PROJECT_ROOT / implementation["design_lock"]).read_text()
    )
    config = design["config_payload"]
    observed_versions = tool_versions()
    attempt = {
        "schema_version": 61,
        "experiment": "v61_verification_attempt",
        "attempt": 1,
        "verification_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "verification_bundle_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "started_unix_seconds": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    started = time.time()
    records = []
    for manifest_row in manifest["policies"]:
        directory = bundle / manifest_row["directory"]
        try:
            record = evaluate_policy_model(directory, manifest_row)
        except Exception as error:
            record = failed_policy_record(manifest_row, error)
        records.append(record)
        print(json.dumps({
            "completed": len(records), "total": len(manifest["policies"]),
            "id": manifest_row["id"], "horizon": manifest_row["horizon"],
            "storm_completed": record["storm_completed"],
            "seconds": time.time() - started,
        }), flush=True)
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    integrity = {
        "truth_field_access_count": manifest["truth_field_access_count"],
        "source_result_mutation_count": source_result_mutation_count(bundle, manifest),
        "verification_bundle_hash_mismatch_count": verify_bundle_hashes(bundle, manifest),
        "tool_version_mismatch_count": int(observed_versions != lock["tool_versions"]),
        "unexpected_verification_attempt_count": 0,
    }
    controls = {
        "mutation_kill_rate": implementation_audit["mutation_kill_rate"],
        "analytic_fixture_pass_rate": implementation_audit[
            "analytic_fixture_pass_rate"
        ],
    }
    aggregated = aggregate(records, config, integrity, controls)
    result = {
        "schema_version": 61,
        "experiment": "v61_bounded_long_horizon_policy_verification",
        "verification_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "verification_bundle_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "verification_run": 1,
        "policy_models": len(manifest["policies"]),
        "v59_audit_truth_records_accessed": 0,
        "records": records,
        "metrics": aggregated["metrics"],
        "qualification": {
            "checks": aggregated["checks"], "passed": aggregated["passed"]
        },
        "runtime_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "checks": aggregated["checks"], "metrics": aggregated["metrics"],
        "passed": aggregated["passed"], "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
