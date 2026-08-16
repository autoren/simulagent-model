#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v56_verification import (
    finite_model,
    run_storm_properties,
    tool_versions,
    transition_rows_normalize,
    verify_compiled_model_symbolically,
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def verify_bundle_hashes(bundle: Path, manifest: dict) -> int:
    mismatches = 0
    for policy in manifest["policies"]:
        for row in policy["files"]:
            path = bundle / row["path"]
            matches = (
                path.exists()
                and path.stat().st_size == row["bytes"]
                and file_sha256(path) == row["sha256"]
            )
            mismatches += int(not matches)
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
    model = dict(meta["model"])
    model["registry"] = meta["registry"]
    symbolic = verify_compiled_model_symbolically(model)
    storm = run_storm_properties(directory)
    direct = meta["direct_executor"]
    finite = (
        finite_model(model)
        and all(math.isfinite(value) for value in storm.values())
        and all(math.isfinite(value) for value in direct.values())
    )
    return {
        "cohort": manifest_row["cohort"],
        "id": manifest_row["id"],
        "record": manifest_row["record"],
        "states": manifest_row["states"],
        "transitions": manifest_row["transitions"],
        "reconstructed_root_action_match": (
            meta["frozen_root_action_key"]
            == meta["reconstructed_root_action_key"]
            == manifest_row["frozen_root_action_key"]
        ),
        "reconstructed_root_value_error": max(
            meta["reconstructed_root_value_error"],
            manifest_row["reconstructed_root_value_error"],
        ),
        "symbolic": symbolic,
        "storm": storm,
        "direct_executor": direct,
        "frozen_root_value": meta["frozen_root_value"],
        "independent_policy_value": meta["independent_policy_value"],
        "termination_probability_error": abs(
            storm["termination_probability"] - 1.0
        ),
        "success_probability_error": abs(
            storm["success_probability"] - direct["success_probability"]
        ),
        "expected_return_error_against_frozen_value": abs(
            storm["expected_return"] - meta["frozen_root_value"]
        ),
        "expected_return_error_against_independent_policy_evaluator": abs(
            storm["expected_return"] - meta["independent_policy_value"]
        ),
        "transition_distribution_normalized": transition_rows_normalize(model),
        "finite": finite,
        "storm_completed": True,
        "completed": True,
        "error": None,
    }


def failed_policy_record(manifest_row: dict, error: Exception) -> dict:
    return {
        "cohort": manifest_row["cohort"],
        "id": manifest_row["id"],
        "record": manifest_row["record"],
        "states": manifest_row["states"],
        "transitions": manifest_row["transitions"],
        "reconstructed_root_action_match": False,
        "reconstructed_root_value_error": math.inf,
        "symbolic": {
            "invariant_checks": 0,
            "invariant_passes": 0,
            "support_checks": 0,
            "support_passes": 0,
            "totality_checks": 0,
            "totality_passes": 0,
            "z3_unknown_count": 0,
            "nonterminal_deadlock_count": 1,
            "counterexamples": [],
        },
        "storm": None,
        "direct_executor": None,
        "frozen_root_value": None,
        "independent_policy_value": None,
        "termination_probability_error": math.inf,
        "success_probability_error": math.inf,
        "expected_return_error_against_frozen_value": math.inf,
        "expected_return_error_against_independent_policy_evaluator": math.inf,
        "transition_distribution_normalized": False,
        "finite": False,
        "storm_completed": False,
        "completed": False,
        "error": f"{type(error).__name__}: {error}",
    }


def aggregate(
    records: list[dict], config: dict, integrity: dict, controls: dict
) -> dict:
    gates = config["gates"]
    cohort_counts = {
        cohort: sum(row["cohort"] == cohort for row in records)
        for cohort in ("v55", "v55r1")
    }
    invariant_checks = sum(
        row["symbolic"]["invariant_checks"] for row in records
    )
    invariant_passes = sum(
        row["symbolic"]["invariant_passes"] for row in records
    )
    support_checks = sum(row["symbolic"]["support_checks"] for row in records)
    support_passes = sum(row["symbolic"]["support_passes"] for row in records)
    totality_checks = sum(row["symbolic"]["totality_checks"] for row in records)
    totality_passes = sum(row["symbolic"]["totality_passes"] for row in records)

    source = {
        "completed_policy_fraction": mean([
            float(row["completed"]) for row in records
        ]),
        "policy_count": len(records),
        "policy_count_by_cohort": cohort_counts,
        "reconstructed_root_action_match_rate": mean([
            float(row["reconstructed_root_action_match"]) for row in records
        ]),
        "maximum_reconstructed_root_value_error": max(
            (row["reconstructed_root_value_error"] for row in records),
            default=math.inf,
        ),
    }
    symbolic = {
        "reachable_state_invariant_proof_rate": (
            invariant_passes / invariant_checks if invariant_checks else 0.0
        ),
        "reachable_state_invariant_checks": invariant_checks,
        "reachable_transition_support_equivalence_proof_rate": (
            support_passes / support_checks if support_checks else 0.0
        ),
        "reachable_transition_support_checks": support_checks,
        "policy_observation_totality_rate": (
            totality_passes / totality_checks if totality_checks else 0.0
        ),
        "policy_observation_totality_checks": totality_checks,
        "nonterminal_deadlock_count": sum(
            row["symbolic"]["nonterminal_deadlock_count"] for row in records
        ),
        "z3_unknown_count": sum(
            row["symbolic"]["z3_unknown_count"] for row in records
        ),
    }
    probabilistic = {
        "storm_completed_model_fraction": mean([
            float(row["storm_completed"]) for row in records
        ]),
        "maximum_termination_probability_error": max(
            (row["termination_probability_error"] for row in records),
            default=math.inf,
        ),
        "maximum_success_probability_error_against_direct_executor": max(
            (row["success_probability_error"] for row in records),
            default=math.inf,
        ),
        "maximum_expected_return_error_against_frozen_value": max(
            (
                row["expected_return_error_against_frozen_value"]
                for row in records
            ),
            default=math.inf,
        ),
        "maximum_expected_return_error_against_independent_policy_evaluator": max(
            (
                row[
                    "expected_return_error_against_independent_policy_evaluator"
                ]
                for row in records
            ),
            default=math.inf,
        ),
        "transition_distribution_normalization_rate": mean([
            float(row["transition_distribution_normalized"]) for row in records
        ]),
        "finite_result_rate": mean([
            float(row["finite"]) for row in records
        ]),
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
        "v55_policy_count": cohort_counts["v55"]
        >= gates["minimumV55PolicyCount"],
        "v55r1_policy_count": cohort_counts["v55r1"]
        >= gates["minimumV55r1PolicyCount"],
        "reconstructed_root_action_match_rate": source[
            "reconstructed_root_action_match_rate"
        ] >= gates["minimumReconstructedRootActionMatchRate"],
        "reconstructed_root_value_error": source[
            "maximum_reconstructed_root_value_error"
        ] <= gates["maximumReconstructedRootValueError"],
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
            "maximum_success_probability_error_against_direct_executor"
        ] <= gates["maximumSuccessProbabilityError"],
        "expected_return_error_against_frozen_value": probabilistic[
            "maximum_expected_return_error_against_frozen_value"
        ] <= gates["maximumExpectedReturnErrorAgainstFrozenValue"],
        "expected_return_error_against_independent_policy_evaluator": probabilistic[
            "maximum_expected_return_error_against_independent_policy_evaluator"
        ] <= gates["maximumExpectedReturnErrorAgainstIndependentPolicyEvaluator"],
        "transition_distribution_normalization_rate": probabilistic[
            "transition_distribution_normalization_rate"
        ] >= gates["minimumTransitionDistributionNormalizationRate"],
        "finite_result_rate": probabilistic["finite_result_rate"]
        >= gates["minimumFiniteResultRate"],
        "implementation_mutant_kill_rate": metrics["implementation_controls"][
            "implementation_mutant_kill_rate"
        ] >= gates["minimumImplementationMutantKillRate"],
        "analytic_fixture_pass_rate": metrics["implementation_controls"][
            "analytic_fixture_pass_rate"
        ] >= gates["minimumAnalyticFixturePassRate"],
        "truth_field_access_count": integrity["truth_field_access_count"]
        <= gates["maximumTruthFieldAccessCount"],
        "source_result_mutation_count": integrity[
            "source_result_mutation_count"
        ] <= gates["maximumSourceResultMutationCount"],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-lock", default="configs/v56-evaluation-implementation-lock.json"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v56-symbolic-probabilistic-policy-verification/evaluation",
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.evaluation_lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if attempt_path.exists() or output_dir.exists():
        raise RuntimeError("V56 permits exactly one sealed candidate verification")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_v56_candidate_verification"]:
        raise RuntimeError("V56 evaluation lock does not authorize the run")
    for section in ("evaluation_files_sha256", "frozen_dependencies_sha256"):
        for relative, digest in lock[section].items():
            if file_sha256(PROJECT_ROOT / relative) != digest:
                raise RuntimeError(f"V56 frozen evaluation input changed: {relative}")
    seal_path = PROJECT_ROOT / lock["verification_bundle_seal"]
    if file_sha256(seal_path) != lock["verification_bundle_seal_sha256"]:
        raise RuntimeError("V56 verification bundle seal changed")
    seal = json.loads(seal_path.read_text())
    bundle = PROJECT_ROOT / seal["bundle"]
    manifest_path = PROJECT_ROOT / seal["manifest"]
    if file_sha256(manifest_path) != seal["manifest_sha256"]:
        raise RuntimeError("V56 sealed manifest changed")
    manifest = json.loads(manifest_path.read_text())
    bundle_mismatches = verify_bundle_hashes(bundle, manifest)
    if bundle_mismatches:
        raise RuntimeError("V56 sealed verification bundle changed")
    observed_versions = tool_versions()
    expected_versions = lock["tool_versions"]
    if observed_versions != expected_versions:
        raise RuntimeError("V56 verification toolchain changed")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = {
        "schema_version": 56,
        "experiment": "v56_candidate_verification_attempt",
        "evaluation_run": 1,
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "verification_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "verification_bundle_seal_sha256": file_sha256(seal_path),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    output_dir.mkdir()

    implementation = json.loads(
        (PROJECT_ROOT / lock["implementation_lock"]).read_text()
    )
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    controls = {
        "mutation_kill_rate": implementation_audit["mutation_controls"][
            "kill_rate"
        ],
        "analytic_fixture_pass_rate": mean([
            float(row["passed"])
            for row in implementation_audit["analytic_storm_fixtures"]
        ]),
    }
    config = json.loads(
        (PROJECT_ROOT / implementation["design_lock"]).read_text()
    )["config_payload"]
    integrity = {
        "truth_field_access_count": 0,
        "source_result_mutation_count": source_result_mutation_count(
            bundle, manifest
        ),
        "verification_bundle_hash_mismatch_count": bundle_mismatches,
        "tool_version_mismatch_count": int(observed_versions != expected_versions),
        "unexpected_verification_attempt_count": 0,
    }

    records = []
    progress_path = output_dir / "progress.jsonl"
    started = time.time()
    for index, manifest_row in enumerate(manifest["policies"], start=1):
        directory = bundle / manifest_row["directory"]
        try:
            record = evaluate_policy_model(directory, manifest_row)
        except Exception as error:  # preserve a complete one-shot result
            record = failed_policy_record(manifest_row, error)
        records.append(record)
        with progress_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps({
            "completed": index,
            "total": len(manifest["policies"]),
            "cohort": record["cohort"],
            "id": record["id"],
            "passed_execution": record["completed"],
            "seconds": time.time() - started,
        }, sort_keys=True), flush=True)

    aggregated = aggregate(records, config, integrity, controls)
    result = {
        "schema_version": 56,
        "experiment": "v56_symbolic_and_probabilistic_policy_verification_result",
        "evaluation_run": 1,
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "verification_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "verification_bundle_seal_sha256": file_sha256(seal_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "tool_versions": observed_versions,
        "records": records,
        "metrics": aggregated["metrics"],
        "qualification": {
            "passed": aggregated["passed"],
            "checks": aggregated["checks"],
        },
        "runtime_seconds": time.time() - started,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "qualification": result["qualification"],
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
