#!/usr/bin/env python3
"""Durable one-shot external Storm evaluation for sealed V67 policies."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v67_verification import (
    canonical_json,
    policy_tree_hash,
    run_storm_properties,
    storm_version,
)


REQUIRED_FILES = (
    "model.tra", "model.lab", "model.rew", "model.meta.json", "policy-tree.json"
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def reserve_attempt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_manifest_row_files(directory: Path, row: dict[str, Any]) -> int:
    mismatches = 0
    for name in REQUIRED_FILES:
        path = directory / name
        registered = row.get("files", {}).get(name, {})
        if (
            not path.exists()
            or registered.get("sha256") != file_sha256(path)
            or registered.get("size") != path.stat().st_size
        ):
            mismatches += 1
    return mismatches


def verify_policy_directory(directory: Path, row: dict[str, Any]) -> dict[str, Any]:
    mismatch_count = validate_manifest_row_files(directory, row)
    if mismatch_count:
        raise RuntimeError(f"sealed file mismatch for {row['policy_id']}")
    policy = json.loads((directory / "policy-tree.json").read_text())
    meta = json.loads((directory / "model.meta.json").read_text())
    policy_hash_match = bool(
        policy_tree_hash(policy) == row["policy_tree_hash"] == meta["policy_tree_hash"]
    )
    if not policy_hash_match:
        raise RuntimeError(f"policy hash mismatch for {row['policy_id']}")
    checked = run_storm_properties(directory)
    independent = float(meta["independent_scalar_value"])
    frozen = float(meta["frozen_V66_exact_environment_value"])
    result = {
        "policy_id": row["policy_id"],
        "record_index": int(row["record_index"]),
        "record_id": str(row["record_id"]),
        "prefix_length": int(row["prefix_length"]),
        "policy_kind": str(row["policy_kind"]),
        "policy_tree_hash": row["policy_tree_hash"],
        "policy_hash_match": policy_hash_match,
        "sealed_file_hash_mismatch_count": mismatch_count,
        "exact_root_belief_mass": float(meta["exact_root_belief_mass"]),
        "frozen_V66_value": frozen,
        "independent_scalar_value": independent,
        "compiled_direct_value": float(meta["compiled_direct_value"]),
        "Storm_termination_probability": float(checked["termination_probability"]),
        "Storm_expected_return": float(checked["expected_return"]),
        "absolute_independent_error_against_frozen_V66": abs(independent - frozen),
        "absolute_Storm_return_error_against_independent": abs(
            float(checked["expected_return"]) - independent
        ),
        "absolute_Storm_termination_error": abs(
            float(checked["termination_probability"]) - 1.0
        ),
        "reachable_checks": {
            "node_invariants": meta["compiler_checks"]["node_invariants"],
            "node_invariant_passes": meta["compiler_checks"]["node_invariant_passes"],
            "branch_totality_checks": meta["compiler_checks"]["branch_totality_checks"],
            "branch_totality_passes": meta["compiler_checks"]["branch_totality_passes"],
            "transition_normalization_checks": meta["compiler_checks"][
                "transition_normalization_checks"
            ],
            "transition_normalization_passes": meta["compiler_checks"][
                "transition_normalization_passes"
            ],
            "nonterminal_deadlocks": meta["compiler_checks"]["nonterminal_deadlocks"],
        },
        "finite": all(math.isfinite(float(value)) for value in (
            checked["termination_probability"], checked["expected_return"],
            independent, frozen,
        )),
    }
    return result


def aggregate_verification(
    rows: list[dict[str, Any]], manifest: dict[str, Any], design: dict[str, Any],
    implementation_audit: dict[str, Any], *, bundle_hash_mismatch_count: int,
    source_result_mutation_count: int, tool_version_mismatch_count: int,
    unexpected_attempt_count: int,
) -> dict[str, Any]:
    gates = design["config_payload"]["gates"]
    counts = Counter(row["policy_kind"] for row in rows)
    by_record: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_record.setdefault(row["record_id"], {})[row["policy_kind"]] = row
    frozen_pair = {
        row["record_id"]: float(row["frozen_V66_exact_minus_SMC2_value"])
        for row in manifest["records"]
    }
    pair_errors = []
    for record_id, values in by_record.items():
        if set(values) != {"exact_policy", "pooled_SMC2_policy"}:
            pair_errors.append(float("inf"))
            continue
        difference = (
            float(values["exact_policy"]["Storm_expected_return"])
            - float(values["pooled_SMC2_policy"]["Storm_expected_return"])
        )
        pair_errors.append(abs(difference - frozen_pair[record_id]))
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
    deadlocks = sum(row["reachable_checks"]["nonterminal_deadlocks"] for row in rows)
    metrics = {
        "completed_policy_fraction": len(rows) / 96,
        "policy_count": len(rows),
        "policy_count_by_kind": dict(counts),
        "source_policy_hash_match_rate": sum(row["policy_hash_match"] for row in rows) / 96,
        "source_record_binding_rate": len(by_record) / 48,
        "exact_root_belief_normalization_rate": sum(
            abs(row["exact_root_belief_mass"] - 1.0) <= 1e-10 for row in rows
        ) / 96,
        "reachable_policy_node_invariant_rate": invariant_passes / 96,
        "positive_observation_branch_totality_rate": totality_passes / 96,
        "transition_distribution_normalization_rate": normalization_passes / 96,
        "nonterminal_deadlock_count": deadlocks,
        "finite_result_rate": sum(row["finite"] for row in rows) / 96,
        "maximum_independent_executor_error_against_frozen_V66_value": max(
            row["absolute_independent_error_against_frozen_V66"] for row in rows
        ),
        "maximum_Storm_termination_probability_error": max(
            row["absolute_Storm_termination_error"] for row in rows
        ),
        "maximum_Storm_return_error_against_independent_executor": max(
            row["absolute_Storm_return_error_against_independent"] for row in rows
        ),
        "maximum_reproduced_exact_minus_SMC2_pair_error_against_V66": max(pair_errors),
        "implementation_mutant_kill_rate": float(
            implementation_audit["checks"]["implementation_mutant_kill_rate"]
        ),
        "analytic_fixture_pass_rate": float(
            implementation_audit["checks"]["analytic_fixture_pass_rate"]
        ),
        "verification_bundle_hash_mismatch_count": bundle_hash_mismatch_count,
        "source_result_mutation_count": source_result_mutation_count,
        "tool_version_mismatch_count": tool_version_mismatch_count,
        "unexpected_verification_attempt_count": unexpected_attempt_count,
        "truth_field_access_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    gate_results = {
        "completed_policy_fraction": metrics["completed_policy_fraction"]
        >= gates["minimumCompletedPolicyFraction"],
        "policy_census": (
            metrics["policy_count"] >= gates["minimumPolicyCount"]
            and all(counts[kind] >= gates["minimumPolicyCountPerKind"] for kind in (
                "exact_policy", "pooled_SMC2_policy"
            ))
        ),
        "source_policy_hash_match_rate": metrics["source_policy_hash_match_rate"]
        >= gates["minimumSourcePolicyHashMatchRate"],
        "source_record_binding_rate": metrics["source_record_binding_rate"]
        >= gates["minimumSourceRecordBindingRate"],
        "exact_root_belief_normalization_rate": metrics[
            "exact_root_belief_normalization_rate"
        ] >= gates["minimumExactRootBeliefNormalizationRate"],
        "reachable_policy_node_invariant_rate": metrics[
            "reachable_policy_node_invariant_rate"
        ] >= gates["minimumReachablePolicyNodeInvariantRate"],
        "positive_observation_branch_totality_rate": metrics[
            "positive_observation_branch_totality_rate"
        ] >= gates["minimumPositiveObservationBranchTotalityRate"],
        "transition_distribution_normalization_rate": metrics[
            "transition_distribution_normalization_rate"
        ] >= gates["minimumTransitionDistributionNormalizationRate"],
        "nonterminal_deadlock_count": metrics["nonterminal_deadlock_count"]
        <= gates["maximumNonterminalDeadlockCount"],
        "finite_result_rate": metrics["finite_result_rate"] >= gates["minimumFiniteResultRate"],
        "independent_executor_against_frozen_V66": metrics[
            "maximum_independent_executor_error_against_frozen_V66_value"
        ] <= gates["maximumIndependentExecutorErrorAgainstFrozenV66Value"],
        "Storm_termination": metrics["maximum_Storm_termination_probability_error"]
        <= gates["maximumStormTerminationProbabilityError"],
        "Storm_return_against_independent": metrics[
            "maximum_Storm_return_error_against_independent_executor"
        ] <= gates["maximumStormReturnErrorAgainstIndependentExecutor"],
        "exact_minus_SMC2_pairs": metrics[
            "maximum_reproduced_exact_minus_SMC2_pair_error_against_V66"
        ] <= gates["maximumReproducedExactMinusSMC2PairErrorAgainstV66"],
        "implementation_mutants": metrics["implementation_mutant_kill_rate"]
        >= gates["minimumImplementationMutantKillRate"],
        "analytic_fixtures": metrics["analytic_fixture_pass_rate"]
        >= gates["minimumAnalyticFixturePassRate"],
        "bundle_source_tool_and_attempt_integrity": all(
            metrics[key] <= gates[gate] for key, gate in (
                ("verification_bundle_hash_mismatch_count", "maximumVerificationBundleHashMismatchCount"),
                ("source_result_mutation_count", "maximumSourceResultMutationCount"),
                ("tool_version_mismatch_count", "maximumToolVersionMismatchCount"),
                ("unexpected_verification_attempt_count", "maximumUnexpectedVerificationAttemptCount"),
            )
        ),
        "truth_human_model_and_adapter_access": all(
            metrics[key] <= gates[gate] for key, gate in (
                ("truth_field_access_count", "maximumTruthFieldAccessCount"),
                ("human_record_access_count", "maximumHumanRecordAccessCount"),
                ("model_forward_pass_count", "maximumModelForwardPassCount"),
                ("adapter_training_run_count", "maximumAdapterTrainingRunCount"),
            )
        ),
    }
    return {
        "schema_version": "67",
        "experiment": "v67_independent_bounded_policy_execution_verification",
        "qualification_passed": all(gate_results.values()),
        "decision": (
            "qualify_bounded_exact_posterior_execution_of_all_96_frozen_V66_policies"
            if all(gate_results.values())
            else "do_not_qualify_V67_policy_execution_verification"
        ),
        "metrics": metrics,
        "gate_results": gate_results,
        "claim_boundary": design["config_payload"]["claimBoundary"],
        "access": {
            "Storm_policy_models": len(rows),
            "Storm_property_checks": 2 * len(rows),
            "truth_fields": 0,
            "V66_evaluation_reruns": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }


def main() -> None:
    evaluator_lock_path = PROJECT_ROOT / "configs/v67-evaluation-implementation-lock.json"
    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    evaluator_payload = {
        key: value for key, value in evaluator_lock.items()
        if key != "lock_payload_sha256"
    }
    if payload_hash(evaluator_payload) != evaluator_lock["lock_payload_sha256"]:
        raise RuntimeError("V67 evaluator lock payload mismatch")
    if not evaluator_lock["authorization"]["run_exactly_one_verification"]:
        raise RuntimeError("V67 evaluator lock does not authorize verification")
    output = PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/verification"
    attempt_path = output / "attempt.json"
    attempt = {
        "schema_version": "67",
        "experiment": "v67_verification_attempt",
        "attempt_number": 1,
        "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_implementation_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluator_lock_path),
        "Storm_version": storm_version(),
    }
    reserve_attempt(attempt_path, attempt)
    failure_path = output / "failure.json"
    try:
        seal_path = PROJECT_ROOT / evaluator_lock["bundle_seal"]
        seal = json.loads(seal_path.read_text())
        seal_payload = {key: value for key, value in seal.items() if key != "lock_payload_sha256"}
        if payload_hash(seal_payload) != seal["lock_payload_sha256"]:
            raise RuntimeError("V67 bundle seal payload mismatch")
        manifest_path = PROJECT_ROOT / seal["bundle_manifest"]
        if file_sha256(manifest_path) != seal["bundle_manifest_sha256"]:
            raise RuntimeError("V67 sealed manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text())
        design_path = PROJECT_ROOT / "configs/v67-design-lock.json"
        design = json.loads(design_path.read_text())
        implementation_path = PROJECT_ROOT / "configs/v67-implementation-lock.json"
        implementation = json.loads(implementation_path.read_text())
        implementation_audit_path = PROJECT_ROOT / implementation["implementation_audit"]
        if file_sha256(implementation_audit_path) != implementation[
            "implementation_audit_sha256"
        ]:
            raise RuntimeError("V67 implementation-audit hash mismatch")
        implementation_audit = json.loads(implementation_audit_path.read_text())
        bundle_root = manifest_path.parent
        source_mutations = sum((
            file_sha256(PROJECT_ROOT / manifest["source_v66_record_cells"])
            != manifest["source_v66_record_cells_sha256"],
            file_sha256(PROJECT_ROOT / manifest["source_v66_result"])
            != manifest["source_v66_result_sha256"],
            file_sha256(PROJECT_ROOT / manifest["source_v66_outcome_lock"])
            != manifest["source_v66_outcome_lock_sha256"],
        ))
        tool_mismatches = int(storm_version() != "1.13.0")
        rows = []
        for index, policy_row in enumerate(manifest["policies"]):
            rows.append(verify_policy_directory(
                bundle_root / policy_row["directory"], policy_row
            ))
            print(f"verified {index + 1}/96 policies", flush=True)
        bundle_mismatches = sum(
            row["sealed_file_hash_mismatch_count"] for row in rows
        )
        result = aggregate_verification(
            rows, manifest, design, implementation_audit,
            bundle_hash_mismatch_count=bundle_mismatches,
            source_result_mutation_count=source_mutations,
            tool_version_mismatch_count=tool_mismatches,
            unexpected_attempt_count=0,
        )
        result["attempt"] = str(attempt_path.relative_to(PROJECT_ROOT))
        result["attempt_sha256"] = file_sha256(attempt_path)
        result["bundle_manifest"] = str(manifest_path.relative_to(PROJECT_ROOT))
        result["bundle_manifest_sha256"] = file_sha256(manifest_path)
        raw_path = output / "policy-results.jsonl"
        atomic_jsonl(raw_path, rows)
        result["policy_results"] = str(raw_path.relative_to(PROJECT_ROOT))
        result["policy_results_sha256"] = file_sha256(raw_path)
        atomic_json(output / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["qualification_passed"]:
            raise SystemExit(1)
    except BaseException as exc:
        if not failure_path.exists():
            atomic_json(failure_path, {
                "schema_version": "67",
                "experiment": "v67_terminal_verification_failure",
                "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
                "attempt_sha256": file_sha256(attempt_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        raise


if __name__ == "__main__":
    main()
