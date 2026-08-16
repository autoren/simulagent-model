#!/usr/bin/env python3
"""Run the single sealed V62 external exact-belief/planning evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from audit_v62_implementation import ref_initial_value, ref_terminal
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import (
    ExactPlanner,
    POMDPModel,
    all_positive_observation_beliefs,
    bellman_residual,
    condition_initial,
    fully_observed_oracle_value,
    initial_observation_distribution,
    public_policy_value,
    return_extrema,
    terminal_mask,
    validate_model,
)


def load_model(path: Path) -> POMDPModel:
    payload = json.loads(path.read_text())
    return POMDPModel(
        payload["name"], tuple(payload["states"]), tuple(payload["actions"]),
        tuple(payload["observations"]), float(payload["discount"]),
        np.asarray(payload["initial"]), np.asarray(payload["transition"]),
        np.asarray(payload["observation"]), np.asarray(payload["reward"]),
    )


def bundle_hash_mismatches(bundle: Path, manifest: dict[str, object]) -> int:
    mismatches = 0
    for relative, binding in manifest["files"].items():
        path = bundle / relative
        if (
            not path.is_file()
            or file_sha256(path) != binding["sha256"]
            or path.stat().st_size != binding["bytes"]
        ):
            mismatches += 1
    return mismatches


def bundle_content_hash(manifest: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            {path: binding["sha256"] for path, binding in manifest["files"].items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def hoeffding_radius(low: float, high: float, episodes: int, comparisons: int, alpha: float) -> float:
    return (high - low) * math.sqrt(math.log(2.0 * comparisons / alpha) / (2.0 * episodes))


def exact_record(model: POMDPModel, model_id: str, horizon: int) -> dict[str, object]:
    planner = ExactPlanner(model)
    candidate_value = planner.initial_value(horizon)
    reference_value, reference_actions = ref_initial_value(model, horizon)
    root_memberships = []
    root_actions = {}
    for observation, probability in enumerate(initial_observation_distribution(model)):
        if probability <= 1e-15:
            continue
        belief = condition_initial(model, observation)[0]
        action = planner.decision(belief, horizon).action
        root_actions[str(observation)] = action
        root_memberships.append(action in reference_actions[observation])
    reachable = list(all_positive_observation_beliefs(model, planner, horizon))
    belief_normalized = [abs(float(belief.sum()) - 1.0) <= 1e-12 for belief, _ in reachable]
    residuals = [bellman_residual(model, planner, belief, remaining) for belief, remaining in reachable]
    validations = validate_model(model)
    exact_values = {
        "exact_history": public_policy_value(model, horizon, "exact_history"),
        "observation_only": public_policy_value(model, horizon, "observation_only"),
        "map_collapse": public_policy_value(model, horizon, "map_collapse"),
        "fully_observed_oracle": fully_observed_oracle_value(model, horizon),
        "uniform_random": public_policy_value(model, horizon, "uniform_random"),
    }
    low, high = return_extrema(model, horizon)
    return {
        "model_id": model_id,
        "horizon": horizon,
        "candidate_value": candidate_value,
        "reference_value": reference_value,
        "candidate_reference_value_error": abs(candidate_value - reference_value),
        "root_actions": root_actions,
        "root_optimal_set_membership_rate": sum(root_memberships) / len(root_memberships),
        "reachable_belief_count": len(reachable),
        "reachable_belief_normalization_rate": sum(belief_normalized) / len(belief_normalized),
        "maximum_bellman_residual": max(residuals, default=0.0),
        "terminal_detection_agreement": bool(np.array_equal(terminal_mask(model), ref_terminal(model))),
        "validation": validations,
        "exact_policy_values": exact_values,
        "return_range": [low, high],
    }


def aggregate(
    exact_records: list[dict[str, object]],
    rollout_records: list[dict[str, object]],
    config: dict[str, object],
    integrity: dict[str, int],
    controls: dict[str, float],
) -> dict[str, object]:
    rollout_by_key = {
        (row["model_id"], row["horizon"], row["policy"]): row for row in rollout_records
    }
    comparisons = config["externalRollout"]["comparisons"]
    alpha = config["externalRollout"]["familywiseAlpha"]
    episodes = config["externalRollout"]["episodesPerTaskPolicy"]
    within, excesses = [], []
    for record in exact_records:
        low, high = record["return_range"]
        radius = hoeffding_radius(low, high, episodes, comparisons, alpha)
        for policy in config["externalRollout"]["policies"]:
            rollout = rollout_by_key[(record["model_id"], record["horizon"], policy)]
            error = abs(rollout["mean_return"] - record["exact_policy_values"][policy])
            rollout["exact_value"] = record["exact_policy_values"][policy]
            rollout["absolute_error"] = error
            rollout["simultaneous_radius"] = radius
            rollout["excess_over_simultaneous_bound"] = max(0.0, error - radius)
            within.append(error <= radius)
            excesses.append(max(0.0, error - radius))

    tiger_info = []
    tmaze_gaps = []
    tiger_map_gaps = []
    for record in exact_records:
        if record["model_id"] == "tiger-alt-start" and record["horizon"] >= 3:
            tiger_info.extend(action == 0 for action in record["root_actions"].values())
            tiger_map_gaps.append(
                record["exact_policy_values"]["exact_history"]
                - record["exact_policy_values"]["map_collapse"]
            )
        if record["model_id"].startswith("tmaze"):
            tmaze_gaps.append(
                record["exact_policy_values"]["exact_history"]
                - record["exact_policy_values"]["observation_only"]
            )
    metrics = {
        "completed_task_fraction": len(exact_records) / config["benchmark"]["taskCells"],
        "external_source_binding_rate": controls["external_source_binding_rate"],
        "license_binding_rate": controls["license_binding_rate"],
        "independent_parser_agreement_rate": controls["independent_parser_agreement_rate"],
        "transition_normalization_rate": sum(row["validation"]["transition_normalized"] for row in exact_records) / len(exact_records),
        "observation_normalization_rate": sum(row["validation"]["observation_normalized"] for row in exact_records) / len(exact_records),
        "initial_belief_normalization_rate": sum(row["validation"]["initial_normalized"] for row in exact_records) / len(exact_records),
        "finite_reward_and_discount_rate": sum(row["validation"]["finite_reward_and_discount"] for row in exact_records) / len(exact_records),
        "maximum_transition_array_error": controls["maximum_transition_array_error"],
        "maximum_observation_array_error": controls["maximum_observation_array_error"],
        "maximum_reward_array_error": controls["maximum_reward_array_error"],
        "maximum_initial_belief_error": controls["maximum_initial_belief_error"],
        "maximum_discount_error": controls["maximum_discount_error"],
        "maximum_candidate_reference_value_error": max(row["candidate_reference_value_error"] for row in exact_records),
        "candidate_reference_optimal_set_membership_rate": sum(row["root_optimal_set_membership_rate"] for row in exact_records) / len(exact_records),
        "reachable_belief_normalization_rate": sum(row["reachable_belief_normalization_rate"] for row in exact_records) / len(exact_records),
        "maximum_independent_bellman_residual": max(row["maximum_bellman_residual"] for row in exact_records),
        "terminal_detection_agreement_rate": sum(row["terminal_detection_agreement"] for row in exact_records) / len(exact_records),
        "official_runtime_completed_cell_fraction": len(rollout_records) / config["externalRollout"]["comparisons"],
        "official_runtime_return_within_simultaneous_bound_rate": sum(within) / len(within),
        "maximum_official_runtime_return_excess_over_simultaneous_bound": max(excesses),
        "tiger_information_gathering_action_rate": sum(tiger_info) / len(tiger_info),
        "minimum_tmaze_exact_history_minus_observation_only_value": min(tmaze_gaps),
        "minimum_tiger_exact_history_minus_map_collapse_value": min(tiger_map_gaps),
        "implementation_mutant_kill_rate": controls["implementation_mutant_kill_rate"],
        "analytic_fixture_pass_rate": controls["analytic_fixture_pass_rate"],
        **integrity,
    }
    gates = config["gates"]
    checks = {
        "completed_task_fraction": metrics["completed_task_fraction"] >= gates["minimumCompletedTaskFraction"],
        "external_source_binding_rate": metrics["external_source_binding_rate"] >= gates["minimumExternalSourceBindingRate"],
        "license_binding_rate": metrics["license_binding_rate"] >= gates["minimumLicenseBindingRate"],
        "independent_parser_agreement_rate": metrics["independent_parser_agreement_rate"] >= gates["minimumIndependentParserAgreementRate"],
        "transition_normalization_rate": metrics["transition_normalization_rate"] >= gates["minimumTransitionNormalizationRate"],
        "observation_normalization_rate": metrics["observation_normalization_rate"] >= gates["minimumObservationNormalizationRate"],
        "initial_belief_normalization_rate": metrics["initial_belief_normalization_rate"] >= gates["minimumInitialBeliefNormalizationRate"],
        "finite_reward_and_discount_rate": metrics["finite_reward_and_discount_rate"] >= gates["minimumFiniteRewardAndDiscountRate"],
        "transition_array_error": metrics["maximum_transition_array_error"] <= gates["maximumTransitionArrayError"],
        "observation_array_error": metrics["maximum_observation_array_error"] <= gates["maximumObservationArrayError"],
        "reward_array_error": metrics["maximum_reward_array_error"] <= gates["maximumRewardArrayError"],
        "initial_belief_error": metrics["maximum_initial_belief_error"] <= gates["maximumInitialBeliefError"],
        "discount_error": metrics["maximum_discount_error"] <= gates["maximumDiscountError"],
        "candidate_reference_value_error": metrics["maximum_candidate_reference_value_error"] <= gates["maximumCandidateReferenceValueError"],
        "candidate_reference_optimal_set_membership_rate": metrics["candidate_reference_optimal_set_membership_rate"] >= gates["minimumCandidateReferenceOptimalSetMembershipRate"],
        "reachable_belief_normalization_rate": metrics["reachable_belief_normalization_rate"] >= gates["minimumReachableBeliefNormalizationRate"],
        "independent_bellman_residual": metrics["maximum_independent_bellman_residual"] <= gates["maximumIndependentBellmanResidual"],
        "terminal_detection_agreement_rate": metrics["terminal_detection_agreement_rate"] >= gates["minimumTerminalDetectionAgreementRate"],
        "official_runtime_completed_cell_fraction": metrics["official_runtime_completed_cell_fraction"] >= gates["minimumOfficialRuntimeCompletedCellFraction"],
        "official_runtime_return_within_simultaneous_bound_rate": metrics["official_runtime_return_within_simultaneous_bound_rate"] >= gates["minimumOfficialRuntimeReturnWithinSimultaneousBoundRate"],
        "official_runtime_return_excess": metrics["maximum_official_runtime_return_excess_over_simultaneous_bound"] <= gates["maximumOfficialRuntimeReturnExcessOverSimultaneousBound"],
        "tiger_information_gathering_action_rate": metrics["tiger_information_gathering_action_rate"] >= gates["minimumTigerInformationGatheringActionRate"],
        "tmaze_history_value_gap": metrics["minimum_tmaze_exact_history_minus_observation_only_value"] >= gates["minimumTmazeExactHistoryMinusObservationOnlyValue"],
        "tiger_map_collapse_value_gap": metrics["minimum_tiger_exact_history_minus_map_collapse_value"] >= gates["minimumTigerExactHistoryMinusMapCollapseValue"],
        "implementation_mutant_kill_rate": metrics["implementation_mutant_kill_rate"] >= gates["minimumImplementationMutantKillRate"],
        "analytic_fixture_pass_rate": metrics["analytic_fixture_pass_rate"] >= gates["minimumAnalyticFixturePassRate"],
        "source_bundle_hash_mismatch_count": metrics["source_bundle_hash_mismatch_count"] <= gates["maximumSourceBundleHashMismatchCount"],
        "upstream_source_mutation_count": metrics["upstream_source_mutation_count"] <= gates["maximumUpstreamSourceMutationCount"],
        "tool_version_mismatch_count": metrics["tool_version_mismatch_count"] <= gates["maximumToolVersionMismatchCount"],
        "unexpected_evaluation_attempt_count": metrics["unexpected_evaluation_attempt_count"] <= gates["maximumUnexpectedEvaluationAttemptCount"],
        "human_record_access_count": metrics["human_record_access_count"] <= gates["maximumHumanRecordAccessCount"],
        "model_forward_pass_count": metrics["model_forward_pass_count"] <= gates["maximumModelForwardPassCount"],
    }
    return {"metrics": metrics, "checks": checks, "passed": len(checks) == 32 and all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-lock", default="configs/v62-evaluation-implementation-lock.json")
    parser.add_argument("--runtime-python", default="data/v62-external-pomdp-transfer/runtime/bin/python")
    parser.add_argument("--output-dir", default="outputs/v62-external-pomdp-transfer/evaluation")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.evaluation_lock).resolve()
    runtime_python = PROJECT_ROOT / args.runtime_python
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if attempt_path.exists() or output_dir.exists():
        raise RuntimeError("V62 permits exactly one candidate evaluation")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_v62_candidate_evaluation"]:
        raise RuntimeError("V62 evaluation lock does not authorize the candidate run")
    for path, digest in lock["evaluation_files_sha256"].items():
        if file_sha256(PROJECT_ROOT / path) != digest:
            raise RuntimeError(f"frozen V62 evaluation file changed: {path}")
    seal_path = PROJECT_ROOT / lock["external_bundle_seal"]
    seal = json.loads(seal_path.read_text())
    if file_sha256(seal_path) != lock["external_bundle_seal_sha256"]:
        raise RuntimeError("V62 external bundle seal changed")
    bundle = PROJECT_ROOT / seal["bundle"]
    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        file_sha256(manifest_path) != seal["manifest_sha256"]
        or bundle_content_hash(manifest) != seal["bundle_content_sha256"]
    ):
        raise RuntimeError("V62 external bundle no longer matches its seal")
    implementation = json.loads((PROJECT_ROOT / seal["implementation_lock"]).read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    attempt = {
        "schema_version": 62,
        "experiment": "v62_evaluation_attempt",
        "attempt": 1,
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "external_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "external_bundle_seal_sha256": file_sha256(seal_path),
    }
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    output_dir.mkdir()

    exact_records = []
    models = {}
    cells = []
    cell_index = 0
    for entry in config["benchmark"]["models"]:
        model_id = entry["id"]
        model = load_model(bundle / f"models/{model_id}/model.json")
        models[model_id] = model
        for horizon in entry["horizons"]:
            exact_records.append(exact_record(model, model_id, horizon))
            for policy in config["externalRollout"]["policies"]:
                cells.append({
                    "model_id": model_id,
                    "horizon": horizon,
                    "policy": policy,
                    "episodes": config["externalRollout"]["episodesPerTaskPolicy"],
                    "seed": config["externalRollout"]["seed"] + cell_index * 1009,
                })
                cell_index += 1
    request_path = output_dir / "official-rollout-request.json"
    official_path = output_dir / "official-rollout.json"
    request_path.write_text(json.dumps({"cells": cells}, indent=2, sort_keys=True) + "\n")
    subprocess.run(
        [
            str(runtime_python), str(PROJECT_ROOT / "python/official_v62_rollout.py"),
            "--bundle", str(bundle), "--request", str(request_path), "--output", str(official_path),
        ],
        check=True,
    )
    official = json.loads(official_path.read_text())
    external = config["externalSource"]
    source_mismatches = sum(
        file_sha256(bundle / "source" / path) != digest
        for path, digest in external["files"].items()
    )
    integrity = {
        "source_bundle_hash_mismatch_count": bundle_hash_mismatches(bundle, manifest),
        "upstream_source_mutation_count": source_mismatches,
        "tool_version_mismatch_count": int(official["runtime_versions"] != lock["runtime_versions"]),
        "unexpected_evaluation_attempt_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
    }
    max_errors = seal["maximum_array_errors"]
    controls = {
        "external_source_binding_rate": float(source_mismatches == 0),
        "license_binding_rate": float(file_sha256(bundle / "source/LICENSE") == external["licenseSha256"]),
        "independent_parser_agreement_rate": seal["independent_parser_agreement_rate"],
        "maximum_transition_array_error": max_errors["transition"],
        "maximum_observation_array_error": max_errors["observation"],
        "maximum_reward_array_error": max_errors["reward"],
        "maximum_initial_belief_error": max_errors["initial"],
        "maximum_discount_error": max_errors["discount"],
        "implementation_mutant_kill_rate": implementation["mutation_kill_rate"],
        "analytic_fixture_pass_rate": implementation["analytic_fixture_pass_rate"],
    }
    aggregated = aggregate(exact_records, official["records"], config, integrity, controls)
    result = {
        "schema_version": 62,
        "experiment": "v62_external_classic_pomdp_transfer_result",
        "evaluation_run": 1,
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "external_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "external_bundle_seal_sha256": file_sha256(seal_path),
        "exact_records": exact_records,
        "official_rollout_records": official["records"],
        "official_runtime_source_sha256": official["runtime_source_sha256"],
        "official_runtime_versions": official["runtime_versions"],
        "controls": controls,
        "integrity": integrity,
        "metrics": aggregated["metrics"],
        "qualification": {"passed": aggregated["passed"], "checks": aggregated["checks"]},
        "claim_boundary": config["claimBoundary"],
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(result_path), **aggregated}, indent=2, sort_keys=True))
    if not aggregated["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
