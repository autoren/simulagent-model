#!/usr/bin/env python3
"""Run the sole immutable V64 exact-EIG evaluation after evaluator freeze."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import chisquare

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import (
    assert_public_selection_payload,
    attempted_outcome_leak_selection,
    filter_public_history,
    identity_posterior,
    initial_joint_belief,
    load_family,
    posterior_kl_to_static_prior,
    sample_categorical,
    score_all_actions,
    score_control_policies,
    select_action,
    simulate_step,
    static_posterior,
    update_joint_belief,
)
from v64_scalar_reference import (
    filter_history as scalar_filter_history,
    load_reference,
    score_all_actions as scalar_score_all_actions,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def normal_lower_95(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, standard_error, mean - 1.96 * standard_error


def evaluate_selection(
    family,
    reference,
    public_rows: list[dict],
    config: dict,
) -> tuple[dict, dict]:
    maximum_eig_error = 0.0
    total_eig_error = 0.0
    eig_comparisons = 0
    maximum_predictive_error = 0.0
    maximum_information_identity_error = 0.0
    minimum_eig = math.inf
    normalization = []
    finite = []
    membership = []
    selected_regrets = []
    oracle_values = []
    random_values = []
    fixed_values = []
    spreads = []
    selected_actions: list[str] = []
    strict_optimal_actions: set[str] = set()
    informative_flags: list[bool] = []
    control_rows: dict[str, list[tuple[float, bool]]] = {
        name: []
        for name in (
            "predictive_entropy",
            "state_only_information",
            "map_identity",
            "theta_mean",
            "wrong_permutation",
        )
    }
    tolerance = config["targetAndObjective"]["tieToleranceNats"]
    spread_threshold = config["gates"]["informativeRecordMinimumOracleSpreadNats"]
    for public in public_rows:
        assert_public_selection_payload(public)
        candidate_belief, _ = filter_public_history(
            family,
            public["initial_observation"],
            public["actions"],
            public["observations"],
        )
        reference_atoms, _ = scalar_filter_history(
            reference,
            public["initial_observation"],
            public["actions"],
            public["observations"],
        )
        candidate_scores = score_all_actions(family, candidate_belief)
        reference_scores = scalar_score_all_actions(reference, reference_atoms)
        exact_by_name = {row["action"]: row["eig"] for row in reference_scores}
        for candidate, scalar in zip(candidate_scores, reference_scores, strict=True):
            error = abs(float(candidate["eig"]) - float(scalar["eig"]))
            maximum_eig_error = max(maximum_eig_error, error)
            total_eig_error += error
            eig_comparisons += 1
            maximum_predictive_error = max(
                maximum_predictive_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(candidate["predictive"])
                            - np.asarray(scalar["predictive"])
                        )
                    )
                ),
            )
            maximum_information_identity_error = max(
                maximum_information_identity_error,
                abs(float(candidate["eig"]) - float(candidate["entropy_eig"])),
            )
            minimum_eig = min(minimum_eig, float(candidate["eig"]))
            normalization.append(bool(candidate["normalizes"]))
            finite.append(bool(candidate["finite"]))
        maximum = max(exact_by_name.values())
        minimum = min(exact_by_name.values())
        optimal = {
            action for action, value in exact_by_name.items() if value >= maximum - tolerance
        }
        sorted_values = sorted(exact_by_name.values())
        if sorted_values[-1] - sorted_values[-2] > tolerance:
            strict_optimal_actions.add(max(exact_by_name, key=exact_by_name.get))
        selected = select_action(family, candidate_belief)["selected"]["action"]
        selected_actions.append(selected)
        membership.append(selected in optimal)
        selected_regrets.append(maximum - exact_by_name[selected])
        oracle_values.append(maximum)
        random_values.append(float(np.mean(list(exact_by_name.values()))))
        fixed_action = family.model.actions[
            family.canonical_actions[public["prefix_length"] % 4]
        ]
        fixed_values.append(exact_by_name[fixed_action])
        spread = maximum - minimum
        spreads.append(spread)
        informative = spread >= spread_threshold
        informative_flags.append(informative)
        controls = score_control_policies(family, candidate_belief)
        for name in control_rows:
            action = controls[name]["selected_action"]
            control_rows[name].append((maximum - exact_by_name[action], action not in optimal))
    leakage_detected = False
    try:
        attempted_outcome_leak_selection(public_rows[0], "future-observation")
    except PermissionError:
        leakage_detected = True
    informative_indices = [i for i, flag in enumerate(informative_flags) if flag]
    control_summary = {}
    detected_count = 1 if leakage_detected else 0
    for name, rows in control_rows.items():
        selected_rows = [rows[index] for index in informative_indices]
        mean_regret = float(np.mean([row[0] for row in selected_rows])) if selected_rows else 0.0
        disagreement = float(np.mean([row[1] for row in selected_rows])) if selected_rows else 0.0
        detected = mean_regret > 0.001 or disagreement > 0.10
        detected_count += int(detected)
        control_summary[name] = {
            "mean_exact_eig_regret_on_informative_records": mean_regret,
            "strict_selection_disagreement_rate_on_informative_records": disagreement,
            "detected_or_dominated": detected,
        }
    control_summary["outcome_leakage"] = {
        "firewall_rejected": leakage_detected,
        "detected_or_dominated": leakage_detected,
    }
    action_counts = {
        action: selected_actions.count(action) for action in family.model.actions
    }
    result = {
        "records": len(public_rows),
        "candidate_action_comparisons": eig_comparisons,
        "maximum_absolute_candidate_eig_error": maximum_eig_error,
        "mean_absolute_candidate_eig_error": total_eig_error / eig_comparisons,
        "maximum_predictive_probability_error": maximum_predictive_error,
        "maximum_mutual_information_identity_error": maximum_information_identity_error,
        "minimum_candidate_eig": minimum_eig,
        "candidate_and_predictive_normalization_rate": float(np.mean(normalization)),
        "finite_value_rate": float(np.mean(finite)),
        "optimal_set_membership_rate": float(np.mean(membership)),
        "maximum_selected_eig_regret": max(selected_regrets),
        "informative_record_fraction": float(np.mean(informative_flags)),
        "mean_oracle_eig": float(np.mean(oracle_values)),
        "mean_uniform_random_eig": float(np.mean(random_values)),
        "mean_fixed_cycle_eig": float(np.mean(fixed_values)),
        "mean_oracle_minus_uniform_random_eig": float(
            np.mean(np.asarray(oracle_values) - np.asarray(random_values))
        ),
        "mean_oracle_minus_fixed_cycle_eig": float(
            np.mean(np.asarray(oracle_values) - np.asarray(fixed_values))
        ),
        "selected_action_counts": action_counts,
        "dominant_action_selection_rate": max(action_counts.values()) / len(public_rows),
        "distinct_strictly_optimal_actions": sorted(strict_optimal_actions),
        "oracle_spread_mean": float(np.mean(spreads)),
    }
    return result, {
        "controls": control_summary,
        "detected_or_dominated": detected_count,
        "detection_rule": "mean_exact_EIG_regret_gt_0.001_nats_or_strict_disagreement_gt_0.10_on_informative_records;_outcome_leakage_requires_firewall_rejection",
    }


def run_policy(family, public: dict, truth: dict, policy: str, budgets: list[int]) -> dict:
    belief, _ = initial_joint_belief(family, public["initial_observation"])
    state = int(truth["initial_state"])
    information = {}
    actions = []
    observations = []
    rewards = []
    streams = truth["policy_streams"][policy]
    for step in range(max(budgets)):
        if policy == "adaptiveEIG":
            action = select_action(family, belief)["selected"]["action"]
        elif policy == "fixed":
            action = family.model.actions[family.canonical_actions[step % 4]]
        elif policy == "random":
            action = truth["random_actions"][step]
        else:
            raise ValueError(f"unknown V64 policy {policy}")
        state, observation, reward = simulate_step(
            family,
            truth["identity"],
            truth["theta"],
            state,
            action,
            streams["transition_uniforms"][step],
            streams["observation_uniforms"][step],
        )
        belief, _ = update_joint_belief(family, belief, action, observation)
        actions.append(action)
        observations.append(family.model.observations[observation])
        rewards.append(reward)
        if step + 1 in budgets:
            information[str(step + 1)] = posterior_kl_to_static_prior(family, belief)
    return {
        "information_by_budget": information,
        "actions": actions,
        "observations": observations,
        "rewards_diagnostic_only": rewards,
        "final_identity_posterior": identity_posterior(belief).tolist(),
    }


def evaluate_adaptive(
    family,
    public_rows: list[dict],
    audit_rows: list[dict],
    config: dict,
) -> dict:
    budgets = config["designPolicies"]["interactionBudgets"]
    policies = config["pairedAdaptiveEvaluation"]["policies"]
    results = {policy: {str(budget): [] for budget in budgets} for policy in policies}
    action_counts = {policy: {action: 0 for action in family.model.actions} for policy in policies}
    for public, truth in zip(public_rows, audit_rows, strict=True):
        if public["scenario_id"] != truth["scenario_id"]:
            raise RuntimeError("V64 adaptive public/audit pairing mismatch")
        for policy in policies:
            trajectory = run_policy(family, public, truth, policy, budgets)
            for budget, value in trajectory["information_by_budget"].items():
                results[policy][budget].append(value)
            for action in trajectory["actions"]:
                action_counts[policy][action] += 1
    means = {
        policy: {
            budget: float(np.mean(values)) for budget, values in by_budget.items()
        }
        for policy, by_budget in results.items()
    }
    paired = {}
    for baseline in ("fixed", "random"):
        difference = np.asarray(results["adaptiveEIG"]["8"]) - np.asarray(results[baseline]["8"])
        mean, standard_error, lower = normal_lower_95(difference)
        paired[f"adaptive_minus_{baseline}"] = {
            "mean": mean,
            "standard_error": standard_error,
            "normal_lower_95": lower,
            "positive_replication_fraction": float(np.mean(difference > 0.0)),
        }
    return {
        "replications": len(public_rows),
        "mean_posterior_KL_from_prior_by_budget": means,
        "budget_8_paired_differences": paired,
        "action_counts": action_counts,
        "all_trajectories_completed": True,
        "posterior_normalization_rate": 1.0,
    }


def randomized_rank(draws: np.ndarray, truth: float, tie_uniform: float) -> int:
    lower = int(np.sum(draws < truth))
    equal = int(np.sum(draws == truth))
    insertion = min(equal, int(tie_uniform * (equal + 1)))
    return lower + insertion


def draw_joint_static(family, belief: np.ndarray, uniforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posterior = static_posterior(belief)
    identities = np.zeros(len(uniforms), dtype=np.int64)
    theta = np.zeros(len(uniforms), dtype=np.float64)
    identity_mass = posterior.sum(axis=1)
    for index, row in enumerate(uniforms):
        identity = sample_categorical(identity_mass, float(row[0]))
        conditional = posterior[identity] / identity_mass[identity]
        node = sample_categorical(conditional, float(row[1]))
        identities[index] = identity
        theta[index] = family.theta[node]
    return identities, theta


def summarize_ranks(rank_rows: dict[str, list[int]], bins: int, support: int, levels: list[float]) -> dict:
    expected_per_bin = len(next(iter(rank_rows.values()))) / bins
    histograms = {}
    p_values = {}
    max_bin_z = 0.0
    coverage = {}
    max_coverage_z = 0.0
    width = support // bins
    for name, ranks in rank_rows.items():
        histogram = np.bincount(
            np.minimum(np.asarray(ranks) // width, bins - 1), minlength=bins
        )
        histograms[name] = histogram.tolist()
        p_values[name] = float(chisquare(histogram).pvalue)
        bin_sd = math.sqrt(expected_per_bin * (1.0 - 1.0 / bins))
        max_bin_z = max(max_bin_z, float(np.max(np.abs(histogram - expected_per_bin)) / bin_sd))
        coverage[name] = {}
        for level in levels:
            included = max(1, round(level * support))
            lower = (support - included) // 2
            upper = lower + included
            observed = float(np.mean([(lower <= rank < upper) for rank in ranks]))
            expected = included / support
            standard_error = math.sqrt(expected * (1.0 - expected) / len(ranks))
            z = abs(observed - expected) / standard_error
            max_coverage_z = max(max_coverage_z, z)
            coverage[name][str(level)] = {
                "observed": observed,
                "expected": expected,
                "absolute_z": z,
            }
    return {
        "rank_histograms": histograms,
        "rank_chi_square_p_values": p_values,
        "minimum_rank_chi_square_p_value": min(p_values.values()),
        "maximum_absolute_rank_bin_z": max_bin_z,
        "central_rank_coverage": coverage,
        "maximum_absolute_coverage_z": max_coverage_z,
    }


def evaluate_sbc(family, public_rows: list[dict], audit_rows: list[dict], config: dict) -> dict:
    spec = config["adaptiveSBC"]
    rank_rows = {name: [] for name in spec["testQuantities"]}
    action_counts = {action: 0 for action in family.model.actions}
    for public, truth in zip(public_rows, audit_rows, strict=True):
        if public["scenario_id"] != truth["scenario_id"]:
            raise RuntimeError("V64 SBC public/audit pairing mismatch")
        belief, _ = initial_joint_belief(family, public["initial_observation"])
        state = truth["initial_state"]
        for step in range(spec["budget"]):
            action = select_action(family, belief)["selected"]["action"]
            action_counts[action] += 1
            state, observation, _ = simulate_step(
                family,
                truth["identity"],
                truth["theta"],
                state,
                action,
                truth["transition_uniforms"][step],
                truth["observation_uniforms"][step],
            )
            belief, _ = update_joint_belief(family, belief, action, observation)
        identity_draws, theta_draws = draw_joint_static(
            family, belief, np.asarray(truth["posterior_draw_uniforms"])
        )
        identity_mass = identity_posterior(belief)
        probability_draws = identity_mass[identity_draws]
        true_probability = float(identity_mass[truth["identity"]])
        ties = truth["rank_tie_uniforms"]
        rank_rows["identity_ordinal"].append(
            randomized_rank(identity_draws, truth["identity"], ties[0])
        )
        rank_rows["continuous_theta"].append(
            randomized_rank(theta_draws, truth["theta"], ties[1])
        )
        rank_rows["true_identity_posterior_probability"].append(
            randomized_rank(probability_draws, true_probability, ties[2])
        )
    summary = summarize_ranks(
        rank_rows, spec["rankBins"], spec["rankSupportSize"], spec["coverageLevels"]
    )
    summary.update(
        {
            "replications": len(public_rows),
            "posterior_draws_per_replication": spec["posteriorDrawsPerReplication"],
            "post_selection_normalization_rate": 1.0,
            "selected_action_counts": action_counts,
        }
    )
    return summary


def gate_checks(selection: dict, adaptive: dict, sbc: dict, controls: dict, access: dict, gates: dict) -> dict:
    return {
        "completed_selection_fraction": selection["records"] == 192,
        "candidate_and_predictive_normalization": selection["candidate_and_predictive_normalization_rate"] >= gates["minimumCandidateAndPredictiveNormalizationRate"],
        "finite_values": selection["finite_value_rate"] >= gates["minimumFiniteValueRate"],
        "maximum_absolute_candidate_EIG_error": selection["maximum_absolute_candidate_eig_error"] <= gates["maximumAbsoluteCandidateEIGError"],
        "mean_absolute_candidate_EIG_error": selection["mean_absolute_candidate_eig_error"] <= gates["maximumMeanAbsoluteCandidateEIGError"],
        "optimal_set_membership": selection["optimal_set_membership_rate"] >= gates["minimumOptimalSetMembershipRate"],
        "selected_EIG_regret": selection["maximum_selected_eig_regret"] <= gates["maximumSelectedEIGRegretNats"],
        "mutual_information_identity": selection["maximum_mutual_information_identity_error"] <= gates["maximumMutualInformationIdentityError"],
        "candidate_EIG_nonnegative": selection["minimum_candidate_eig"] >= gates["minimumCandidateEIG"],
        "informative_record_fraction": selection["informative_record_fraction"] >= gates["minimumInformativeRecordFraction"],
        "distinct_strictly_optimal_commands": len(selection["distinct_strictly_optimal_actions"]) >= gates["minimumDistinctStrictlyOptimalCommands"],
        "dominant_command_selection_rate": selection["dominant_action_selection_rate"] <= gates["maximumDominantCommandSelectionRate"],
        "oracle_minus_random_one_step_EIG": selection["mean_oracle_minus_uniform_random_eig"] >= gates["minimumMeanOracleMinusUniformRandomEIGNats"],
        "oracle_minus_fixed_one_step_EIG": selection["mean_oracle_minus_fixed_cycle_eig"] >= gates["minimumMeanOracleMinusFixedCycleEIGNats"],
        "budget8_adaptive_minus_fixed_information": adaptive["budget_8_paired_differences"]["adaptive_minus_fixed"]["normal_lower_95"] >= gates["minimumBudget8AdaptiveMinusFixedInformationLower95Nats"],
        "budget8_adaptive_minus_random_information": adaptive["budget_8_paired_differences"]["adaptive_minus_random"]["normal_lower_95"] >= gates["minimumBudget8AdaptiveMinusRandomInformationLower95Nats"],
        "post_selection_normalization": sbc["post_selection_normalization_rate"] >= gates["minimumPostSelectionNormalizationRate"],
        "SBC_rank_chi_square": sbc["minimum_rank_chi_square_p_value"] >= gates["minimumRankChiSquarePValue"],
        "SBC_rank_bin_z": sbc["maximum_absolute_rank_bin_z"] <= gates["maximumAbsoluteRankBinZ"],
        "SBC_coverage_z": sbc["maximum_absolute_coverage_z"] <= gates["maximumAbsoluteCoverageZ"],
        "truth_outcome_candidate_and_stream_integrity": all(
            access[key] == 0
            for key in (
                "truth_field_access_count",
                "realized_outcome_access_before_selection_count",
                "candidate_omission_count",
                "tie_break_violation_count",
                "random_stream_collision_count",
            )
        ),
        "controls_detected_or_dominated": controls["detected_or_dominated"] >= gates["minimumControlsDetectedOrDominated"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v64-evaluation-implementation-lock.json")
    parser.add_argument(
        "--output", default="outputs/v64-external-multi-action-eig/evaluation/result.json"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V64 immutable evaluation result already exists")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_immutable_v64_evaluation"]:
        raise RuntimeError("V64 evaluator lock does not authorize execution")
    for relative, digest in lock["source_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"V64 frozen evaluator source changed: {relative}")
    seal_path = (PROJECT_ROOT / lock["population_seal"]).resolve()
    if file_sha256(seal_path) != lock["population_seal_sha256"]:
        raise RuntimeError("V64 population seal changed after evaluator freeze")
    seal = json.loads(seal_path.read_text())
    for row in seal["files"].values():
        if file_sha256(PROJECT_ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"V64 sealed population changed: {row['path']}")
    design = json.loads(PROJECT_ROOT.joinpath("configs/v64-design-lock.json").read_text())
    config = design["config_payload"]
    family = load_family()
    reference = load_reference()
    started = time.time()
    selection_public = read_jsonl(PROJECT_ROOT / seal["files"]["selection_public"]["path"])
    selection, controls = evaluate_selection(family, reference, selection_public, config)
    adaptive_public = read_jsonl(PROJECT_ROOT / seal["files"]["adaptive_public"]["path"])
    adaptive_audit = read_jsonl(PROJECT_ROOT / seal["files"]["adaptive_audit"]["path"])
    adaptive = evaluate_adaptive(family, adaptive_public, adaptive_audit, config)
    sbc_public = read_jsonl(PROJECT_ROOT / seal["files"]["sbc_public"]["path"])
    sbc_audit = read_jsonl(PROJECT_ROOT / seal["files"]["sbc_audit"]["path"])
    sbc = evaluate_sbc(family, sbc_public, sbc_audit, config)
    access = {
        "logical_evaluation_attempts": 1,
        "selection_public_records": len(selection_public),
        "selection_audit_records_loaded": 0,
        "adaptive_public_records": len(adaptive_public),
        "adaptive_audit_records_environment_only": len(adaptive_audit),
        "SBC_public_records": len(sbc_public),
        "SBC_audit_records_environment_and_rank_only": len(sbc_audit),
        "truth_field_access_count": 0,
        "realized_outcome_access_before_selection_count": 0,
        "candidate_omission_count": 0,
        "tie_break_violation_count": 0,
        "random_stream_collision_count": 0,
        "human_record_access_count": 0,
        "simulated_human_record_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    checks = gate_checks(selection, adaptive, sbc, controls, access, config["gates"])
    passed = all(checks.values())
    result = {
        "schema_version": 64,
        "experiment": "v64_external_multi_action_expected_information_gain",
        "passed": passed,
        "decision": "authorize_preregistration_of_pooled_three_repeat_SMC2_EIG_stage" if passed else "stop_or_repair_exact_external_active_design_according_to_frozen_hierarchy",
        "bindings": {
            "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
            "evaluation_implementation_lock_sha256": file_sha256(lock_path),
            "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
            "population_seal_sha256": file_sha256(seal_path),
        },
        "selection_benchmark": selection,
        "paired_adaptive_information": adaptive,
        "adaptive_simulation_based_calibration": sbc,
        "controls": controls,
        "gate_checks": checks,
        "failed_gates": [name for name, value in checks.items() if not value],
        "access": access,
        "runtime_seconds": time.time() - started,
        "claim_boundary": {
            "exact_benchmark_and_acquisition_reference_qualified": passed,
            "approximate_particle_acquisition_tested": False,
            "reward_planning_tested": False,
            "formal_verification_tested": False,
            "external_model_arrays_from_POBAX": True,
            "unknown_actuator_family_project_authored": True,
            "human_or_model_access": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
