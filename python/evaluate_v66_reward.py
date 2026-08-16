#!/usr/bin/env python3
"""Durable one-shot V66 external Bayes-adaptive reward evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import load_family
from v65r3_smc2_eig import pool_repeats, smc2_inference
from v66_bayes_adaptive_reward import (
    StaticKernel,
    compact_policy,
    evaluate_policy,
    evaluate_policy_information,
    evaluate_root_action_values,
    exact_history_kernel_and_belief,
    map_model_policy,
    persistent_posterior_sampling_mixture,
    plan_bayes_adaptive,
    plan_information_only_policy,
    plan_invalid_mean_transition_policy,
    plan_myopic_reward_policy,
    pooled_measure_kernel_and_belief,
)


TERMINAL_NAMES = (
    "attempt.json",
    "result.json",
    "failure.json",
    "record-cells.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"V66 stale atomic-write temporary exists: {temporary.name}")
    try:
        with temporary.open("x") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def reserve_attempt(output_dir: Path, marker: dict[str, Any]) -> Path:
    existing = [name for name in TERMINAL_NAMES if (output_dir / name).exists()]
    if existing:
        raise RuntimeError(
            "V66 one-shot evaluation already attempted or materialized: "
            + ",".join(existing)
        )
    attempt_path = output_dir / "attempt.json"
    atomic_write_json(attempt_path, marker)
    return attempt_path


def failure_payload(
    *,
    lock_path: Path,
    attempt_path: Path,
    stage: str,
    progress: dict[str, int],
    access: dict[str, int],
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": "66",
        "experiment": "v66_external_Bayes_adaptive_reward_decisions",
        "passed": False,
        "status": "terminal_exception",
        "decision": "do_not_authorize_policy_verification",
        "one_shot_authorization_consumed": True,
        "stage": stage,
        "progress": progress,
        "access": access,
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "bindings": {
            "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
            "evaluation_implementation_lock_sha256": file_sha256(lock_path),
            "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
            "attempt_sha256": file_sha256(attempt_path),
        },
        "claim_boundary": {
            "reward_gates_evaluable": False,
            "V66_rerun_authorized": False,
            "policy_verification_authorized": False,
        },
    }


def batched_known_model_oracle(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Exact vectorization of independently optimal known-static-model POMDPs."""
    joint = np.asarray(belief, dtype=np.float64)
    static = joint.sum(axis=1)
    active = static > 0.0
    states = np.zeros_like(joint)
    states[active] = joint[active] / static[active, None]
    states[~active, 0] = 1.0

    def recurse(current: np.ndarray, remaining: int) -> tuple[np.ndarray, np.ndarray]:
        if remaining == 0:
            return np.zeros(len(current), dtype=np.float64), np.zeros(len(current), dtype=np.int16)
        q = np.zeros((len(current), len(kernel.canonical_actions)), dtype=np.float64)
        for position, action in enumerate(kernel.canonical_actions):
            q[:, position] = np.einsum(
                "ms,mst,st->m",
                current,
                kernel.transitions[:, action],
                kernel.rewards[action],
                optimize=True,
            )
            if remaining <= 1:
                continue
            predicted = np.einsum(
                "ms,mst->mt", current, kernel.transitions[:, action], optimize=True
            )
            raw = predicted[:, :, None] * kernel.observations[action][None, :, :]
            probabilities = raw.sum(axis=1)
            continuation = np.zeros(len(current), dtype=np.float64)
            for observation in range(len(kernel.observation_names)):
                probability = probabilities[:, observation]
                posterior = np.zeros_like(current)
                valid = probability > 0.0
                posterior[valid] = raw[valid, :, observation] / probability[valid, None]
                posterior[~valid, 0] = 1.0
                child, _ = recurse(posterior, remaining - 1)
                continuation += probability * child
            q[:, position] += kernel.discount * continuation
        maximum = q.max(axis=1)
        selected_position = np.zeros(len(current), dtype=np.int16)
        for model in range(len(current)):
            selected_position[model] = next(
                position
                for position, value in enumerate(q[model])
                if maximum[model] - value <= tie_tolerance
            )
        selected = np.asarray(
            [kernel.canonical_actions[position] for position in selected_position],
            dtype=np.int16,
        )
        return maximum, selected

    values, actions = recurse(states, int(horizon))
    return {
        "value": float(np.sum(static * values)),
        "per_model_values": values,
        "per_model_actions": actions,
        "static_weights": static,
        "active_models": int(np.count_nonzero(active)),
        "physical_state_revealed": False,
    }


def duplicate_single_repeat_measure(repeat: dict[str, Any]) -> dict[str, Any]:
    """Represent one repeat as a three-copy pool without changing its posterior measure."""
    duplicated = pool_repeats([repeat, repeat, repeat])
    duplicated["diagnostic_source"] = "one_repeat_duplicated_three_times_without_weight_change"
    return duplicated


def _q95(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95))


def summary(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0 or np.any(~np.isfinite(data)):
        raise ValueError("V66 summary requires finite nonempty values")
    return {
        "mean": float(np.mean(data)),
        "q95": _q95(data),
        "max": float(np.max(data)),
        "min": float(np.min(data)),
    }


def _tie_valid(
    policy: dict[str, Any], canonical_actions: Sequence[int], tolerance: float
) -> bool:
    maximum = max(float(value) for value in policy["q_values"])
    expected = next(
        action
        for action, value in zip(canonical_actions, policy["q_values"], strict=True)
        if maximum - float(value) <= tolerance
    )
    return int(policy["selected_action"]) == int(expected)


def evaluate_record(
    family,
    record: dict[str, Any],
    config: dict[str, Any],
    smc_config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    horizon = int(config["planning"]["horizonActions"])
    tolerance = float(config["planning"]["tieToleranceReward"])
    budget = int(config["approximatePosterior"]["outerThetaParticlesPerIdentity"])
    repeat_count = int(config["approximatePosterior"]["independentRepeats"])

    exact_started = time.perf_counter()
    exact_kernel, exact_belief, exact_log_evidence = exact_history_kernel_and_belief(
        family, record
    )
    exact_stats: dict[str, int] = {}
    exact_policy = plan_bayes_adaptive(
        exact_kernel,
        exact_belief,
        horizon,
        tie_tolerance=tolerance,
        retain_forced_root_actions=True,
        stats=exact_stats,
    )
    exact_policy_value = evaluate_policy(exact_kernel, exact_belief, exact_policy, horizon)
    exact_q_evaluated = evaluate_root_action_values(
        exact_kernel, exact_belief, exact_policy, horizon
    )
    exact_seconds = time.perf_counter() - exact_started

    inference_started = time.perf_counter()
    repeats = [
        smc2_inference(family, record, smc_config, budget, repeat)
        for repeat in range(repeat_count)
    ]
    pooled = pool_repeats(repeats)
    approximate_kernel, approximate_belief, _ = pooled_measure_kernel_and_belief(
        family, record, pooled
    )
    inference_seconds = time.perf_counter() - inference_started

    approximate_started = time.perf_counter()
    approximate_stats: dict[str, int] = {}
    approximate_policy = plan_bayes_adaptive(
        approximate_kernel,
        approximate_belief,
        horizon,
        tie_tolerance=tolerance,
        retain_forced_root_actions=True,
        stats=approximate_stats,
    )
    approximate_self_value = evaluate_policy(
        approximate_kernel, approximate_belief, approximate_policy, horizon
    )
    approximate_exact_value = evaluate_policy(
        exact_kernel, exact_belief, approximate_policy, horizon
    )
    approximate_exact_q = evaluate_root_action_values(
        exact_kernel, exact_belief, approximate_policy, horizon
    )
    approximate_seconds = time.perf_counter() - approximate_started

    oracle_started = time.perf_counter()
    oracle = batched_known_model_oracle(exact_kernel, exact_belief, horizon)
    map_result = map_model_policy(exact_kernel, exact_belief, horizon, tie_tolerance=tolerance)
    mixture_config = config["persistentMixtureQuadrature"]
    mixture = persistent_posterior_sampling_mixture(
        exact_kernel,
        exact_belief,
        horizon,
        points=int(mixture_config["primaryPoints"]),
        offset=float(mixture_config["primarySystematicOffset"]),
        tie_tolerance=tolerance,
    )
    mixture_sensitivity = persistent_posterior_sampling_mixture(
        exact_kernel,
        exact_belief,
        horizon,
        points=int(mixture_config["sensitivityPoints"]),
        offset=float(mixture_config["sensitivitySystematicOffset"]),
        tie_tolerance=tolerance,
    )
    oracle_control_seconds = time.perf_counter() - oracle_started

    controls_started = time.perf_counter()
    myopic = plan_myopic_reward_policy(
        exact_kernel, exact_belief, horizon, tie_tolerance=tolerance
    )
    information = plan_information_only_policy(
        exact_kernel, exact_belief, horizon, tie_tolerance=tolerance
    )
    invalid_mean = plan_invalid_mean_transition_policy(
        exact_kernel, exact_belief, horizon, tie_tolerance=tolerance
    )
    control_policies = {
        "myopic_expected_reward": myopic,
        "information_only_EIG": information,
        "invalid_mean_transition": invalid_mean,
    }
    control_values = {
        name: evaluate_policy(exact_kernel, exact_belief, policy, horizon)
        for name, policy in control_policies.items()
    }
    controls_seconds = time.perf_counter() - controls_started

    repeat_diagnostics = []
    for repeat, inference in enumerate(repeats):
        single_pool = duplicate_single_repeat_measure(inference)
        single_kernel, single_belief, _ = pooled_measure_kernel_and_belief(
            family, record, single_pool
        )
        single_policy = plan_bayes_adaptive(
            single_kernel,
            single_belief,
            horizon,
            tie_tolerance=tolerance,
        )
        single_exact_value = evaluate_policy(
            exact_kernel, exact_belief, single_policy, horizon
        )
        repeat_diagnostics.append(
            {
                "repeat": repeat,
                "selected_action": int(single_policy["selected_action"]),
                "selected_action_name": single_policy["selected_action_name"],
                "exact_environment_value": float(single_exact_value),
                "exact_value_regret": float(exact_policy["value"] - single_exact_value),
                "random_stream_collision_count": int(
                    inference["diagnostics"]["random_stream_collision_count"]
                ),
                "exact_zero_identity_count": int(
                    inference["diagnostics"]["exact_zero_identity_count"]
                ),
                "work": inference["diagnostics"]["work"],
                "inference_runtime_seconds": float(
                    inference["diagnostics"]["runtime_seconds"]
                ),
            }
        )

    exact_max = float(exact_policy["value"])
    selected_position = exact_kernel.canonical_actions.index(
        int(approximate_policy["selected_action"])
    )
    exact_selected_q = float(exact_policy["q_values"][selected_position])
    epsilon = float(config["gates"]["epsilonOptimalRootReward"])
    exact_optimal = {int(value) for value in exact_policy["optimal_actions"]}
    root_q_errors = [
        abs(float(a) - float(b))
        for a, b in zip(exact_policy["q_values"], approximate_exact_q, strict=True)
    ]
    strategy_values = {
        "exact_Bayes_adaptive": exact_max,
        "pooled_SMC2_Bayes_adaptive_exact_environment": float(approximate_exact_value),
        "pooled_SMC2_Bayes_adaptive_self": float(approximate_self_value),
        "posterior_weighted_model_oracle": float(oracle["value"]),
        "joint_MAP_certainty_equivalent": float(map_result["exact_environment_value"]),
        "persistent_posterior_sampling_mixture_32": float(mixture["value"]),
        "persistent_posterior_sampling_mixture_64": float(mixture_sensitivity["value"]),
        **{name: float(value) for name, value in control_values.items()},
    }
    strategy_actions = {
        "exact_Bayes_adaptive": int(exact_policy["selected_action"]),
        "pooled_SMC2_Bayes_adaptive": int(approximate_policy["selected_action"]),
        "joint_MAP_certainty_equivalent": int(map_result["policy"]["selected_action"]),
        "myopic_expected_reward": int(myopic["selected_action"]),
        "information_only_EIG": int(information["selected_action"]),
        "invalid_mean_transition": int(invalid_mean["selected_action"]),
    }
    normalizes = bool(
        abs(float(exact_belief.sum()) - 1.0) <= 1e-10
        and abs(float(approximate_belief.sum()) - 1.0) <= 1e-10
        and abs(float(exact_policy["observation_probabilities"].sum()) - 1.0) <= 1e-10
        and abs(float(approximate_policy["observation_probabilities"].sum()) - 1.0) <= 1e-10
        and abs(sum(mixture["root_action_distribution"]) - 1.0) <= 1e-10
    )
    finite = bool(
        all(math.isfinite(value) for value in strategy_values.values())
        and all(math.isfinite(value) for value in root_q_errors)
    )
    tie_valid = bool(
        _tie_valid(exact_policy, exact_kernel.canonical_actions, tolerance)
        and _tie_valid(approximate_policy, approximate_kernel.canonical_actions, tolerance)
    )
    candidate_complete = bool(
        len(exact_policy["q_values"]) == len(exact_kernel.canonical_actions) == 4
        and len(approximate_policy["q_values"])
        == len(approximate_kernel.canonical_actions)
        == 4
    )

    policy_information = {
        "exact_Bayes_adaptive": evaluate_policy_information(
            exact_kernel, exact_belief, exact_policy, horizon
        ),
        "pooled_SMC2_Bayes_adaptive": evaluate_policy_information(
            exact_kernel, exact_belief, approximate_policy, horizon
        ),
        "myopic_expected_reward": evaluate_policy_information(
            exact_kernel, exact_belief, myopic, horizon
        ),
        "information_only_EIG": evaluate_policy_information(
            exact_kernel, exact_belief, information, horizon
        ),
    }
    return {
        "record_id": record["record_id"],
        "prefix_length": int(record["prefix_length"]),
        "exact_log_evidence": exact_log_evidence,
        "strategy_values": strategy_values,
        "strategy_actions": strategy_actions,
        "policy_information_nats": policy_information,
        "primary": {
            "exact_value_regret": float(exact_max - approximate_exact_value),
            "strict_optimal_membership": int(approximate_policy["selected_action"])
            in exact_optimal,
            "epsilon_optimal_membership": exact_max - exact_selected_q <= epsilon + 1e-12,
            "selected_action_exact_Q_regret": float(exact_max - exact_selected_q),
            "root_Q_absolute_errors": root_q_errors,
            "self_value_absolute_calibration_error": abs(
                float(approximate_self_value) - float(approximate_exact_value)
            ),
            "MAP_minus_SMC2_value": float(
                map_result["exact_environment_value"] - approximate_exact_value
            ),
            "mixture32_minus_SMC2_value": float(
                mixture["value"] - approximate_exact_value
            ),
            "oracle_dominance_residual": max(0.0, exact_max - float(oracle["value"])),
            "exact_Bellman_root_Q_reference_error": max(
                abs(float(a) - float(b))
                for a, b in zip(exact_policy["q_values"], exact_q_evaluated, strict=True)
            ),
            "exact_policy_evaluation_reference_error": abs(
                exact_max - exact_policy_value
            ),
        },
        "persistent_mixture": {
            "primary_points": mixture["points"],
            "primary_value": mixture["value"],
            "sensitivity_points": mixture_sensitivity["points"],
            "sensitivity_value": mixture_sensitivity["value"],
            "absolute_sensitivity_difference": abs(
                float(mixture["value"]) - float(mixture_sensitivity["value"])
            ),
            "root_action_distribution": mixture["root_action_distribution"],
            "selected_static_indices": mixture["selected_static_indices"],
            "sampled_model_persists_for_full_policy": mixture[
                "sampled_model_persists_for_full_policy"
            ],
        },
        "repeat_diagnostics": repeat_diagnostics,
        "exact_policy": compact_policy(exact_policy),
        "pooled_SMC2_policy": compact_policy(approximate_policy),
        "runtime": {
            "exact_seconds": exact_seconds,
            "SMC2_inference_seconds": inference_seconds,
            "approximate_planning_seconds": approximate_seconds,
            "oracle_and_mixture_seconds": oracle_control_seconds,
            "other_controls_seconds": controls_seconds,
            "total_seconds": time.perf_counter() - started,
            "exact_Bellman_nodes": exact_stats.get("bellman_nodes", 0),
            "approximate_Bellman_nodes": approximate_stats.get("bellman_nodes", 0),
            "exact_static_atoms": len(exact_kernel.identities),
            "approximate_static_atoms": len(approximate_kernel.identities),
        },
        "integrity": {
            "finite": finite,
            "normalizes": normalizes,
            "candidate_complete": candidate_complete,
            "tie_break_valid": tie_valid,
            "oracle_physical_state_revealed": oracle["physical_state_revealed"],
            "invalid_mean_transition_labeled_invalid": "invalid_static_semantics"
            in invalid_mean,
        },
    }


def aggregate_evaluation(
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    implementation_audit: dict[str, Any],
    access: dict[str, int],
) -> dict[str, Any]:
    gates = config["gates"]
    regrets = [float(row["primary"]["exact_value_regret"]) for row in rows]
    root_errors = [
        float(value)
        for row in rows
        for value in row["primary"]["root_Q_absolute_errors"]
    ]
    calibration = [
        float(row["primary"]["self_value_absolute_calibration_error"])
        for row in rows
    ]
    map_minus = [float(row["primary"]["MAP_minus_SMC2_value"]) for row in rows]
    mixture_minus = [
        float(row["primary"]["mixture32_minus_SMC2_value"]) for row in rows
    ]
    strict_rate = float(
        np.mean([bool(row["primary"]["strict_optimal_membership"]) for row in rows])
    ) if rows else 0.0
    epsilon_rate = float(
        np.mean([bool(row["primary"]["epsilon_optimal_membership"]) for row in rows])
    ) if rows else 0.0
    strategy_names = list(rows[0]["strategy_values"]) if rows else []
    strategy_summaries = {
        name: summary([float(row["strategy_values"][name]) for row in rows])
        for name in strategy_names
    }

    empirical_controls: dict[str, dict[str, Any]] = {}
    mappings = {
        "MAP": ("joint_MAP_certainty_equivalent", "joint_MAP_certainty_equivalent"),
        "persistentPosteriorSamplingMixture": (
            "persistent_posterior_sampling_mixture_32",
            None,
        ),
        "myopicExpectedReward": ("myopic_expected_reward", "myopic_expected_reward"),
        "informationOnlyEIG": ("information_only_EIG", "information_only_EIG"),
        "firstRepeatOnly": (None, None),
    }
    for name, (value_name, action_name) in mappings.items():
        if name == "firstRepeatOnly":
            values = [
                float(row["repeat_diagnostics"][0]["exact_environment_value"])
                for row in rows
            ]
            actions = [
                int(row["repeat_diagnostics"][0]["selected_action"]) for row in rows
            ]
        else:
            values = [float(row["strategy_values"][value_name]) for row in rows]
            if action_name is None:
                disagreement = [
                    1.0
                    - sum(
                        row["persistent_mixture"]["root_action_distribution"][action]
                        for action in row["exact_policy"]["optimal_actions"]
                    )
                    for row in rows
                ]
                actions = None
            else:
                actions = [int(row["strategy_actions"][action_name]) for row in rows]
        control_regrets = [
            float(row["strategy_values"]["exact_Bayes_adaptive"]) - value
            for row, value in zip(rows, values, strict=True)
        ]
        if actions is not None:
            disagreement = [
                float(action not in set(row["exact_policy"]["optimal_actions"]))
                for row, action in zip(rows, actions, strict=True)
            ]
        mean_regret = float(np.mean(control_regrets)) if rows else 0.0
        disagreement_rate = float(np.mean(disagreement)) if rows else 0.0
        empirical_controls[name] = {
            "mean_exact_value_regret": mean_regret,
            "root_action_disagreement_rate": disagreement_rate,
            "detected_or_dominated": bool(
                mean_regret > 0.005 or disagreement_rate > 0.10
            ),
        }
    control_detection = {
        **{name: row["detected_or_dominated"] for name, row in empirical_controls.items()},
        "invalidMeanTransition": all(
            row["integrity"]["invalid_mean_transition_labeled_invalid"] for row in rows
        ),
        "sharedRandomStream": bool(
            implementation_audit["inherited_smc_shared_stream_detected"]
        ),
        "outcomeLeakage": bool(
            implementation_audit["mutation_checks"]["allow_truth_field_in_planner_fixture"]
        ),
    }
    detected = sum(bool(value) for value in control_detection.values())

    complete = len(rows) == int(config["population"]["records"])
    finite = bool(rows) and all(row["integrity"]["finite"] for row in rows)
    normalizes = bool(rows) and all(row["integrity"]["normalizes"] for row in rows)
    candidate_omissions = sum(
        not row["integrity"]["candidate_complete"] for row in rows
    )
    tie_violations = sum(not row["integrity"]["tie_break_valid"] for row in rows)
    oracle_residual = max(
        [float(row["primary"]["oracle_dominance_residual"]) for row in rows] or [math.inf]
    )
    bellman_error = max(
        [float(row["primary"]["exact_Bellman_root_Q_reference_error"]) for row in rows]
        or [math.inf]
    )
    policy_error = max(
        [float(row["primary"]["exact_policy_evaluation_reference_error"]) for row in rows]
        or [math.inf]
    )
    implementation_mutant_rate = float(np.mean(list(implementation_audit["mutation_checks"].values())))
    analytic_rate = float(np.mean(list(implementation_audit["analytic_checks"].values())))

    regret_summary = summary(regrets) if rows else {"mean": math.inf, "q95": math.inf, "max": math.inf, "min": math.inf}
    root_error_summary = summary(root_errors) if root_errors else {"mean": math.inf, "q95": math.inf, "max": math.inf, "min": math.inf}
    calibration_summary = summary(calibration) if calibration else {"mean": math.inf, "q95": math.inf, "max": math.inf, "min": math.inf}
    gate_checks = {
        "mean_SMC2_policy_value_regret": regret_summary["mean"] <= gates["maximumMeanSMC2PolicyValueRegret"],
        "q95_SMC2_policy_value_regret": regret_summary["q95"] <= gates["maximumQ95SMC2PolicyValueRegret"],
        "maximum_SMC2_policy_value_regret": regret_summary["max"] <= gates["maximumSMC2PolicyValueRegret"],
        "strict_root_optimal_membership": strict_rate >= gates["minimumStrictExactRootOptimalSetMembershipRate"],
        "epsilon_root_optimal_membership": epsilon_rate >= gates["minimumEpsilonOptimalRootMembershipRate"],
        "mean_root_Q_absolute_error": root_error_summary["mean"] <= gates["maximumMeanRootQAbsoluteError"],
        "q95_root_Q_absolute_error": root_error_summary["q95"] <= gates["maximumQ95RootQAbsoluteError"],
        "mean_self_value_calibration_error": calibration_summary["mean"] <= gates["maximumMeanAbsoluteSMC2SelfValueCalibrationError"],
        "q95_self_value_calibration_error": calibration_summary["q95"] <= gates["maximumQ95AbsoluteSMC2SelfValueCalibrationError"],
        "mean_noninferiority_to_MAP": float(np.mean(map_minus)) <= gates["maximumMeanRegretAboveMAPMargin"] if rows else False,
        "mean_noninferiority_to_persistent_mixture": float(np.mean(mixture_minus)) <= gates["maximumMeanRegretAbovePersistentMixtureMargin"] if rows else False,
        "oracle_dominance": oracle_residual <= gates["maximumOracleDominanceResidual"],
        "exact_Bellman_reference": bellman_error <= gates["maximumExactBellmanReferenceError"],
        "policy_evaluation_reference": policy_error <= gates["maximumPolicyEvaluationReferenceError"],
        "minimum_controls_detected_or_dominated": detected >= gates["minimumControlsDetectedOrDominated"],
        "complete_record_fraction": complete,
        "finite_value_rate": finite,
        "belief_normalization_rate": normalizes,
        "implementation_mutant_kill_rate": implementation_mutant_rate >= gates["minimumImplementationMutantKillRate"],
        "analytic_fixture_pass_rate": analytic_rate >= gates["minimumAnalyticFixturePassRate"],
        "single_expected_evaluation_attempt": access["logical_evaluation_attempts"] == 1,
        "candidate_omission_count": candidate_omissions <= gates["maximumCandidateOmissionCount"],
        "tie_break_violation_count": tie_violations <= gates["maximumTieBreakViolationCount"],
        "truth_field_access_count": access["truth_field_access_count"] <= gates["maximumTruthFieldAccessCount"],
        "V64_or_V65_evaluation_result_record_access": access["V64_or_V65_evaluation_result_record_access"] <= gates["maximumV64OrV65EvaluationResultRecordAccess"],
        "human_record_access_count": access["human_record_access_count"] <= gates["maximumHumanRecordAccessCount"],
        "model_forward_pass_count": access["model_forward_pass_count"] <= gates["maximumModelForwardPassCount"],
        "adapter_training_run_count": access["adapter_training_run_count"] <= gates["maximumAdapterTrainingRunCount"],
        "random_stream_collision_count": access["random_stream_collision_count"] <= gates["maximumUnintendedRandomStreamCollisions"],
    }
    gate_checks = {key: bool(value) for key, value in gate_checks.items()}
    failed = [key for key, value in gate_checks.items() if not value]
    passed = not failed
    return {
        "passed": passed,
        "decision": (
            "authorize_independent_bounded_policy_verification_only"
            if passed
            else "do_not_authorize_policy_verification"
        ),
        "failed_gates": failed,
        "gate_checks": gate_checks,
        "primary": {
            "policy_value_regret": regret_summary,
            "strict_root_optimal_membership_rate": strict_rate,
            "epsilon_root_optimal_membership_rate": epsilon_rate,
            "root_Q_absolute_error": root_error_summary,
            "self_value_absolute_calibration_error": calibration_summary,
            "mean_MAP_minus_SMC2_value": float(np.mean(map_minus)) if rows else math.inf,
            "mean_mixture32_minus_SMC2_value": float(np.mean(mixture_minus)) if rows else math.inf,
            "maximum_oracle_dominance_residual": oracle_residual,
            "maximum_exact_Bellman_reference_error": bellman_error,
            "maximum_policy_evaluation_reference_error": policy_error,
        },
        "strategy_value_summaries": strategy_summaries,
        "controls": {
            "empirical": empirical_controls,
            "detected": control_detection,
            "detected_or_dominated": detected,
            "minimum_required": gates["minimumControlsDetectedOrDominated"],
        },
        "integrity": {
            "records": len(rows),
            "complete": complete,
            "finite": finite,
            "normalizes": normalizes,
            "candidate_omission_count": int(candidate_omissions),
            "tie_break_violation_count": int(tie_violations),
            "implementation_mutant_kill_rate": implementation_mutant_rate,
            "analytic_fixture_pass_rate": analytic_rate,
        },
        "by_prefix_length": {
            str(prefix): {
                "records": sum(int(row["prefix_length"]) == prefix for row in rows),
                "policy_value_regret": summary([
                    float(row["primary"]["exact_value_regret"])
                    for row in rows
                    if int(row["prefix_length"]) == prefix
                ]),
            }
            for prefix in config["population"]["prefixLengths"]
            if any(int(row["prefix_length"]) == prefix for row in rows)
        },
        "access": access,
    }


def run_evaluation(lock_path: Path, output_dir: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_immutable_evaluation"]:
        raise RuntimeError("V66 evaluator lock does not authorize evaluation")
    if lock["authorization"]["run_additional_evaluation"]:
        raise RuntimeError("V66 evaluator lock improperly authorizes additional evaluation")
    for relative, digest in lock["source_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"frozen V66 evaluator or dependency changed: {relative}")
    subset_seal_path = PROJECT_ROOT / lock["subset_seal"]
    if file_sha256(subset_seal_path) != lock["subset_seal_sha256"]:
        raise RuntimeError("V66 sealed subset changed before attempt")
    implementation_path = PROJECT_ROOT / lock["implementation_lock"]
    if file_sha256(implementation_path) != lock["implementation_lock_sha256"]:
        raise RuntimeError("V66 implementation lock changed before attempt")
    smc_design_path = PROJECT_ROOT / lock["source_v65r3_design_lock"]
    if file_sha256(smc_design_path) != lock["source_v65r3_design_lock_sha256"]:
        raise RuntimeError("V66 frozen V65r3 SMC2 design changed before attempt")
    smc_implementation_path = PROJECT_ROOT / lock["source_v65r3_implementation_lock"]
    if (
        file_sha256(smc_implementation_path)
        != lock["source_v65r3_implementation_lock_sha256"]
    ):
        raise RuntimeError("V66 frozen V65r3 SMC2 implementation changed before attempt")

    marker = {
        "schema_version": "66",
        "experiment": "v66_immutable_evaluation_attempt",
        "logical_evaluation_attempt": 1,
        "one_shot_authorization_consumed": True,
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "subset_seal_sha256": lock["subset_seal_sha256"],
        "implementation_lock_sha256": lock["implementation_lock_sha256"],
    }
    attempt_path = reserve_attempt(output_dir, marker)
    stage = "attempt_reserved_before_subset_load"
    progress = {
        "sealed_records_loaded": 0,
        "records_completed": 0,
        "SMC2_repeats_completed": 0,
        "exact_policies_completed": 0,
        "approximate_policies_completed": 0,
    }
    access = {
        "logical_evaluation_attempts": 1,
        "subset_public_records_loaded": 0,
        "V64_or_V65_evaluation_result_record_access": 0,
        "truth_field_access_count": 0,
        "candidate_omission_count": 0,
        "tie_break_violation_count": 0,
        "random_stream_collision_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
        "V65r3_evaluation_reruns": 0,
    }
    started = time.perf_counter()
    try:
        stage = "load_frozen_configuration_and_subset"
        implementation = json.loads(implementation_path.read_text())
        design_path = PROJECT_ROOT / implementation["design_lock"]
        design = json.loads(design_path.read_text())
        config = design["config_payload"]
        implementation_audit = json.loads(
            (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
        )
        smc_implementation = json.loads(smc_implementation_path.read_text())
        smc_implementation_audit = json.loads(
            (PROJECT_ROOT / smc_implementation["implementation_audit"]).read_text()
        )
        implementation_audit["inherited_smc_shared_stream_detected"] = bool(
            smc_implementation_audit["mutation_audit"]["checks"][
                "share_inner_random_streams"
            ]
        )
        smc_config = json.loads(smc_design_path.read_text())["config_payload"]
        subset_seal = json.loads(subset_seal_path.read_text())
        subset_path = PROJECT_ROOT / subset_seal["files"]["subset_public"]["path"]
        if file_sha256(subset_path) != subset_seal["files"]["subset_public"]["sha256"]:
            raise RuntimeError("V66 public subset changed after seal")
        records = read_jsonl(subset_path)
        progress["sealed_records_loaded"] = len(records)
        access["subset_public_records_loaded"] = len(records)
        family = load_family()

        rows = []
        stage = "record_inference_planning_and_exact_evaluation"
        for index, record in enumerate(records):
            row = evaluate_record(family, record, config, smc_config)
            rows.append(row)
            progress["records_completed"] += 1
            progress["SMC2_repeats_completed"] += len(row["repeat_diagnostics"])
            progress["exact_policies_completed"] += 1
            progress["approximate_policies_completed"] += 1
            access["random_stream_collision_count"] += sum(
                int(repeat["random_stream_collision_count"])
                for repeat in row["repeat_diagnostics"]
            )
            access["candidate_omission_count"] += int(
                not row["integrity"]["candidate_complete"]
            )
            access["tie_break_violation_count"] += int(
                not row["integrity"]["tie_break_valid"]
            )
            print(
                json.dumps(
                    {
                        "V66_record": index + 1,
                        "of": len(records),
                        "record_id": record["record_id"],
                        "regret": row["primary"]["exact_value_regret"],
                        "runtime_seconds": row["runtime"]["total_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        stage = "aggregate_original_noncompensatory_gates"
        result = aggregate_evaluation(rows, config, implementation_audit, access)
        result["schema_version"] = "66"
        result["experiment"] = "v66_external_Bayes_adaptive_reward_decisions"
        result["runtime_seconds"] = time.perf_counter() - started
        result["bindings"] = {
            "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
            "evaluation_implementation_lock_sha256": file_sha256(lock_path),
            "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
            "implementation_lock_sha256": file_sha256(implementation_path),
            "subset_seal": str(subset_seal_path.relative_to(PROJECT_ROOT)),
            "subset_seal_sha256": file_sha256(subset_seal_path),
            "source_v65r3_design_lock": str(smc_design_path.relative_to(PROJECT_ROOT)),
            "source_v65r3_design_lock_sha256": file_sha256(smc_design_path),
            "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
            "attempt_sha256": file_sha256(attempt_path),
        }
        stage = "atomically_write_raw_cells_and_result"
        raw_path = output_dir / "record-cells.jsonl"
        atomic_write_jsonl(raw_path, rows)
        result["record_cells"] = str(raw_path.relative_to(PROJECT_ROOT))
        result["record_cells_sha256"] = file_sha256(raw_path)
        atomic_write_json(output_dir / "result.json", result)
        return result
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        failure = failure_payload(
            lock_path=lock_path,
            attempt_path=attempt_path,
            stage=stage,
            progress=progress,
            access=access,
            error=error,
        )
        atomic_write_json(output_dir / "failure.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        default="configs/v66-evaluation-implementation-lock.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v66-external-bayes-adaptive-reward/evaluation",
    )
    arguments = parser.parse_args()
    lock_path = Path(arguments.lock)
    if not lock_path.is_absolute():
        lock_path = PROJECT_ROOT / lock_path
    output_dir = Path(arguments.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    result = run_evaluation(lock_path, output_dir)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "decision": result["decision"],
                "failed_gates": result["failed_gates"],
                "runtime_seconds": result["runtime_seconds"],
                "primary": result["primary"],
                "controls": result["controls"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
