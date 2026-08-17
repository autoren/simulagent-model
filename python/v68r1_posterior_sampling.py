#!/usr/bin/env python3
"""Total posterior-sampling control for V68r1 union-support evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np

from v66_bayes_adaptive_reward import (
    StaticKernel,
    _assert_belief,
    plan_bayes_adaptive,
    point_model_kernel_and_belief,
    step_belief,
    systematic_quantile_indices,
)
from v68_multi_environment_exact import evaluate_action_sequence


def evaluate_point_policy_with_total_fallback(
    exact_kernel: StaticKernel,
    exact_belief: np.ndarray,
    point_kernel: StaticKernel,
    point_belief: np.ndarray,
    horizon: int,
    *,
    fallback_action: int,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Execute a point-model policy on union support with a fixed fallback.

    The point model remains fixed at every supported history.  After an exact-
    mixture-positive observation with exact-zero point-model probability, the
    remaining policy is the open-loop repetition of ``fallback_action``.
    """
    exact = _assert_belief(exact_kernel, exact_belief)
    point = _assert_belief(point_kernel, point_belief)
    if horizon <= 0:
        raise ValueError("V68r1 totalized policy horizon must be positive")
    if fallback_action != exact_kernel.canonical_actions[0]:
        raise ValueError("V68r1 fallback must be the first frozen canonical action")
    if exact_kernel.action_names != point_kernel.action_names:
        raise ValueError("V68r1 exact and point action alphabets differ")
    if exact_kernel.observation_names != point_kernel.observation_names:
        raise ValueError("V68r1 exact and point observation alphabets differ")

    point_policy = plan_bayes_adaptive(
        point_kernel, point, horizon, tie_tolerance=tie_tolerance
    )
    action = int(point_policy["selected_action"])
    exact_step = step_belief(exact_kernel, exact, action)
    value = float(exact_step["reward"])
    expected_entry_probability = 0.0
    off_support_branch_count = 0
    if horizon > 1:
        point_step = step_belief(point_kernel, point, action)
        continuation = 0.0
        for observation, exact_posterior in exact_step["posteriors"].items():
            probability = float(exact_step["probabilities"][observation])
            point_probability = float(point_step["probabilities"][observation])
            if point_probability > 0.0:
                if observation not in point_step["posteriors"]:
                    raise RuntimeError("V68r1 positive point probability lacks a posterior")
                child = evaluate_point_policy_with_total_fallback(
                    exact_kernel,
                    exact_posterior,
                    point_kernel,
                    point_step["posteriors"][observation],
                    horizon - 1,
                    fallback_action=fallback_action,
                    tie_tolerance=tie_tolerance,
                )
                continuation += probability * float(child["value"])
                expected_entry_probability += probability * float(
                    child["expected_off_support_entry_probability"]
                )
                off_support_branch_count += int(child["off_support_branch_count"])
            else:
                continuation += probability * evaluate_action_sequence(
                    exact_kernel,
                    exact_posterior,
                    (fallback_action,) * (horizon - 1),
                )
                expected_entry_probability += probability
                off_support_branch_count += 1
        value += exact_kernel.discount * continuation
    return {
        "value": float(value),
        "selected_action": action,
        "expected_off_support_entry_probability": float(expected_entry_probability),
        "off_support_branch_count": int(off_support_branch_count),
        "fallback_action": int(fallback_action),
    }


def totalized_persistent_posterior_sampling_mixture(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    points: int,
    offset: float,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Deterministic posterior-sampling quadrature with a total union policy."""
    value = _assert_belief(kernel, belief)
    static = value.sum(axis=1)
    indices = systematic_quantile_indices(static, points, offset)
    fallback_action = int(kernel.canonical_actions[0])
    rows = []
    exact_values = []
    root_actions = np.zeros(len(kernel.action_names), dtype=np.float64)
    expected_entry_probabilities = []
    branch_count = 0
    for static_index in indices:
        point_kernel, point_belief, _ = point_model_kernel_and_belief(
            kernel, value, int(static_index)
        )
        evaluated = evaluate_point_policy_with_total_fallback(
            kernel,
            value,
            point_kernel,
            point_belief,
            horizon,
            fallback_action=fallback_action,
            tie_tolerance=tie_tolerance,
        )
        exact_values.append(float(evaluated["value"]))
        root_actions[int(evaluated["selected_action"])] += 1.0 / points
        expected_entry_probabilities.append(
            float(evaluated["expected_off_support_entry_probability"])
        )
        branch_count += int(evaluated["off_support_branch_count"])
        rows.append(
            {
                "static_index": int(static_index),
                "identity": int(kernel.identities[static_index]),
                "theta": float(kernel.thetas[static_index]),
                "selected_action": int(evaluated["selected_action"]),
                "exact_environment_value": float(evaluated["value"]),
                "off_support_branch_count": int(evaluated["off_support_branch_count"]),
                "expected_off_support_entry_probability": float(
                    evaluated["expected_off_support_entry_probability"]
                ),
            }
        )
    return {
        "value": float(np.mean(exact_values)),
        "points": int(points),
        "offset": float(offset),
        "selected_static_indices": indices.tolist(),
        "root_action_distribution": root_actions.tolist(),
        "models": rows,
        "off_support_branch_count": int(branch_count),
        "expected_off_support_entry_probability": float(
            np.mean(expected_entry_probabilities)
        ),
        "fallback_action": fallback_action,
        "fallback_action_name": kernel.action_names[fallback_action],
        "sampled_model_persists_on_supported_histories": True,
        "off_support_fallback_is_open_loop": True,
        "off_support_model_resampling": False,
        "epsilon_smoothing": False,
    }
