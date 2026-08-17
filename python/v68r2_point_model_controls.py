#!/usr/bin/env python3
"""Total point-model controls for the V68r2 union-support evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np

from v66_bayes_adaptive_reward import (
    StaticKernel,
    _assert_belief,
    plan_bayes_adaptive,
    point_model_kernel_and_belief,
)
from v68r1_posterior_sampling import (
    evaluate_point_policy_with_total_fallback,
    totalized_persistent_posterior_sampling_mixture,
)


def totalized_map_model_policy(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate the unchanged MAP point model as a total union-support policy."""
    value = _assert_belief(kernel, belief)
    static_index = int(np.argmax(value.sum(axis=1)))
    point_kernel, point_belief, mass = point_model_kernel_and_belief(
        kernel, value, static_index
    )
    policy = plan_bayes_adaptive(
        point_kernel, point_belief, horizon, tie_tolerance=tie_tolerance
    )
    fallback_action = int(kernel.canonical_actions[0])
    evaluated = evaluate_point_policy_with_total_fallback(
        kernel,
        value,
        point_kernel,
        point_belief,
        horizon,
        fallback_action=fallback_action,
        tie_tolerance=tie_tolerance,
    )
    if int(policy["selected_action"]) != int(evaluated["selected_action"]):
        raise RuntimeError("V68r2 MAP root action changed during totalized evaluation")
    return {
        "static_index": static_index,
        "static_mass": float(mass),
        "policy": policy,
        "exact_environment_value": float(evaluated["value"]),
        "off_support_branch_count": int(evaluated["off_support_branch_count"]),
        "expected_off_support_entry_probability": float(
            evaluated["expected_off_support_entry_probability"]
        ),
        "fallback_action": fallback_action,
        "fallback_action_name": kernel.action_names[fallback_action],
        "point_model_persists_on_supported_histories": True,
        "off_support_fallback_is_open_loop": True,
        "off_support_model_reselection": False,
        "epsilon_smoothing": False,
    }


__all__ = [
    "totalized_map_model_policy",
    "totalized_persistent_posterior_sampling_mixture",
]
