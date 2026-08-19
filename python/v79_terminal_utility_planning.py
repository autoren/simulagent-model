#!/usr/bin/env python3
"""Exact finite-horizon planning with V79 unresolved terminal utility."""
from __future__ import annotations

from itertools import product
from typing import Any, Sequence

import numpy as np

from v77_clarification_benchmark import (
    ClarificationKernel,
    certified_actions,
    exact_step,
    finite_horizon_return_scale,
    validate_belief,
)


ACTIVE_UNRESOLVED_TERMINAL_UTILITY = -6.0
TERMINAL_STATE_TERMINAL_UTILITY = 0.0


def terminal_utility(kernel: ClarificationKernel, belief: np.ndarray) -> float:
    value = validate_belief(kernel, belief)
    active_mass = float(value[:, 0].sum())
    terminal_mass = float(value[:, 1].sum())
    return float(
        active_mass * ACTIVE_UNRESOLVED_TERMINAL_UTILITY
        + terminal_mass * TERMINAL_STATE_TERMINAL_UTILITY
    )


def terminal_belief(kernel: ClarificationKernel, belief: np.ndarray) -> bool:
    return bool(validate_belief(kernel, belief)[:, 1].sum() >= 1.0 - 1e-12)


def select(
    rows: Sequence[tuple[int, float]], tolerance: float
) -> tuple[int, tuple[int, ...], float]:
    maximum = max(float(value) for _, value in rows)
    optimal = tuple(action for action, value in rows if maximum - value <= tolerance)
    return optimal[0], optimal, maximum


def plan_exact(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    if horizon < 0:
        raise ValueError("V79 horizon cannot be negative")
    if stats is not None:
        stats["belief_checks"] = stats.get("belief_checks", 0) + 1
        stats["normalized_belief_checks"] = (
            stats.get("normalized_belief_checks", 0) + 1
        )
        stats["bellman_nodes"] = stats.get("bellman_nodes", 0) + 1
    if terminal_belief(kernel, value):
        return {"terminal": True, "horizon": horizon, "value": 0.0}
    if horizon == 0:
        return {
            "terminal": True,
            "horizon": 0,
            "value": terminal_utility(kernel, value),
            "unresolved_terminal": True,
        }
    candidates: list[dict[str, Any]] = []
    for action in certified_actions(kernel, value):
        step = exact_step(kernel, value, action)
        branches: dict[int, dict[str, Any]] = {}
        continuation = 0.0
        for observation, posterior in step["posteriors"].items():
            child = plan_exact(
                kernel,
                posterior,
                horizon - 1,
                tie_tolerance=tie_tolerance,
                stats=stats,
            )
            branches[observation] = child
            continuation += float(step["probabilities"][observation]) * float(
                child["value"]
            )
        candidates.append(
            {
                "action": action,
                "value": float(step["reward"] + kernel.discount * continuation),
                "branches": branches,
            }
        )
    selected, optimal, maximum = select(
        [(row["action"], row["value"]) for row in candidates], tie_tolerance
    )
    selected_row = next(row for row in candidates if row["action"] == selected)
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(maximum),
        "selected_action": selected,
        "optimal_actions": optimal,
        "q_values": {row["action"]: row["value"] for row in candidates},
        "certified_actions": certified_actions(kernel, value),
        "hypothesis_masses": value.sum(axis=1).tolist(),
        "branches": selected_row["branches"],
    }


def evaluate_policy_exact(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    policy: dict[str, Any],
    horizon: int,
    *,
    certificate_violations: list[dict[str, Any]] | None = None,
) -> float:
    value = validate_belief(kernel, belief)
    if terminal_belief(kernel, value):
        return 0.0
    if horizon == 0:
        return terminal_utility(kernel, value)
    if policy.get("terminal") or int(policy.get("horizon", -1)) != horizon:
        raise ValueError("V79 policy horizon or terminal marker is invalid")
    action = int(policy["selected_action"])
    if certificate_violations is not None and action not in certified_actions(kernel, value):
        certificate_violations.append(
            {
                "horizon": horizon,
                "action": action,
                "hypothesis_masses": value.sum(axis=1).tolist(),
            }
        )
    step = exact_step(kernel, value, action)
    continuation = 0.0
    for observation, posterior in step["posteriors"].items():
        if observation in policy.get("branches", {}):
            child_value = evaluate_policy_exact(
                kernel,
                posterior,
                policy["branches"][observation],
                horizon - 1,
                certificate_violations=certificate_violations,
            )
        elif terminal_belief(kernel, posterior):
            child_value = 0.0
        else:
            raise RuntimeError("V79 policy omits a reachable unresolved branch")
        continuation += float(step["probabilities"][observation]) * child_value
    return float(step["reward"] + kernel.discount * continuation)


def point_policy(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    hypothesis: int,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    masses = value.sum(axis=1)
    if hypothesis not in range(len(kernel.hypothesis_names)) or masses[hypothesis] <= 0.0:
        raise ValueError("V79 point hypothesis has zero or invalid mass")
    point = np.zeros_like(value)
    point[hypothesis] = value[hypothesis] / float(masses[hypothesis])
    return plan_exact(kernel, point, horizon, tie_tolerance=tie_tolerance)


def map_control(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    masses = value.sum(axis=1)
    hypothesis = int(np.argmax(masses))
    policy = point_policy(
        kernel, value, hypothesis, horizon, tie_tolerance=tie_tolerance
    )
    violations: list[dict[str, Any]] = []
    exact_value = evaluate_policy_exact(
        kernel, value, policy, horizon, certificate_violations=violations
    )
    return {
        "hypothesis": hypothesis,
        "hypothesis_name": kernel.hypothesis_names[hypothesis],
        "hypothesis_mass": float(masses[hypothesis]),
        "policy": policy,
        "value": exact_value,
        "complete_belief_certificate_violations": violations,
        "shadow_only": True,
        "off_support_fallback_count": 0,
    }


def posterior_sampling_control(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    masses = value.sum(axis=1)
    expected = 0.0
    root_distribution = np.zeros(len(kernel.action_names), dtype=np.float64)
    rows = []
    violations: list[dict[str, Any]] = []
    for hypothesis, mass in enumerate(masses):
        if mass <= 0.0:
            continue
        policy = point_policy(
            kernel, value, hypothesis, horizon, tie_tolerance=tie_tolerance
        )
        local_violations: list[dict[str, Any]] = []
        exact_value = evaluate_policy_exact(
            kernel,
            value,
            policy,
            horizon,
            certificate_violations=local_violations,
        )
        expected += float(mass) * exact_value
        root_distribution[int(policy["selected_action"])] += float(mass)
        violations.extend(local_violations)
        rows.append(
            {
                "hypothesis": hypothesis,
                "hypothesis_name": kernel.hypothesis_names[hypothesis],
                "mass": float(mass),
                "root_action": int(policy["selected_action"]),
                "exact_environment_value": exact_value,
                "complete_belief_certificate_violation_count": len(local_violations),
            }
        )
    return {
        "value": float(expected),
        "root_action_distribution": root_distribution.tolist(),
        "hypotheses": rows,
        "complete_belief_certificate_violations": violations,
        "sampled_hypothesis_persists_for_full_policy": True,
        "shadow_only": True,
        "off_support_fallback_count": 0,
    }


def evaluate_action_sequence(
    kernel: ClarificationKernel, belief: np.ndarray, actions: Sequence[int]
) -> float:
    value = validate_belief(kernel, belief)
    if terminal_belief(kernel, value):
        return 0.0
    if not actions:
        return terminal_utility(kernel, value)
    step = exact_step(kernel, value, int(actions[0]))
    continuation = sum(
        float(step["probabilities"][observation])
        * evaluate_action_sequence(kernel, posterior, actions[1:])
        for observation, posterior in step["posteriors"].items()
    )
    return float(step["reward"] + kernel.discount * continuation)


def best_open_loop_sequence(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    rows = [
        (actions, evaluate_action_sequence(kernel, belief, actions))
        for actions in product(range(len(kernel.action_names)), repeat=horizon)
    ]
    maximum = max(value for _, value in rows)
    optimal = tuple(
        actions for actions, value in rows if maximum - value <= tie_tolerance
    )
    return {
        "value": float(maximum),
        "selected_actions": optimal[0],
        "optimal_sequences": optimal,
        "sequence_count": len(rows),
    }


def oracle_interpretation_value(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    masses = value.sum(axis=1)
    expected = 0.0
    rows = []
    for hypothesis, mass in enumerate(masses):
        if mass <= 0.0:
            continue
        point = np.zeros_like(value)
        point[hypothesis] = value[hypothesis] / float(mass)
        policy = plan_exact(kernel, point, horizon, tie_tolerance=tie_tolerance)
        conditional = evaluate_policy_exact(kernel, point, policy, horizon)
        expected += float(mass) * conditional
        rows.append(
            {
                "hypothesis": hypothesis,
                "hypothesis_name": kernel.hypothesis_names[hypothesis],
                "mass": float(mass),
                "root_action": int(policy["selected_action"]),
                "conditional_value": conditional,
            }
        )
    return {"value": float(expected), "hypotheses": rows}


__all__ = [
    "ACTIVE_UNRESOLVED_TERMINAL_UTILITY",
    "TERMINAL_STATE_TERMINAL_UTILITY",
    "best_open_loop_sequence",
    "evaluate_policy_exact",
    "finite_horizon_return_scale",
    "map_control",
    "oracle_interpretation_value",
    "plan_exact",
    "posterior_sampling_control",
    "terminal_utility",
]
