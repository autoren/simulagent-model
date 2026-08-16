#!/usr/bin/env python3
"""Bounded Bayes-adaptive reward planning for the frozen V66 external family."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import (
    V64Family,
    filter_public_history,
    score_action as exact_eig_score_action,
    true_transition,
)
from v65r3_smc2_eig import rao_blackwellize_measure


PUBLIC_FIELDS = (
    "record_id",
    "prefix_length",
    "initial_observation",
    "actions",
    "observations",
)


def _history_key(record: dict[str, Any]) -> str:
    if set(record) != set(PUBLIC_FIELDS):
        raise ValueError("V66 planner fixture must contain exactly the public fields")
    return json.dumps(
        {
            "prefix_length": int(record["prefix_length"]),
            "initial_observation": record["initial_observation"],
            "actions": record["actions"],
            "observations": record["observations"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def assert_synthetic_planner_fixture(
    record: dict[str, Any], sealed_records: Sequence[dict[str, Any]]
) -> None:
    """Reject sealed IDs or histories during the implementation-only stage."""
    record_id = str(record["record_id"])
    history = _history_key(record)
    sealed_ids = {str(row["record_id"]) for row in sealed_records}
    sealed_histories = {_history_key(row) for row in sealed_records}
    if record_id in sealed_ids:
        raise PermissionError("V66 implementation planner rejected a sealed record ID")
    if history in sealed_histories:
        raise PermissionError("V66 implementation planner rejected a sealed public history")


@dataclass(frozen=True)
class StaticKernel:
    """A finite static-model POMDP kernel with a joint static/state belief."""

    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    state_names: tuple[str, ...]
    canonical_actions: tuple[int, ...]
    transitions: np.ndarray  # static, action, state, successor
    observations: np.ndarray  # action, successor, observation
    rewards: np.ndarray  # action, state, successor
    discount: float
    identities: np.ndarray
    thetas: np.ndarray

    def __post_init__(self) -> None:
        static_count = len(self.identities)
        action_count = len(self.action_names)
        state_count = len(self.state_names)
        observation_count = len(self.observation_names)
        expected = {
            "transitions": (static_count, action_count, state_count, state_count),
            "observations": (action_count, state_count, observation_count),
            "rewards": (action_count, state_count, state_count),
            "identities": (static_count,),
            "thetas": (static_count,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"V66 {name} shape {value.shape} != {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"V66 {name} contains a non-finite value")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if tuple(sorted(self.canonical_actions)) != tuple(range(action_count)):
            raise ValueError("V66 canonical actions must enumerate every action once")
        if np.min(self.transitions) < -1e-15 or np.max(
            np.abs(self.transitions.sum(axis=-1) - 1.0)
        ) > 1e-11:
            raise ValueError("V66 transition rows do not normalize")
        if np.min(self.observations) < -1e-15 or np.max(
            np.abs(self.observations.sum(axis=-1) - 1.0)
        ) > 1e-11:
            raise ValueError("V66 observation rows do not normalize")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("V66 discount lies outside [0, 1]")


def load_config(path: str | Path = "configs/v66-design-lock.json") -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return json.loads(value.read_text())["config_payload"]


def _assert_belief(kernel: StaticKernel, belief: np.ndarray) -> np.ndarray:
    value = np.asarray(belief, dtype=np.float64)
    expected = (len(kernel.identities), len(kernel.state_names))
    if value.shape != expected:
        raise ValueError(f"V66 belief shape {value.shape} != {expected}")
    if np.any(~np.isfinite(value)) or np.min(value) < -1e-15:
        raise ValueError("V66 belief is not a finite nonnegative measure")
    if abs(float(value.sum()) - 1.0) > 1e-10:
        raise ValueError("V66 belief does not normalize")
    return value


def exact_kernel_and_belief(
    family: V64Family, belief: np.ndarray
) -> tuple[StaticKernel, np.ndarray]:
    expected = (2, len(family.theta), len(family.model.states))
    value = np.asarray(belief, dtype=np.float64)
    if value.shape != expected:
        raise ValueError(f"V66 exact family belief shape {value.shape} != {expected}")
    identities = np.repeat(np.arange(2, dtype=np.int16), len(family.theta))
    thetas = np.tile(family.theta, 2)
    kernel = StaticKernel(
        action_names=family.model.actions,
        observation_names=family.model.observations,
        state_names=family.model.states,
        canonical_actions=family.canonical_actions,
        transitions=family.transitions.reshape(
            2 * len(family.theta),
            len(family.model.actions),
            len(family.model.states),
            len(family.model.states),
        ),
        observations=family.model.observation,
        rewards=family.model.reward,
        discount=family.model.discount,
        identities=identities,
        thetas=thetas,
    )
    flat = value.reshape(2 * len(family.theta), len(family.model.states)).copy()
    _assert_belief(kernel, flat)
    return kernel, flat


def exact_history_kernel_and_belief(
    family: V64Family, record: dict[str, Any]
) -> tuple[StaticKernel, np.ndarray, float]:
    belief, log_evidence = filter_public_history(
        family,
        record["initial_observation"],
        record["actions"],
        record["observations"],
    )
    kernel, flat = exact_kernel_and_belief(family, belief)
    return kernel, flat, float(log_evidence)


def pooled_measure_kernel_and_belief(
    family: V64Family,
    record: dict[str, Any],
    pooled_measure: dict[str, Any],
) -> tuple[StaticKernel, np.ndarray, dict[str, Any]]:
    rb = rao_blackwellize_measure(family, pooled_measure, record)
    atoms = rb["atoms"]
    transitions = np.asarray(
        [
            [
                true_transition(
                    family, int(atom["identity"]), float(atom["theta"]), action
                )
                for action in range(len(family.model.actions))
            ]
            for atom in atoms
        ],
        dtype=np.float64,
    )
    kernel = StaticKernel(
        action_names=family.model.actions,
        observation_names=family.model.observations,
        state_names=family.model.states,
        canonical_actions=family.canonical_actions,
        transitions=transitions,
        observations=family.model.observation,
        rewards=family.model.reward,
        discount=family.model.discount,
        identities=np.asarray([int(atom["identity"]) for atom in atoms], dtype=np.int16),
        thetas=np.asarray([float(atom["theta"]) for atom in atoms], dtype=np.float64),
    )
    belief = np.asarray(
        [float(atom["weight"]) * np.asarray(atom["state"], dtype=np.float64) for atom in atoms],
        dtype=np.float64,
    )
    _assert_belief(kernel, belief)
    return kernel, belief, rb


def point_model_kernel_and_belief(
    kernel: StaticKernel, belief: np.ndarray, static_index: int
) -> tuple[StaticKernel, np.ndarray, float]:
    value = _assert_belief(kernel, belief)
    if static_index not in range(len(kernel.identities)):
        raise ValueError("V66 point-model static index is invalid")
    mass = float(value[static_index].sum())
    if mass <= 0.0:
        raise ValueError("V66 cannot condition on a zero-mass static model")
    point = StaticKernel(
        action_names=kernel.action_names,
        observation_names=kernel.observation_names,
        state_names=kernel.state_names,
        canonical_actions=kernel.canonical_actions,
        transitions=kernel.transitions[static_index : static_index + 1],
        observations=kernel.observations,
        rewards=kernel.rewards,
        discount=kernel.discount,
        identities=kernel.identities[static_index : static_index + 1],
        thetas=kernel.thetas[static_index : static_index + 1],
    )
    point_belief = (value[static_index] / mass)[None, :]
    _assert_belief(point, point_belief)
    return point, point_belief, mass


def expected_reward(kernel: StaticKernel, belief: np.ndarray, action: int) -> float:
    value = _assert_belief(kernel, belief)
    if action not in range(len(kernel.action_names)):
        raise ValueError("V66 reward action is invalid")
    return float(
        np.einsum(
            "ms,mst,st->",
            value,
            kernel.transitions[:, action],
            kernel.rewards[action],
            optimize=True,
        )
    )


def step_belief(
    kernel: StaticKernel, belief: np.ndarray, action: int
) -> dict[str, Any]:
    value = _assert_belief(kernel, belief)
    if action not in range(len(kernel.action_names)):
        raise ValueError("V66 transition action is invalid")
    predicted = np.einsum(
        "ms,mst->mt", value, kernel.transitions[:, action], optimize=True
    )
    joint = predicted[:, :, None] * kernel.observations[action][None, :, :]
    probabilities = joint.sum(axis=(0, 1))
    if abs(float(probabilities.sum()) - 1.0) > 1e-10:
        raise RuntimeError("V66 observation predictive does not normalize")
    posteriors: dict[int, np.ndarray] = {}
    for observation, probability in enumerate(probabilities):
        if probability <= 0.0:
            continue
        posterior = joint[:, :, observation] / float(probability)
        _assert_belief(kernel, posterior)
        posteriors[observation] = posterior
    return {
        "reward": expected_reward(kernel, value, action),
        "probabilities": probabilities,
        "posteriors": posteriors,
    }


def _select_action(
    kernel: StaticKernel, values: Sequence[float], tolerance: float
) -> tuple[int, tuple[int, ...], float]:
    if len(values) != len(kernel.canonical_actions):
        raise ValueError("V66 selection score vector omits an action")
    maximum = max(float(value) for value in values)
    optimal_positions = tuple(
        position
        for position, value in enumerate(values)
        if maximum - float(value) <= tolerance
    )
    if not optimal_positions:
        raise RuntimeError("V66 selection has no optimal action")
    optimal_actions = tuple(kernel.canonical_actions[position] for position in optimal_positions)
    return optimal_actions[0], optimal_actions, maximum


def plan_bayes_adaptive(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
    retain_forced_root_actions: bool = False,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Exact finite-horizon Bellman recursion over a joint static/state belief."""
    value = _assert_belief(kernel, belief)
    if horizon < 0:
        raise ValueError("V66 horizon cannot be negative")
    if stats is not None:
        stats["bellman_nodes"] = stats.get("bellman_nodes", 0) + 1
    if horizon == 0:
        return {"terminal": True, "horizon": 0, "value": 0.0}

    rows = []
    for action in kernel.canonical_actions:
        step = step_belief(kernel, value, action)
        children: dict[int, dict[str, Any]] = {}
        continuation = 0.0
        if horizon > 1:
            for observation, posterior in step["posteriors"].items():
                child = plan_bayes_adaptive(
                    kernel,
                    posterior,
                    horizon - 1,
                    tie_tolerance=tie_tolerance,
                    retain_forced_root_actions=False,
                    stats=stats,
                )
                children[observation] = child
                continuation += float(step["probabilities"][observation]) * float(
                    child["value"]
                )
        rows.append(
            {
                "action": int(action),
                "value": float(step["reward"] + kernel.discount * continuation),
                "branches": children,
                "observation_probabilities": np.asarray(
                    step["probabilities"], dtype=np.float64
                ),
            }
        )
    q_values = tuple(float(row["value"]) for row in rows)
    selected, optimal, maximum = _select_action(kernel, q_values, tie_tolerance)
    selected_row = next(row for row in rows if row["action"] == selected)
    result: dict[str, Any] = {
        "terminal": False,
        "horizon": int(horizon),
        "value": float(maximum),
        "selected_action": int(selected),
        "selected_action_name": kernel.action_names[selected],
        "optimal_actions": optimal,
        "optimal_action_names": tuple(kernel.action_names[action] for action in optimal),
        "q_values": q_values,
        "branches": selected_row["branches"],
        "observation_probabilities": selected_row["observation_probabilities"],
    }
    if retain_forced_root_actions:
        result["forced_action_branches"] = {
            int(row["action"]): row["branches"] for row in rows
        }
    return result


def scalar_plan_bayes_adaptive(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Independent loop-based Bellman reference for small implementation fixtures."""
    value = _assert_belief(kernel, belief)
    if horizon == 0:
        return {"terminal": True, "value": 0.0}
    q_values: list[float] = []
    for action in kernel.canonical_actions:
        immediate = 0.0
        raw = np.zeros(
            (len(kernel.identities), len(kernel.state_names), len(kernel.observation_names)),
            dtype=np.float64,
        )
        for model in range(len(kernel.identities)):
            for state in range(len(kernel.state_names)):
                for successor in range(len(kernel.state_names)):
                    edge = float(value[model, state]) * float(
                        kernel.transitions[model, action, state, successor]
                    )
                    immediate += edge * float(kernel.rewards[action, state, successor])
                    for observation in range(len(kernel.observation_names)):
                        raw[model, successor, observation] += edge * float(
                            kernel.observations[action, successor, observation]
                        )
        continuation = 0.0
        if horizon > 1:
            for observation in range(len(kernel.observation_names)):
                probability = float(raw[:, :, observation].sum())
                if probability <= 0.0:
                    continue
                posterior = raw[:, :, observation] / probability
                continuation += probability * scalar_plan_bayes_adaptive(
                    kernel,
                    posterior,
                    horizon - 1,
                    tie_tolerance=tie_tolerance,
                )["value"]
        q_values.append(float(immediate + kernel.discount * continuation))
    selected, optimal, maximum = _select_action(kernel, q_values, tie_tolerance)
    return {
        "terminal": False,
        "value": float(maximum),
        "selected_action": int(selected),
        "optimal_actions": optimal,
        "q_values": tuple(q_values),
    }


def evaluate_policy(
    kernel: StaticKernel,
    belief: np.ndarray,
    policy: dict[str, Any],
    horizon: int,
) -> float:
    value = _assert_belief(kernel, belief)
    if horizon == 0:
        return 0.0
    if policy.get("terminal") or int(policy.get("horizon", -1)) != horizon:
        raise ValueError("V66 policy horizon or terminal marker is invalid")
    action = int(policy["selected_action"])
    step = step_belief(kernel, value, action)
    total = float(step["reward"])
    if horizon > 1:
        continuation = 0.0
        for observation, posterior in step["posteriors"].items():
            if observation not in policy["branches"]:
                raise RuntimeError(
                    "V66 policy omits an observation reachable under the evaluation belief"
                )
            continuation += float(step["probabilities"][observation]) * evaluate_policy(
                kernel,
                posterior,
                policy["branches"][observation],
                horizon - 1,
            )
        total += kernel.discount * continuation
    return float(total)


def evaluate_root_action_values(
    kernel: StaticKernel,
    belief: np.ndarray,
    policy: dict[str, Any],
    horizon: int,
) -> tuple[float, ...]:
    value = _assert_belief(kernel, belief)
    forced = policy.get("forced_action_branches")
    if horizon <= 0 or not isinstance(forced, dict):
        raise ValueError("V66 root action evaluation requires retained forced branches")
    result = []
    for action in kernel.canonical_actions:
        step = step_belief(kernel, value, action)
        continuation = 0.0
        if horizon > 1:
            branches = forced[action]
            for observation, posterior in step["posteriors"].items():
                if observation not in branches:
                    raise RuntimeError(
                        "V66 forced root action omits an evaluation-reachable observation"
                    )
                continuation += float(step["probabilities"][observation]) * evaluate_policy(
                    kernel, posterior, branches[observation], horizon - 1
                )
        result.append(float(step["reward"] + kernel.discount * continuation))
    return tuple(result)


def static_entropy(belief: np.ndarray) -> float:
    probabilities = np.asarray(belief, dtype=np.float64).sum(axis=1)
    return -sum(float(p) * math.log(float(p)) for p in probabilities if p > 0.0)


def evaluate_policy_information(
    kernel: StaticKernel,
    belief: np.ndarray,
    policy: dict[str, Any],
    horizon: int,
) -> float:
    """Expected reduction in static-model entropy under an exact environment belief."""
    value = _assert_belief(kernel, belief)
    initial_entropy = static_entropy(value)

    def terminal_entropy(
        current: np.ndarray, node: dict[str, Any], remaining: int
    ) -> float:
        if remaining == 0:
            return static_entropy(current)
        action = int(node["selected_action"])
        step = step_belief(kernel, current, action)
        if remaining == 1:
            return sum(
                float(step["probabilities"][observation]) * static_entropy(posterior)
                for observation, posterior in step["posteriors"].items()
            )
        return sum(
            float(step["probabilities"][observation])
            * terminal_entropy(
                posterior, node["branches"][observation], remaining - 1
            )
            for observation, posterior in step["posteriors"].items()
        )

    return float(initial_entropy - terminal_entropy(value, policy, horizon))


def plan_myopic_reward_policy(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = _assert_belief(kernel, belief)
    if horizon == 0:
        return {"terminal": True, "horizon": 0, "value": 0.0}
    scores = [expected_reward(kernel, value, action) for action in kernel.canonical_actions]
    selected, optimal, maximum = _select_action(kernel, scores, tie_tolerance)
    step = step_belief(kernel, value, selected)
    children = {
        observation: plan_myopic_reward_policy(
            kernel, posterior, horizon - 1, tie_tolerance=tie_tolerance
        )
        for observation, posterior in step["posteriors"].items()
        if horizon > 1
    }
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(maximum),
        "selected_action": selected,
        "selected_action_name": kernel.action_names[selected],
        "optimal_actions": optimal,
        "q_values": tuple(float(score) for score in scores),
        "branches": children,
        "observation_probabilities": step["probabilities"],
    }


def static_eig(kernel: StaticKernel, belief: np.ndarray, action: int) -> float:
    value = _assert_belief(kernel, belief)
    step = step_belief(kernel, value, action)
    prior = value.sum(axis=1)
    information = 0.0
    for observation, posterior in step["posteriors"].items():
        probability = float(step["probabilities"][observation])
        target = posterior.sum(axis=1)
        mask = target > 0.0
        if np.any(mask & (prior <= 0.0)):
            raise RuntimeError("V66 EIG posterior lies outside static prior support")
        information += probability * float(
            np.sum(target[mask] * np.log(target[mask] / prior[mask]))
        )
    return float(information)


def plan_information_only_policy(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = _assert_belief(kernel, belief)
    if horizon == 0:
        return {"terminal": True, "horizon": 0, "value": 0.0}
    scores = [static_eig(kernel, value, action) for action in kernel.canonical_actions]
    selected, optimal, maximum = _select_action(kernel, scores, tie_tolerance)
    step = step_belief(kernel, value, selected)
    children = {
        observation: plan_information_only_policy(
            kernel, posterior, horizon - 1, tie_tolerance=tie_tolerance
        )
        for observation, posterior in step["posteriors"].items()
        if horizon > 1
    }
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(maximum),
        "selected_action": selected,
        "selected_action_name": kernel.action_names[selected],
        "optimal_actions": optimal,
        "q_values": tuple(float(score) for score in scores),
        "branches": children,
        "observation_probabilities": step["probabilities"],
    }


def posterior_weighted_model_oracle(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = _assert_belief(kernel, belief)
    rows = []
    total = 0.0
    for static_index, mass in enumerate(value.sum(axis=1)):
        if mass <= 0.0:
            continue
        point_kernel, point_belief, _ = point_model_kernel_and_belief(
            kernel, value, static_index
        )
        decision = plan_bayes_adaptive(
            point_kernel, point_belief, horizon, tie_tolerance=tie_tolerance
        )
        contribution = float(mass) * float(decision["value"])
        total += contribution
        rows.append(
            {
                "static_index": static_index,
                "mass": float(mass),
                "value": float(decision["value"]),
                "selected_action": int(decision["selected_action"]),
            }
        )
    return {"value": float(total), "models": rows}


def map_model_policy(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = _assert_belief(kernel, belief)
    static_index = int(np.argmax(value.sum(axis=1)))
    point_kernel, point_belief, mass = point_model_kernel_and_belief(
        kernel, value, static_index
    )
    policy = plan_bayes_adaptive(
        point_kernel, point_belief, horizon, tie_tolerance=tie_tolerance
    )
    return {
        "static_index": static_index,
        "static_mass": mass,
        "policy": policy,
        "exact_environment_value": evaluate_policy(kernel, value, policy, horizon),
    }


def systematic_quantile_indices(
    weights: Sequence[float], count: int, offset: float
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.min(values) < 0.0:
        raise ValueError("V66 systematic weights are invalid")
    if count <= 0 or not 0.0 <= offset < 1.0 / count:
        raise ValueError("V66 systematic count or offset is invalid")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("V66 systematic weights have zero mass")
    values = values / total
    positions = offset + np.arange(count, dtype=np.float64) / count
    indices = np.searchsorted(np.cumsum(values), positions, side="right")
    return np.minimum(indices, len(values) - 1).astype(np.int64)


def persistent_posterior_sampling_mixture(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    points: int,
    offset: float,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Deterministic quadrature of policies that sample one persistent model once."""
    value = _assert_belief(kernel, belief)
    static = value.sum(axis=1)
    indices = systematic_quantile_indices(static, points, offset)
    rows = []
    exact_values = []
    root_actions = np.zeros(len(kernel.action_names), dtype=np.float64)
    for static_index in indices:
        point_kernel, point_belief, _ = point_model_kernel_and_belief(
            kernel, value, int(static_index)
        )
        policy = plan_bayes_adaptive(
            point_kernel, point_belief, horizon, tie_tolerance=tie_tolerance
        )
        exact_value = evaluate_policy(kernel, value, policy, horizon)
        exact_values.append(exact_value)
        root_actions[int(policy["selected_action"])] += 1.0 / points
        rows.append(
            {
                "static_index": int(static_index),
                "identity": int(kernel.identities[static_index]),
                "theta": float(kernel.thetas[static_index]),
                "selected_action": int(policy["selected_action"]),
                "exact_environment_value": float(exact_value),
            }
        )
    return {
        "value": float(np.mean(exact_values)),
        "points": int(points),
        "offset": float(offset),
        "selected_static_indices": indices.tolist(),
        "root_action_distribution": root_actions.tolist(),
        "models": rows,
        "sampled_model_persists_for_full_policy": True,
    }


def _mean_transition_point_kernel(
    kernel: StaticKernel, belief: np.ndarray
) -> tuple[StaticKernel, np.ndarray]:
    value = _assert_belief(kernel, belief)
    static = value.sum(axis=1)
    mean_transition = np.einsum("m,mast->ast", static, kernel.transitions, optimize=True)
    state = value.sum(axis=0)[None, :]
    surrogate = StaticKernel(
        action_names=kernel.action_names,
        observation_names=kernel.observation_names,
        state_names=kernel.state_names,
        canonical_actions=kernel.canonical_actions,
        transitions=mean_transition[None, :, :, :],
        observations=kernel.observations,
        rewards=kernel.rewards,
        discount=kernel.discount,
        identities=np.asarray([-1], dtype=np.int16),
        thetas=np.asarray([float(np.sum(static * kernel.thetas))]),
    )
    _assert_belief(surrogate, state)
    return surrogate, state


def plan_invalid_mean_transition_policy(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Negative control: re-average static transitions after every observation history."""
    value = _assert_belief(kernel, belief)
    if horizon == 0:
        return {"terminal": True, "horizon": 0, "value": 0.0}
    surrogate, state = _mean_transition_point_kernel(kernel, value)
    decision = plan_bayes_adaptive(
        surrogate, state, horizon, tie_tolerance=tie_tolerance
    )
    action = int(decision["selected_action"])
    exact_step = step_belief(kernel, value, action)
    children = {
        observation: plan_invalid_mean_transition_policy(
            kernel, posterior, horizon - 1, tie_tolerance=tie_tolerance
        )
        for observation, posterior in exact_step["posteriors"].items()
        if horizon > 1
    }
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(decision["value"]),
        "selected_action": action,
        "selected_action_name": kernel.action_names[action],
        "optimal_actions": decision["optimal_actions"],
        "q_values": decision["q_values"],
        "branches": children,
        "observation_probabilities": exact_step["probabilities"],
        "invalid_static_semantics": "transition_matrices_reaveraged_at_each_history",
    }


def compact_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("terminal"):
        return {"terminal": True, "horizon": 0, "value": 0.0}
    return {
        "terminal": False,
        "horizon": int(policy["horizon"]),
        "value": float(policy["value"]),
        "selected_action": int(policy["selected_action"]),
        "selected_action_name": str(policy["selected_action_name"]),
        "optimal_actions": [int(value) for value in policy["optimal_actions"]],
        "q_values": [float(value) for value in policy["q_values"]],
        "observation_probabilities": [
            float(value) for value in policy["observation_probabilities"]
        ],
        "branches": {
            str(observation): compact_policy(child)
            for observation, child in sorted(policy["branches"].items())
        },
    }


def restore_compact_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("terminal"):
        return {"terminal": True, "horizon": 0, "value": 0.0}
    return {
        **policy,
        "optimal_actions": tuple(int(value) for value in policy["optimal_actions"]),
        "q_values": tuple(float(value) for value in policy["q_values"]),
        "observation_probabilities": np.asarray(
            policy["observation_probabilities"], dtype=np.float64
        ),
        "branches": {
            int(observation): restore_compact_policy(child)
            for observation, child in policy["branches"].items()
        },
    }


def exact_eig_crosscheck(family: V64Family, belief: np.ndarray) -> float:
    """Fixture helper binding generic static EIG to the frozen V64 scorer."""
    kernel, flat = exact_kernel_and_belief(family, belief)
    return max(
        abs(
            static_eig(kernel, flat, action)
            - float(exact_eig_score_action(family, belief, action)["eig"])
        )
        for action in family.canonical_actions
    )
