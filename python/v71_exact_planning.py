#!/usr/bin/env python3
"""Exact finite-horizon planning for the V71 sensor-codebook family."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import numpy as np

from v71_cassandra_pomdp import ParsedPOMDPSource
from v71_sensor_codebook import LATENT_NAMES, sensor_observation_models


@dataclass(frozen=True)
class SensorCodebookKernel:
    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    state_names: tuple[str, ...]
    transition: np.ndarray  # action, state, successor
    observation: np.ndarray  # latent, action, successor, observation
    reward: np.ndarray  # action, state, successor
    discount: float

    def __post_init__(self) -> None:
        actions = len(self.action_names)
        states = len(self.state_names)
        observations = len(self.observation_names)
        transition = np.asarray(self.transition, dtype=np.float64)
        observation = np.asarray(self.observation, dtype=np.float64)
        reward = np.asarray(self.reward, dtype=np.float64)
        if transition.shape != (actions, states, states):
            raise ValueError("V71 transition shape mismatch")
        if observation.shape != (len(LATENT_NAMES), actions, states, observations):
            raise ValueError("V71 observation shape mismatch")
        if reward.shape != (actions, states, states):
            raise ValueError("V71 reward shape mismatch")
        if not all(np.isfinite(value).all() for value in (transition, observation, reward)):
            raise ValueError("V71 kernel contains a non-finite value")
        if np.any(transition < 0.0) or np.any(observation < 0.0):
            raise ValueError("V71 kernel probabilities must be nonnegative")
        if not np.allclose(transition.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V71 transitions are not normalized")
        if not np.allclose(observation.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V71 observations are not normalized")
        if not np.array_equal(observation[0] > 0.0, observation[1] > 0.0):
            raise ValueError("V71 point-model observation supports differ")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("V71 discount must lie in (0,1]")
        for value in (transition, observation, reward):
            value.setflags(write=False)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "reward", reward)


def kernel_from_parsed(
    parsed: ParsedPOMDPSource, *, reliability: float = 0.85
) -> SensorCodebookKernel:
    model = parsed.model
    return SensorCodebookKernel(
        action_names=model.actions,
        observation_names=model.observations,
        state_names=model.states,
        transition=model.transition,
        observation=sensor_observation_models(parsed, reliability=reliability),
        reward=model.reward,
        discount=model.discount,
    )


def _joint_belief(kernel: SensorCodebookKernel, belief: np.ndarray) -> np.ndarray:
    value = np.asarray(belief, dtype=np.float64)
    if value.shape != (len(LATENT_NAMES), len(kernel.state_names)):
        raise ValueError("V71 joint belief shape mismatch")
    if np.any(value < 0.0) or not np.isfinite(value).all():
        raise ValueError("V71 joint belief is invalid")
    if not np.isclose(value.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("V71 joint belief is not normalized")
    return value


def _state_belief(kernel: SensorCodebookKernel, belief: np.ndarray) -> np.ndarray:
    value = np.asarray(belief, dtype=np.float64)
    if value.shape != (len(kernel.state_names),):
        raise ValueError("V71 point state belief shape mismatch")
    if np.any(value < 0.0) or not np.isfinite(value).all():
        raise ValueError("V71 point state belief is invalid")
    if not np.isclose(value.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("V71 point state belief is not normalized")
    return value


def _select(values: Sequence[float], tolerance: float) -> tuple[int, tuple[int, ...], float]:
    maximum = max(float(value) for value in values)
    optimal = tuple(
        index for index, value in enumerate(values) if maximum - float(value) <= tolerance
    )
    return optimal[0], optimal, maximum


def exact_step(
    kernel: SensorCodebookKernel, belief: np.ndarray, action: int
) -> dict[str, Any]:
    value = _joint_belief(kernel, belief)
    predicted = np.einsum(
        "zs,sq->zq", value, kernel.transition[action], optimize=True
    )
    joint = predicted[:, :, None] * kernel.observation[:, action]
    probabilities = joint.sum(axis=(0, 1))
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("V71 exact predictive observations do not normalize")
    posteriors = {
        observation: joint[:, :, observation] / float(probability)
        for observation, probability in enumerate(probabilities)
        if probability > 0.0
    }
    immediate = float(
        np.einsum(
            "zs,sq,sq->",
            value,
            kernel.transition[action],
            kernel.reward[action],
            optimize=True,
        )
    )
    return {
        "reward": immediate,
        "probabilities": probabilities,
        "posteriors": posteriors,
    }


def point_step(
    kernel: SensorCodebookKernel, state_belief: np.ndarray, latent: int, action: int
) -> dict[str, Any]:
    value = _state_belief(kernel, state_belief)
    predicted = value @ kernel.transition[action]
    joint = predicted[:, None] * kernel.observation[latent, action]
    probabilities = joint.sum(axis=0)
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("V71 point predictive observations do not normalize")
    posteriors = {
        observation: joint[:, observation] / float(probability)
        for observation, probability in enumerate(probabilities)
        if probability > 0.0
    }
    immediate = float(
        np.einsum(
            "s,sq,sq->",
            value,
            kernel.transition[action],
            kernel.reward[action],
            optimize=True,
        )
    )
    return {
        "reward": immediate,
        "probabilities": probabilities,
        "posteriors": posteriors,
    }


def plan_exact(
    kernel: SensorCodebookKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    value = _joint_belief(kernel, belief)
    if horizon < 0:
        raise ValueError("V71 horizon cannot be negative")
    if stats is not None:
        stats["bellman_nodes"] = stats.get("bellman_nodes", 0) + 1
    if horizon == 0:
        return {"terminal": True, "horizon": 0, "value": 0.0}
    rows = []
    for action in range(len(kernel.action_names)):
        step = exact_step(kernel, value, action)
        branches: dict[int, dict[str, Any]] = {}
        continuation = 0.0
        if horizon > 1:
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
        rows.append(
            {
                "action": action,
                "value": float(step["reward"] + kernel.discount * continuation),
                "branches": branches,
            }
        )
    q_values = tuple(float(row["value"]) for row in rows)
    selected, optimal, maximum = _select(q_values, tie_tolerance)
    selected_row = rows[selected]
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(maximum),
        "selected_action": selected,
        "optimal_actions": optimal,
        "q_values": q_values,
        "branches": selected_row["branches"],
    }


def plan_point(
    kernel: SensorCodebookKernel,
    state_belief: np.ndarray,
    latent: int,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = _state_belief(kernel, state_belief)
    if latent not in range(len(LATENT_NAMES)):
        raise ValueError("V71 point latent index is invalid")
    if horizon == 0:
        return {"terminal": True, "horizon": 0, "value": 0.0}
    rows = []
    for action in range(len(kernel.action_names)):
        step = point_step(kernel, value, latent, action)
        branches: dict[int, dict[str, Any]] = {}
        continuation = 0.0
        if horizon > 1:
            for observation, posterior in step["posteriors"].items():
                child = plan_point(
                    kernel,
                    posterior,
                    latent,
                    horizon - 1,
                    tie_tolerance=tie_tolerance,
                )
                branches[observation] = child
                continuation += float(step["probabilities"][observation]) * float(
                    child["value"]
                )
        rows.append(
            {
                "action": action,
                "value": float(step["reward"] + kernel.discount * continuation),
                "branches": branches,
            }
        )
    q_values = tuple(float(row["value"]) for row in rows)
    selected, optimal, maximum = _select(q_values, tie_tolerance)
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(maximum),
        "selected_action": selected,
        "optimal_actions": optimal,
        "q_values": q_values,
        "branches": rows[selected]["branches"],
        "point_latent": latent,
    }


def evaluate_policy_exact(
    kernel: SensorCodebookKernel,
    belief: np.ndarray,
    policy: dict[str, Any],
    horizon: int,
) -> float:
    value = _joint_belief(kernel, belief)
    if horizon == 0:
        return 0.0
    if policy.get("terminal") or int(policy.get("horizon", -1)) != horizon:
        raise ValueError("V71 policy horizon or terminal marker is invalid")
    action = int(policy["selected_action"])
    step = exact_step(kernel, value, action)
    continuation = 0.0
    if horizon > 1:
        for observation, posterior in step["posteriors"].items():
            if observation not in policy["branches"]:
                raise RuntimeError("V71 point policy reached an off-support branch")
            continuation += float(step["probabilities"][observation]) * evaluate_policy_exact(
                kernel, posterior, policy["branches"][observation], horizon - 1
            )
    return float(step["reward"] + kernel.discount * continuation)


def map_control(
    kernel: SensorCodebookKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = _joint_belief(kernel, belief)
    masses = value.sum(axis=1)
    latent = int(np.argmax(masses))
    state = value[latent] / float(masses[latent])
    policy = plan_point(
        kernel, state, latent, horizon, tie_tolerance=tie_tolerance
    )
    return {
        "latent": latent,
        "latent_name": LATENT_NAMES[latent],
        "latent_mass": float(masses[latent]),
        "policy": policy,
        "value": evaluate_policy_exact(kernel, value, policy, horizon),
        "on_support": True,
    }


def posterior_sampling_control(
    kernel: SensorCodebookKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = _joint_belief(kernel, belief)
    masses = value.sum(axis=1)
    rows = []
    expected = 0.0
    root_distribution = np.zeros(len(kernel.action_names), dtype=np.float64)
    for latent, mass in enumerate(masses):
        if mass <= 0.0:
            continue
        state = value[latent] / float(mass)
        policy = plan_point(
            kernel, state, latent, horizon, tie_tolerance=tie_tolerance
        )
        exact_value = evaluate_policy_exact(kernel, value, policy, horizon)
        expected += float(mass) * exact_value
        root_distribution[int(policy["selected_action"])] += float(mass)
        rows.append(
            {
                "latent": latent,
                "latent_name": LATENT_NAMES[latent],
                "mass": float(mass),
                "selected_action": int(policy["selected_action"]),
                "exact_environment_value": exact_value,
            }
        )
    return {
        "value": float(expected),
        "root_action_distribution": root_distribution.tolist(),
        "models": rows,
        "sampled_model_persists_for_full_policy": True,
        "on_support": True,
    }


def evaluate_action_sequence(
    kernel: SensorCodebookKernel, belief: np.ndarray, actions: Sequence[int]
) -> float:
    value = _joint_belief(kernel, belief)
    if not actions:
        return 0.0
    step = exact_step(kernel, value, int(actions[0]))
    continuation = 0.0
    if len(actions) > 1:
        continuation = sum(
            float(step["probabilities"][observation])
            * evaluate_action_sequence(kernel, posterior, actions[1:])
            for observation, posterior in step["posteriors"].items()
        )
    return float(step["reward"] + kernel.discount * continuation)


def best_open_loop_sequence(
    kernel: SensorCodebookKernel,
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
    optimal = tuple(actions for actions, value in rows if maximum - value <= tie_tolerance)
    return {
        "value": float(maximum),
        "selected_actions": optimal[0],
        "optimal_sequences": optimal,
        "sequence_count": len(rows),
    }


def plan_myopic(
    kernel: SensorCodebookKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = _joint_belief(kernel, belief)
    if horizon == 0:
        return {"terminal": True, "horizon": 0, "value": 0.0}
    steps = [exact_step(kernel, value, action) for action in range(len(kernel.action_names))]
    scores = tuple(float(step["reward"]) for step in steps)
    selected, optimal, maximum = _select(scores, tie_tolerance)
    step = steps[selected]
    branches = {
        observation: plan_myopic(
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
        "optimal_actions": optimal,
        "q_values": scores,
        "branches": branches,
    }


def finite_horizon_return_scale(kernel: SensorCodebookKernel, horizon: int) -> float:
    reward_span = float(kernel.reward.max() - kernel.reward.min())
    discount_sum = sum(kernel.discount**depth for depth in range(horizon))
    return float(max(1.0, reward_span * discount_sum))
