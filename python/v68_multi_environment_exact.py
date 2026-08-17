#!/usr/bin/env python3
"""Exact, environment-generic infrastructure for the V68 development screen."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from typing import Sequence

import numpy as np

from v62_external_pomdp import POMDPModel
from v64_external_eig import scaled_beta_2_2_quadrature
from v66_bayes_adaptive_reward import StaticKernel, _assert_belief, step_belief


@dataclass(frozen=True)
class CommandChannelFamily:
    """A persistent two-identity command-substitution family on one POMDP."""

    model: POMDPModel
    kernel: StaticKernel
    initial_belief: np.ndarray
    theta: np.ndarray
    theta_weights: np.ndarray
    permutations: np.ndarray
    canonical_action_labels: tuple[str, ...]
    identity_names: tuple[str, str]
    theta_support: tuple[float, float]

    def __post_init__(self) -> None:
        nodes = len(self.theta)
        action_count = len(self.model.actions)
        expected = {
            "initial_belief": (2 * nodes, len(self.model.states)),
            "theta": (nodes,),
            "theta_weights": (nodes,),
            "permutations": (2, action_count),
        }
        for name, shape in expected.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape:
                raise ValueError(f"V68 {name} shape {array.shape} != {shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"V68 {name} contains non-finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if len(self.identity_names) != 2:
            raise ValueError("V68 requires two identity names")
        if set(self.canonical_action_labels) != set(self.model.actions):
            raise ValueError("V68 canonical action cycle must enumerate every source action")
        if abs(float(self.theta_weights.sum()) - 1.0) > 1e-12:
            raise ValueError("V68 theta weights do not normalize")
        _assert_belief(self.kernel, self.initial_belief)


@dataclass(frozen=True)
class PublicPrefix:
    record_id: str
    model_name: str
    actions: tuple[int, ...]
    observations: tuple[int, ...]
    belief: np.ndarray
    probability: float
    log_evidence: float

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.observations):
            raise ValueError("V68 public prefix action/observation lengths differ")
        if not 0.0 < self.probability <= 1.0 + 1e-12:
            raise ValueError("V68 public prefix probability is invalid")
        belief = np.asarray(self.belief, dtype=np.float64)
        if np.any(~np.isfinite(belief)) or np.min(belief) < -1e-15:
            raise ValueError("V68 public prefix belief is invalid")
        if abs(float(belief.sum()) - 1.0) > 1e-10:
            raise ValueError("V68 public prefix belief does not normalize")
        belief.setflags(write=False)
        object.__setattr__(self, "belief", belief)

    @property
    def depth(self) -> int:
        return len(self.actions)


def _action_cycle_indices(
    model: POMDPModel, canonical_action_cycle: Sequence[str]
) -> tuple[int, ...]:
    labels = tuple(str(label) for label in canonical_action_cycle)
    if len(labels) != len(model.actions) or set(labels) != set(model.actions):
        raise ValueError("V68 action cycle must contain every source action exactly once")
    return tuple(model.actions.index(label) for label in labels)


def cycle_permutations(
    model: POMDPModel, canonical_action_cycle: Sequence[str]
) -> tuple[np.ndarray, tuple[int, ...]]:
    canonical = _action_cycle_indices(model, canonical_action_cycle)
    position = {action: offset for offset, action in enumerate(canonical)}
    forward = np.asarray(
        [canonical[(position[action] + 1) % len(canonical)] for action in range(len(canonical))],
        dtype=np.int64,
    )
    backward = np.asarray(
        [canonical[(position[action] - 1) % len(canonical)] for action in range(len(canonical))],
        dtype=np.int64,
    )
    return np.stack([forward, backward]), canonical


def build_command_channel_family(
    model: POMDPModel,
    canonical_action_cycle: Sequence[str],
    *,
    quadrature_nodes: int,
    theta_support: tuple[float, float] = (0.6, 0.95),
) -> CommandChannelFamily:
    """Construct the V64 family on any valid source action alphabet."""
    low, high = map(float, theta_support)
    theta, theta_weights = scaled_beta_2_2_quadrature(quadrature_nodes, low, high)
    permutations, canonical = cycle_permutations(model, canonical_action_cycle)
    transitions = np.asarray(
        [
            [
                [
                    value * model.transition[action]
                    + (1.0 - value) * model.transition[permutation[action]]
                    for action in range(len(model.actions))
                ]
                for value in theta
            ]
            for permutation in permutations
        ],
        dtype=np.float64,
    )
    static_weights = np.concatenate([0.5 * theta_weights, 0.5 * theta_weights])
    identities = np.repeat(np.arange(2, dtype=np.int16), quadrature_nodes)
    thetas = np.tile(theta, 2)
    kernel = StaticKernel(
        action_names=model.actions,
        observation_names=model.observations,
        state_names=model.states,
        canonical_actions=canonical,
        transitions=transitions.reshape(
            2 * quadrature_nodes,
            len(model.actions),
            len(model.states),
            len(model.states),
        ),
        observations=model.observation,
        rewards=model.reward,
        discount=model.discount,
        identities=identities,
        thetas=thetas,
    )
    initial_belief = static_weights[:, None] * model.initial[None, :]
    return CommandChannelFamily(
        model=model,
        kernel=kernel,
        initial_belief=initial_belief,
        theta=theta,
        theta_weights=theta_weights,
        permutations=permutations,
        canonical_action_labels=tuple(str(label) for label in canonical_action_cycle),
        identity_names=("forward_cycle_failure", "backward_cycle_failure"),
        theta_support=(low, high),
    )


def action_index(family: CommandChannelFamily, action: int | str) -> int:
    if isinstance(action, str):
        try:
            return family.model.actions.index(action)
        except ValueError as exc:
            raise ValueError(f"unknown V68 action {action!r}") from exc
    if action not in range(len(family.model.actions)):
        raise ValueError(f"invalid V68 action index {action}")
    return int(action)


def observation_index(family: CommandChannelFamily, observation: int | str) -> int:
    if isinstance(observation, str):
        try:
            return family.model.observations.index(observation)
        except ValueError as exc:
            raise ValueError(f"unknown V68 observation {observation!r}") from exc
    if observation not in range(len(family.model.observations)):
        raise ValueError(f"invalid V68 observation index {observation}")
    return int(observation)


def filter_action_observation_history(
    family: CommandChannelFamily,
    actions: Sequence[int | str],
    observations: Sequence[int | str],
) -> tuple[np.ndarray, float]:
    if len(actions) != len(observations):
        raise ValueError("V68 history action/observation lengths differ")
    belief = np.asarray(family.initial_belief, dtype=np.float64).copy()
    log_evidence = 0.0
    for raw_action, raw_observation in zip(actions, observations):
        action = action_index(family, raw_action)
        observation = observation_index(family, raw_observation)
        step = step_belief(family.kernel, belief, action)
        probability = float(step["probabilities"][observation])
        if probability <= 0.0 or observation not in step["posteriors"]:
            raise ValueError("impossible V68 public history")
        belief = step["posteriors"][observation]
        log_evidence += math.log(probability)
    _assert_belief(family.kernel, belief)
    return belief, float(log_evidence)


def _record_id(model_name: str, actions: Sequence[int], observations: Sequence[int]) -> str:
    payload = json.dumps(
        {"model": model_name, "actions": list(actions), "observations": list(observations)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{model_name}-d{len(actions)}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def enumerate_public_prefixes(
    family: CommandChannelFamily, *, maximum_depth: int
) -> list[PublicPrefix]:
    """Retain every positive-probability action/observation history through a depth."""
    if maximum_depth < 0:
        raise ValueError("V68 maximum prefix depth cannot be negative")
    root = PublicPrefix(
        record_id=_record_id(family.model.name, (), ()),
        model_name=family.model.name,
        actions=(),
        observations=(),
        belief=np.asarray(family.initial_belief, dtype=np.float64),
        probability=1.0,
        log_evidence=0.0,
    )
    retained = [root]
    frontier = [root]
    for _ in range(maximum_depth):
        next_frontier: list[PublicPrefix] = []
        for prefix in frontier:
            for action in family.kernel.canonical_actions:
                step = step_belief(family.kernel, prefix.belief, action)
                for observation in sorted(step["posteriors"]):
                    conditional = float(step["probabilities"][observation])
                    probability = prefix.probability * conditional
                    actions = prefix.actions + (int(action),)
                    observations = prefix.observations + (int(observation),)
                    next_frontier.append(
                        PublicPrefix(
                            record_id=_record_id(family.model.name, actions, observations),
                            model_name=family.model.name,
                            actions=actions,
                            observations=observations,
                            belief=step["posteriors"][observation],
                            probability=probability,
                            log_evidence=prefix.log_evidence + math.log(conditional),
                        )
                    )
        retained.extend(next_frontier)
        frontier = next_frontier
    if len({row.record_id for row in retained}) != len(retained):
        raise RuntimeError("V68 public prefix IDs are not unique")
    return retained


def evaluate_action_sequence(
    kernel: StaticKernel, belief: np.ndarray, actions: Sequence[int]
) -> float:
    """Evaluate an open-loop sequence while filtering observations exactly."""
    value = _assert_belief(kernel, belief)
    if not actions:
        return 0.0
    action = int(actions[0])
    if action not in kernel.canonical_actions:
        raise ValueError("V68 open-loop sequence contains a noncanonical action")
    step = step_belief(kernel, value, action)
    continuation = 0.0
    if len(actions) > 1:
        continuation = sum(
            float(step["probabilities"][observation])
            * evaluate_action_sequence(kernel, posterior, actions[1:])
            for observation, posterior in step["posteriors"].items()
        )
    return float(step["reward"] + kernel.discount * continuation)


def best_open_loop_sequence(
    kernel: StaticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, object]:
    if horizon <= 0:
        raise ValueError("V68 open-loop horizon must be positive")
    rows = []
    for actions in product(kernel.canonical_actions, repeat=horizon):
        rows.append((tuple(int(action) for action in actions), evaluate_action_sequence(kernel, belief, actions)))
    maximum = max(value for _, value in rows)
    optimal = tuple(actions for actions, value in rows if maximum - value <= tie_tolerance)
    return {
        "value": float(maximum),
        "selected_actions": optimal[0],
        "selected_action_names": tuple(kernel.action_names[action] for action in optimal[0]),
        "optimal_sequences": optimal,
        "sequence_count": len(rows),
    }


def finite_horizon_return_scale(model: POMDPModel, horizon: int) -> float:
    if horizon <= 0:
        raise ValueError("V68 return-scale horizon must be positive")
    reward_span = float(model.reward.max() - model.reward.min())
    discount_sum = sum(model.discount**depth for depth in range(horizon))
    return float(max(1.0, reward_span * discount_sum))
