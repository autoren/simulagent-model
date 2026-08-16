#!/usr/bin/env python3
"""Exact joint filtering and one-step EIG for the frozen V64 actuator family."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import POMDPModel, parse_pomdp_file


@dataclass(frozen=True)
class V64Family:
    model: POMDPModel
    theta: np.ndarray
    theta_weights: np.ndarray
    static_prior: np.ndarray
    transitions: np.ndarray
    permutations: np.ndarray
    canonical_actions: tuple[int, ...]
    identity_names: tuple[str, ...]
    theta_support: tuple[float, float]

    def __post_init__(self) -> None:
        arrays = {
            "theta": (self.theta, (len(self.theta),)),
            "theta_weights": (self.theta_weights, (len(self.theta),)),
            "static_prior": (self.static_prior, (2, len(self.theta))),
            "transitions": (
                self.transitions,
                (
                    2,
                    len(self.theta),
                    len(self.model.actions),
                    len(self.model.states),
                    len(self.model.states),
                ),
            ),
            "permutations": (self.permutations, (2, len(self.model.actions))),
        }
        for name, (value, shape) in arrays.items():
            array = np.asarray(value)
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} != {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if tuple(sorted(self.canonical_actions)) != tuple(range(len(self.model.actions))):
            raise ValueError("canonical actions must be a permutation of every source action")
        if abs(float(self.static_prior.sum()) - 1.0) > 1e-12:
            raise ValueError("static prior does not normalize")
        if np.min(self.transitions) < -1e-15:
            raise ValueError("transition family contains a negative probability")
        if np.max(np.abs(self.transitions.sum(axis=-1) - 1.0)) > 1e-12:
            raise ValueError("transition family rows do not normalize")


def scaled_beta_2_2_quadrature(
    nodes: int, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    if nodes < 2 or not 0.0 <= low < high <= 1.0:
        raise ValueError("invalid V64 quadrature specification")
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(nodes)
    theta = low + (raw_nodes + 1.0) * (high - low) / 2.0
    unit = (theta - low) / (high - low)
    density = 6.0 * unit * (1.0 - unit) / (high - low)
    weights = raw_weights * (high - low) / 2.0 * density
    weights /= weights.sum()
    return theta.astype(np.float64), weights.astype(np.float64)


def _action_structure(model: POMDPModel) -> tuple[np.ndarray, tuple[int, ...]]:
    if set(model.actions) != {"n", "s", "e", "w"}:
        raise ValueError("V64 requires the four named cardinal source actions")
    index = {name: i for i, name in enumerate(model.actions)}
    clockwise = np.asarray(
        [index["e"], index["w"], index["s"], index["n"]], dtype=np.int64
    )
    counterclockwise = np.asarray(
        [index["w"], index["e"], index["n"], index["s"]], dtype=np.int64
    )
    canonical = tuple(index[name] for name in ("n", "e", "s", "w"))
    return np.stack([clockwise, counterclockwise]), canonical


def load_family(
    design_lock: str | Path = "configs/v64-design-lock.json",
    *,
    quadrature_nodes: int | None = None,
) -> V64Family:
    lock_path = Path(design_lock)
    if not lock_path.is_absolute():
        lock_path = PROJECT_ROOT / lock_path
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    external_path = PROJECT_ROOT / config["externalAnchor"]["workspaceModelPath"]
    model = parse_pomdp_file(external_path)
    low, high = map(float, config["unknownDynamicsFamily"]["thetaSupport"])
    nodes = quadrature_nodes or int(config["exactOracle"]["quadratureNodes"])
    theta, theta_weights = scaled_beta_2_2_quadrature(nodes, low, high)
    permutations, canonical = _action_structure(model)
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
    static_prior = np.stack([0.5 * theta_weights, 0.5 * theta_weights])
    return V64Family(
        model=model,
        theta=theta,
        theta_weights=theta_weights,
        static_prior=static_prior,
        transitions=transitions,
        permutations=permutations,
        canonical_actions=canonical,
        identity_names=tuple(config["unknownDynamicsFamily"]["identityNames"]),
        theta_support=(low, high),
    )


def action_index(family: V64Family, action: int | str) -> int:
    if isinstance(action, str):
        try:
            return family.model.actions.index(action)
        except ValueError as exc:
            raise ValueError(f"unknown V64 action {action!r}") from exc
    if action not in range(len(family.model.actions)):
        raise ValueError(f"invalid V64 action index {action}")
    return int(action)


def observation_index(family: V64Family, observation: int | str) -> int:
    if isinstance(observation, str):
        try:
            return family.model.observations.index(observation)
        except ValueError as exc:
            raise ValueError(f"unknown V64 observation {observation!r}") from exc
    if observation not in range(len(family.model.observations)):
        raise ValueError(f"invalid V64 observation index {observation}")
    return int(observation)


def initial_joint_belief(
    family: V64Family, initial_observation: int | str
) -> tuple[np.ndarray, float]:
    observation = observation_index(family, initial_observation)
    # POBAX emits an observation at reset. The pinned O tensor is action invariant;
    # use the first row only after explicitly checking that invariant.
    if not np.allclose(
        family.model.observation,
        family.model.observation[0][None, :, :],
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("V64 reset observation is ambiguous for an action-dependent O")
    weighted = (
        family.static_prior[:, :, None]
        * family.model.initial[None, None, :]
        * family.model.observation[0, :, observation][None, None, :]
    )
    probability = float(weighted.sum())
    if probability <= 0.0:
        raise ValueError("impossible V64 initial observation")
    posterior = weighted / probability
    _assert_belief(family, posterior)
    return posterior, probability


def _assert_belief(family: V64Family, belief: np.ndarray) -> None:
    expected = (2, len(family.theta), len(family.model.states))
    if belief.shape != expected:
        raise ValueError(f"V64 belief shape {belief.shape} != {expected}")
    if not np.all(np.isfinite(belief)) or np.min(belief) < -1e-15:
        raise ValueError("V64 belief is not a finite probability distribution")
    if abs(float(belief.sum()) - 1.0) > 1e-10:
        raise ValueError("V64 belief does not normalize")


def predict_joint_parameter_observation(
    family: V64Family, belief: np.ndarray, action: int | str
) -> tuple[np.ndarray, np.ndarray]:
    _assert_belief(family, belief)
    action_id = action_index(family, action)
    predicted_state = np.einsum(
        "zqs,zqst->zqt", belief, family.transitions[:, :, action_id]
    )
    joint = np.einsum(
        "zqs,so->zqo", predicted_state, family.model.observation[action_id]
    )
    if abs(float(joint.sum()) - 1.0) > 1e-10:
        raise RuntimeError("V64 parameter-observation predictive does not normalize")
    return predicted_state, joint


def update_joint_belief(
    family: V64Family,
    belief: np.ndarray,
    action: int | str,
    observation: int | str,
) -> tuple[np.ndarray, float]:
    action_id = action_index(family, action)
    observation_id = observation_index(family, observation)
    predicted_state, joint = predict_joint_parameter_observation(
        family, belief, action_id
    )
    probability = float(joint[:, :, observation_id].sum())
    if probability <= 0.0:
        raise ValueError("impossible V64 action-observation update")
    weighted = (
        predicted_state
        * family.model.observation[action_id, :, observation_id][None, None, :]
    )
    posterior = weighted / probability
    _assert_belief(family, posterior)
    return posterior, probability


def filter_public_history(
    family: V64Family,
    initial_observation: int | str,
    actions: Sequence[int | str],
    observations: Sequence[int | str],
) -> tuple[np.ndarray, float]:
    if len(actions) != len(observations):
        raise ValueError("V64 history must have one post-action observation per action")
    belief, initial_probability = initial_joint_belief(family, initial_observation)
    log_evidence = math.log(initial_probability)
    for action, observation in zip(actions, observations, strict=True):
        belief, probability = update_joint_belief(
            family, belief, action, observation
        )
        log_evidence += math.log(probability)
    return belief, log_evidence


def entropy(probabilities: Iterable[float]) -> float:
    return -sum(
        float(probability) * math.log(float(probability))
        for probability in probabilities
        if probability > 0.0
    )


def score_action(
    family: V64Family, belief: np.ndarray, action: int | str
) -> dict:
    action_id = action_index(family, action)
    _, joint = predict_joint_parameter_observation(family, belief, action_id)
    target_prior = belief.sum(axis=2)
    predictive = joint.sum(axis=(0, 1))
    information = 0.0
    expected_posterior_entropy = 0.0
    for observation in range(len(family.model.observations)):
        outcome_probability = float(predictive[observation])
        if outcome_probability <= 0.0:
            continue
        posterior = joint[:, :, observation] / outcome_probability
        mask = posterior > 0.0
        if np.any(mask & (target_prior <= 0.0)):
            raise RuntimeError("V64 posterior mass lies outside the static prior support")
        information += outcome_probability * float(
            np.sum(posterior[mask] * np.log(posterior[mask] / target_prior[mask]))
        )
        expected_posterior_entropy += outcome_probability * entropy(posterior.flat)
    prior_entropy = entropy(target_prior.flat)
    entropy_information = prior_entropy - expected_posterior_entropy
    return {
        "action": family.model.actions[action_id],
        "action_index": action_id,
        "eig": information,
        "entropy_eig": entropy_information,
        "prior_entropy": prior_entropy,
        "expected_posterior_entropy": expected_posterior_entropy,
        "predictive": predictive.tolist(),
        "predictive_entropy": entropy(predictive),
        "normalizes": bool(abs(float(predictive.sum()) - 1.0) <= 1e-12),
        "finite": bool(
            all(
                math.isfinite(value)
                for value in (
                    information,
                    entropy_information,
                    prior_entropy,
                    expected_posterior_entropy,
                )
            )
        ),
    }


def score_all_actions(family: V64Family, belief: np.ndarray) -> list[dict]:
    return [score_action(family, belief, action) for action in family.canonical_actions]


def select_action(
    family: V64Family,
    belief: np.ndarray,
    *,
    tolerance: float = 1e-12,
    field: str = "eig",
) -> dict:
    scores = score_all_actions(family, belief)
    maximum = max(float(row[field]) for row in scores)
    optimal = [row for row in scores if float(row[field]) >= maximum - tolerance]
    if not optimal:
        raise RuntimeError("V64 selection has no optimal command")
    return {
        "selected": optimal[0],
        "maximum": maximum,
        "optimal_actions": [row["action"] for row in optimal],
        "scores": scores,
    }


def select_from_named_values(
    family: V64Family,
    values: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> dict:
    if len(values) != len(family.canonical_actions):
        raise ValueError("V64 control score vector omits a command")
    maximum = max(float(value) for value in values)
    optimal_positions = [
        position
        for position, value in enumerate(values)
        if float(value) >= maximum - tolerance
    ]
    selected_position = optimal_positions[0]
    selected_action = family.canonical_actions[selected_position]
    return {
        "selected_action": family.model.actions[selected_action],
        "selected_action_index": selected_action,
        "maximum": maximum,
        "optimal_actions": [
            family.model.actions[family.canonical_actions[position]]
            for position in optimal_positions
        ],
        "values": [float(value) for value in values],
    }


def current_state_information(
    family: V64Family, belief: np.ndarray, action: int | str
) -> float:
    """Information in the next observation about the pre-action maze state."""
    _assert_belief(family, belief)
    action_id = action_index(family, action)
    state_prior = belief.sum(axis=(0, 1))
    joint = np.zeros((len(family.model.states), len(family.model.observations)))
    for source in range(len(family.model.states)):
        for identity in range(2):
            for node in range(len(family.theta)):
                atom_mass = float(belief[identity, node, source])
                if atom_mass <= 0.0:
                    continue
                successor_observation = (
                    family.transitions[identity, node, action_id]
                    @ family.model.observation[action_id]
                )
                joint[source] += atom_mass * successor_observation[source]
    predictive = joint.sum(axis=0)
    information = 0.0
    for source in range(len(family.model.states)):
        for observation in range(len(family.model.observations)):
            mass = float(joint[source, observation])
            denominator = float(state_prior[source] * predictive[observation])
            if mass > 0.0:
                information += mass * math.log(mass / denominator)
    return information


def map_identity_belief(belief: np.ndarray) -> np.ndarray:
    masses = belief.sum(axis=(1, 2))
    selected = int(np.argmax(masses))
    collapsed = np.zeros_like(belief)
    collapsed[selected] = belief[selected] / masses[selected]
    return collapsed


def theta_mean_point_family(
    family: V64Family, belief: np.ndarray
) -> tuple[V64Family, np.ndarray]:
    posterior = static_posterior(belief)
    mean = float(np.sum(posterior * family.theta[None, :]))
    collapsed_belief = belief.sum(axis=1)[:, None, :]
    identity_mass = collapsed_belief.sum(axis=2)[:, 0]
    transitions = np.asarray(
        [
            [
                [
                    mean * family.model.transition[action]
                    + (1.0 - mean)
                    * family.model.transition[
                        family.permutations[identity, action]
                    ]
                    for action in range(len(family.model.actions))
                ]
            ]
            for identity in range(2)
        ]
    )
    point_family = V64Family(
        model=family.model,
        theta=np.asarray([mean]),
        theta_weights=np.asarray([1.0]),
        static_prior=identity_mass[:, None],
        transitions=transitions,
        permutations=family.permutations,
        canonical_actions=family.canonical_actions,
        identity_names=family.identity_names,
        theta_support=family.theta_support,
    )
    _assert_belief(point_family, collapsed_belief)
    return point_family, collapsed_belief


def wrong_permutation_family(family: V64Family) -> V64Family:
    permutations = family.permutations.copy()
    east = family.model.actions.index("e")
    permutations[0, east], permutations[1, east] = (
        permutations[1, east],
        permutations[0, east],
    )
    transitions = np.asarray(
        [
            [
                [
                    theta * family.model.transition[action]
                    + (1.0 - theta)
                    * family.model.transition[permutations[identity, action]]
                    for action in range(len(family.model.actions))
                ]
                for theta in family.theta
            ]
            for identity in range(2)
        ]
    )
    return V64Family(
        model=family.model,
        theta=family.theta.copy(),
        theta_weights=family.theta_weights.copy(),
        static_prior=family.static_prior.copy(),
        transitions=transitions,
        permutations=permutations,
        canonical_actions=family.canonical_actions,
        identity_names=family.identity_names,
        theta_support=family.theta_support,
    )


def score_control_policies(family: V64Family, belief: np.ndarray) -> dict:
    primary_scores = score_all_actions(family, belief)
    primary_values = [row["eig"] for row in primary_scores]
    predictive_values = [row["predictive_entropy"] for row in primary_scores]
    state_values = [
        current_state_information(family, belief, action)
        for action in family.canonical_actions
    ]
    map_belief = map_identity_belief(belief)
    map_values = [row["eig"] for row in score_all_actions(family, map_belief)]
    point_family, point_belief = theta_mean_point_family(family, belief)
    point_values = [
        row["eig"] for row in score_all_actions(point_family, point_belief)
    ]
    wrong_family = wrong_permutation_family(family)
    wrong_values = [
        row["eig"] for row in score_all_actions(wrong_family, belief)
    ]
    return {
        "primary": select_from_named_values(family, primary_values),
        "uniform_random_mean_eig": float(np.mean(primary_values)),
        "predictive_entropy": select_from_named_values(family, predictive_values),
        "state_only_information": select_from_named_values(family, state_values),
        "map_identity": select_from_named_values(family, map_values),
        "theta_mean": select_from_named_values(family, point_values),
        "wrong_permutation": select_from_named_values(family, wrong_values),
    }


def static_posterior(belief: np.ndarray) -> np.ndarray:
    posterior = np.asarray(belief, dtype=np.float64).sum(axis=2)
    posterior /= posterior.sum()
    return posterior


def identity_posterior(belief: np.ndarray) -> np.ndarray:
    posterior = static_posterior(belief).sum(axis=1)
    posterior /= posterior.sum()
    return posterior


def theta_posterior(belief: np.ndarray) -> np.ndarray:
    posterior = static_posterior(belief).sum(axis=0)
    posterior /= posterior.sum()
    return posterior


def posterior_kl_to_static_prior(family: V64Family, belief: np.ndarray) -> float:
    posterior = static_posterior(belief)
    mask = posterior > 0.0
    if np.any(mask & (family.static_prior <= 0.0)):
        raise RuntimeError("V64 posterior lies outside the frozen static prior")
    return float(
        np.sum(
            posterior[mask]
            * np.log(posterior[mask] / family.static_prior[mask])
        )
    )


def true_transition(
    family: V64Family, identity: int, theta: float, action: int | str
) -> np.ndarray:
    if identity not in (0, 1):
        raise ValueError("V64 identity must be zero or one")
    low, high = family.theta_support
    if not low <= theta <= high:
        raise ValueError("V64 theta lies outside its frozen support")
    action_id = action_index(family, action)
    return (
        theta * family.model.transition[action_id]
        + (1.0 - theta)
        * family.model.transition[family.permutations[identity, action_id]]
    )


def sample_categorical(probabilities: np.ndarray, uniform: float) -> int:
    if not 0.0 <= uniform < 1.0:
        raise ValueError("V64 categorical uniform must lie in [0, 1)")
    values = np.asarray(probabilities, dtype=np.float64)
    if abs(float(values.sum()) - 1.0) > 1e-10 or np.min(values) < 0.0:
        raise ValueError("V64 categorical probabilities are invalid")
    return min(
        len(values) - 1,
        int(np.searchsorted(np.cumsum(values), uniform, side="right")),
    )


def simulate_step(
    family: V64Family,
    identity: int,
    theta: float,
    state: int,
    action: int | str,
    transition_uniform: float,
    observation_uniform: float,
) -> tuple[int, int, float]:
    action_id = action_index(family, action)
    if state not in range(len(family.model.states)):
        raise ValueError("invalid V64 state")
    successor = sample_categorical(
        true_transition(family, identity, theta, action_id)[state],
        transition_uniform,
    )
    observation = sample_categorical(
        family.model.observation[action_id, successor], observation_uniform
    )
    reward = float(family.model.reward[action_id, state, successor])
    return successor, observation, reward


_FORBIDDEN_PUBLIC_KEYS = {
    "identity",
    "theta",
    "state",
    "states",
    "initial_state",
    "transition_uniforms",
    "observation_uniforms",
    "audit",
    "truth",
}


def assert_public_selection_payload(payload: dict) -> None:
    forbidden = _FORBIDDEN_PUBLIC_KEYS.intersection(payload)
    if forbidden:
        raise PermissionError(
            f"V64 selection payload exposes audit-only fields: {sorted(forbidden)}"
        )
    allowed = {"record_id", "prefix_length", "initial_observation", "actions", "observations"}
    unknown = set(payload) - allowed
    if unknown:
        raise PermissionError(f"V64 selection payload has undeclared fields: {sorted(unknown)}")


def attempted_outcome_leak_selection(payload: dict, realized_outcome: object) -> None:
    assert_public_selection_payload(payload)
    if realized_outcome is not None:
        raise PermissionError("V64 action selection cannot read the next realized outcome")
