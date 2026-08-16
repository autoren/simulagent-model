#!/usr/bin/env python3
"""Structurally separate scalar filtering and EIG reference for V64."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import parse_pomdp_file


def load_reference(
    design_lock: str | Path = "configs/v64-design-lock.json",
    *,
    quadrature_nodes: int | None = None,
) -> dict:
    path = Path(design_lock)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = json.loads(path.read_text())["config_payload"]
    model = parse_pomdp_file(
        PROJECT_ROOT / config["externalAnchor"]["workspaceModelPath"]
    )
    low, high = map(float, config["unknownDynamicsFamily"]["thetaSupport"])
    nodes = quadrature_nodes or int(config["exactOracle"]["quadratureNodes"])
    raw_x, raw_w = np.polynomial.legendre.leggauss(nodes)
    theta = low + (raw_x + 1.0) * (high - low) / 2.0
    prior_weight = []
    for value, weight in zip(theta, raw_w, strict=True):
        unit = (float(value) - low) / (high - low)
        density = 6.0 * unit * (1.0 - unit) / (high - low)
        prior_weight.append(float(weight) * (high - low) / 2.0 * density)
    normalizer = sum(prior_weight)
    prior_weight = np.asarray([weight / normalizer for weight in prior_weight])
    index = {name: i for i, name in enumerate(model.actions)}
    permutations = (
        {index["n"]: index["e"], index["e"]: index["s"], index["s"]: index["w"], index["w"]: index["n"]},
        {index["n"]: index["w"], index["w"]: index["s"], index["s"]: index["e"], index["e"]: index["n"]},
    )
    return {
        "model": model,
        "theta": np.asarray(theta, dtype=np.float64),
        "theta_weights": prior_weight,
        "permutations": permutations,
        "canonical_actions": tuple(index[name] for name in ("n", "e", "s", "w")),
    }


def _action(reference: dict, action: int | str) -> int:
    model = reference["model"]
    if isinstance(action, str):
        return model.actions.index(action)
    return int(action)


def _observation(reference: dict, observation: int | str) -> int:
    model = reference["model"]
    if isinstance(observation, str):
        return model.observations.index(observation)
    return int(observation)


def _transition_probability(
    reference: dict,
    identity: int,
    node: int,
    action: int,
    source: int,
    successor: int,
) -> float:
    model = reference["model"]
    theta = float(reference["theta"][node])
    fallback = reference["permutations"][identity][action]
    return float(
        theta * model.transition[action, source, successor]
        + (1.0 - theta) * model.transition[fallback, source, successor]
    )


def initial_atoms(reference: dict, initial_observation: int | str) -> dict[tuple[int, int, int], float]:
    model = reference["model"]
    observation = _observation(reference, initial_observation)
    atoms: dict[tuple[int, int, int], float] = {}
    for identity in range(2):
        for node, theta_weight in enumerate(reference["theta_weights"]):
            for state, state_weight in enumerate(model.initial):
                mass = (
                    0.5
                    * float(theta_weight)
                    * float(state_weight)
                    * float(model.observation[0, state, observation])
                )
                if mass > 0.0:
                    atoms[(identity, node, state)] = mass
    normalizer = sum(atoms.values())
    if normalizer <= 0.0:
        raise ValueError("impossible scalar-reference initial observation")
    return {key: value / normalizer for key, value in atoms.items()}


def update_atoms(
    reference: dict,
    atoms: dict[tuple[int, int, int], float],
    action: int | str,
    observation: int | str,
) -> tuple[dict[tuple[int, int, int], float], float]:
    model = reference["model"]
    action_id = _action(reference, action)
    observation_id = _observation(reference, observation)
    updated: dict[tuple[int, int, int], float] = {}
    for (identity, node, source), prior_mass in atoms.items():
        for successor in range(len(model.states)):
            probability = _transition_probability(
                reference, identity, node, action_id, source, successor
            )
            likelihood = float(model.observation[action_id, successor, observation_id])
            mass = prior_mass * probability * likelihood
            if mass > 0.0:
                key = (identity, node, successor)
                updated[key] = updated.get(key, 0.0) + mass
    evidence = sum(updated.values())
    if evidence <= 0.0:
        raise ValueError("impossible scalar-reference action observation")
    return {key: value / evidence for key, value in updated.items()}, evidence


def filter_history(
    reference: dict,
    initial_observation: int | str,
    actions: Sequence[int | str],
    observations: Sequence[int | str],
) -> tuple[dict[tuple[int, int, int], float], float]:
    if len(actions) != len(observations):
        raise ValueError("scalar-reference history lengths differ")
    atoms = initial_atoms(reference, initial_observation)
    # Recompute reset evidence without sharing the candidate implementation.
    model = reference["model"]
    initial_id = _observation(reference, initial_observation)
    reset_evidence = sum(
        float(model.initial[state]) * float(model.observation[0, state, initial_id])
        for state in range(len(model.states))
    )
    log_evidence = math.log(reset_evidence)
    for action, observation in zip(actions, observations, strict=True):
        atoms, evidence = update_atoms(reference, atoms, action, observation)
        log_evidence += math.log(evidence)
    return atoms, log_evidence


def score_action(
    reference: dict,
    atoms: dict[tuple[int, int, int], float],
    action: int | str,
) -> dict:
    model = reference["model"]
    action_id = _action(reference, action)
    target_prior: dict[tuple[int, int], float] = {}
    for (identity, node, _), mass in atoms.items():
        key = (identity, node)
        target_prior[key] = target_prior.get(key, 0.0) + mass
    joint: dict[tuple[int, int, int], float] = {}
    for (identity, node, source), atom_mass in atoms.items():
        for successor in range(len(model.states)):
            transition = _transition_probability(
                reference, identity, node, action_id, source, successor
            )
            if transition <= 0.0:
                continue
            for observation in range(len(model.observations)):
                likelihood = float(model.observation[action_id, successor, observation])
                mass = atom_mass * transition * likelihood
                if mass > 0.0:
                    key = (identity, node, observation)
                    joint[key] = joint.get(key, 0.0) + mass
    predictive = [0.0] * len(model.observations)
    for (_, _, observation), mass in joint.items():
        predictive[observation] += mass
    information = 0.0
    for (identity, node, observation), mass in joint.items():
        denominator = target_prior[(identity, node)] * predictive[observation]
        information += mass * math.log(mass / denominator)
    if abs(sum(predictive) - 1.0) > 1e-12:
        raise RuntimeError("scalar-reference predictive does not normalize")
    return {
        "action": model.actions[action_id],
        "action_index": action_id,
        "eig": information,
        "predictive": predictive,
    }


def score_all_actions(reference: dict, atoms: dict[tuple[int, int, int], float]) -> list[dict]:
    return [score_action(reference, atoms, action) for action in reference["canonical_actions"]]


def select_action(
    reference: dict,
    atoms: dict[tuple[int, int, int], float],
    tolerance: float = 1e-12,
) -> dict:
    scores = score_all_actions(reference, atoms)
    maximum = max(row["eig"] for row in scores)
    optimal = [row for row in scores if row["eig"] >= maximum - tolerance]
    return {
        "selected": optimal[0],
        "maximum": maximum,
        "optimal_actions": [row["action"] for row in optimal],
        "scores": scores,
    }


def atoms_to_dense(reference: dict, atoms: dict[tuple[int, int, int], float]) -> np.ndarray:
    dense = np.zeros(
        (2, len(reference["theta"]), len(reference["model"].states)), dtype=np.float64
    )
    for key, mass in atoms.items():
        dense[key] = mass
    return dense
