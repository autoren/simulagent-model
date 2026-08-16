#!/usr/bin/env python3
"""Independent scalar references for V65 particle-measure summaries and EIG."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from v64_external_eig import V64Family, action_index


def _group_atoms(atoms: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], dict[str, Any]] = {}
    for source in atoms:
        identity = int(source["identity"])
        theta = float(source["theta"])
        weight = float(source["weight"])
        if weight <= 0.0:
            continue
        state = [float(value) for value in np.asarray(source["state"]).tolist()]
        key = (identity, theta.hex())
        if key not in groups:
            groups[key] = {
                "identity": identity,
                "theta": theta,
                "weight": 0.0,
                "state_mass": [0.0 for _ in state],
            }
        groups[key]["weight"] += weight
        for index, probability in enumerate(state):
            groups[key]["state_mass"][index] += weight * probability
    total = sum(float(group["weight"]) for group in groups.values())
    if total <= 0.0:
        raise ValueError("scalar V65 reference received no positive atom mass")
    result = []
    for group in groups.values():
        unnormalized_weight = float(group["weight"])
        state = [value / unnormalized_weight for value in group["state_mass"]]
        state_total = sum(state)
        result.append(
            {
                "identity": int(group["identity"]),
                "theta": float(group["theta"]),
                "weight": unnormalized_weight / total,
                "state": [value / state_total for value in state],
            }
        )
    return sorted(result, key=lambda row: (row["identity"], row["theta"]))


def scalar_transition(
    family: V64Family, identity: int, theta: float, action: int
) -> list[list[float]]:
    source = family.model.transition[action]
    fallback = family.model.transition[int(family.permutations[identity, action])]
    result = []
    for state in range(len(family.model.states)):
        row = []
        for successor in range(len(family.model.states)):
            row.append(
                theta * float(source[state, successor])
                + (1.0 - theta) * float(fallback[state, successor])
            )
        result.append(row)
    return result


def scalar_atom_predictive(
    family: V64Family, atom: dict[str, Any], action: int
) -> list[float]:
    transition = scalar_transition(
        family, int(atom["identity"]), float(atom["theta"]), action
    )
    result = [0.0 for _ in family.model.observations]
    for state, state_mass in enumerate(atom["state"]):
        for successor, transition_mass in enumerate(transition[state]):
            for observation in range(len(family.model.observations)):
                result[observation] += (
                    float(state_mass)
                    * transition_mass
                    * float(family.model.observation[action, successor, observation])
                )
    total = sum(result)
    return [value / total for value in result]


def scalar_score_action(
    family: V64Family,
    measure: dict[str, Any] | Sequence[dict[str, Any]],
    action: int | str,
) -> dict[str, Any]:
    action_id = action_index(family, action)
    source = measure["atoms"] if isinstance(measure, dict) else measure
    atoms = _group_atoms(source)
    conditionals = [scalar_atom_predictive(family, atom, action_id) for atom in atoms]
    predictive = [0.0 for _ in family.model.observations]
    for atom, conditional in zip(atoms, conditionals, strict=True):
        for observation, probability in enumerate(conditional):
            predictive[observation] += float(atom["weight"]) * probability
    information = 0.0
    for atom, conditional in zip(atoms, conditionals, strict=True):
        for observation, probability in enumerate(conditional):
            if probability > 0.0:
                information += (
                    float(atom["weight"])
                    * probability
                    * math.log(probability / predictive[observation])
                )
    return {
        "action": family.model.actions[action_id],
        "action_index": action_id,
        "eig": information,
        "predictive": predictive,
        "normalizes": abs(sum(predictive) - 1.0) <= 1e-12,
        "finite": math.isfinite(information)
        and all(math.isfinite(value) for value in predictive),
        "static_atom_count": len(atoms),
    }


def scalar_score_all_actions(
    family: V64Family, measure: dict[str, Any] | Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        scalar_score_action(family, measure, action)
        for action in family.canonical_actions
    ]


def scalar_select_action(
    family: V64Family,
    measure: dict[str, Any] | Sequence[dict[str, Any]],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    scores = scalar_score_all_actions(family, measure)
    maximum = max(float(row["eig"]) for row in scores)
    optimal = [row for row in scores if float(row["eig"]) >= maximum - tolerance]
    return {
        "selected": optimal[0],
        "maximum": maximum,
        "optimal_actions": [row["action"] for row in optimal],
        "scores": scores,
    }


def scalar_summary(
    family: V64Family,
    measure: dict[str, Any] | Sequence[dict[str, Any]],
    bins: int = 16,
) -> dict[str, Any]:
    source = measure["atoms"] if isinstance(measure, dict) else measure
    atoms = _group_atoms(source)
    identity = [0.0, 0.0]
    state = [0.0 for _ in family.model.states]
    joint = [[0.0 for _ in range(bins)] for _ in range(2)]
    theta_values = []
    theta_weights = []
    low, high = family.theta_support
    for atom in atoms:
        identity_id = int(atom["identity"])
        weight = float(atom["weight"])
        theta = float(atom["theta"])
        identity[identity_id] += weight
        theta_values.append(theta)
        theta_weights.append(weight)
        for state_id, probability in enumerate(atom["state"]):
            state[state_id] += weight * float(probability)
        index = min(bins - 1, max(0, int((theta - low) / (high - low) * bins)))
        joint[identity_id][index] += weight
    return {
        "identity": identity,
        "theta_values": theta_values,
        "theta_weights": theta_weights,
        "joint_bins": joint,
        "state": state,
        "normalizes": (
            abs(sum(identity) - 1.0) <= 1e-12
            and abs(sum(state) - 1.0) <= 1e-12
            and abs(sum(sum(row) for row in joint) - 1.0) <= 1e-12
        ),
    }


def scalar_state_as_target(
    family: V64Family,
    measure: dict[str, Any] | Sequence[dict[str, Any]],
    action: int | str,
) -> float:
    action_id = action_index(family, action)
    source = measure["atoms"] if isinstance(measure, dict) else measure
    atoms = _group_atoms(source)
    rows: list[tuple[float, list[float]]] = []
    for atom in atoms:
        transition = scalar_transition(
            family, int(atom["identity"]), float(atom["theta"]), action_id
        )
        for state, state_probability in enumerate(atom["state"]):
            mass = float(atom["weight"]) * float(state_probability)
            if mass <= 0.0:
                continue
            conditional = [0.0 for _ in family.model.observations]
            for successor, transition_mass in enumerate(transition[state]):
                for observation in range(len(family.model.observations)):
                    conditional[observation] += transition_mass * float(
                        family.model.observation[action_id, successor, observation]
                    )
            rows.append((mass, conditional))
    predictive = [0.0 for _ in family.model.observations]
    for mass, conditional in rows:
        for observation, probability in enumerate(conditional):
            predictive[observation] += mass * probability
    information = 0.0
    for mass, conditional in rows:
        for observation, probability in enumerate(conditional):
            if probability > 0.0:
                information += mass * probability * math.log(
                    probability / predictive[observation]
                )
    return information


def scalar_rao_blackwellize_measure(
    family: V64Family,
    measure: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Independent nested-loop conditional-state integration for V65r1."""
    repaired = []
    initial_observation = family.model.observations.index(record["initial_observation"])
    for atom in _group_atoms(measure["atoms"]):
        identity = int(atom["identity"])
        theta = float(atom["theta"])
        state = [
            float(family.model.initial[index])
            * float(family.model.observation[0, index, initial_observation])
            for index in range(len(family.model.states))
        ]
        total = sum(state)
        state = [value / total for value in state]
        for action_name, observation_name in zip(
            record["actions"], record["observations"], strict=True
        ):
            action = family.model.actions.index(action_name)
            observation = family.model.observations.index(observation_name)
            transition = scalar_transition(family, identity, theta, action)
            predicted = [0.0 for _ in family.model.states]
            for source, source_mass in enumerate(state):
                for successor, transition_mass in enumerate(transition[source]):
                    predicted[successor] += source_mass * transition_mass
            state = [
                predicted[index]
                * float(family.model.observation[action, index, observation])
                for index in range(len(family.model.states))
            ]
            total = sum(state)
            if total <= 0.0:
                raise ValueError("impossible scalar V65r1 point-parameter history")
            state = [value / total for value in state]
        repaired.append({**atom, "state": state})
    return {
        **measure,
        "atoms": _group_atoms(repaired),
        "rao_blackwellized_known_state": True,
    }
