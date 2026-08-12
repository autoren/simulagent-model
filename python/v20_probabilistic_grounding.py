"""Deterministic uncertainty propagation for the V20 development challenger."""

from __future__ import annotations

import math
from itertools import product
from typing import Any, Sequence

import numpy as np

from v18_schema import ProgramHypothesis, all_assignments, evaluate_program


BOOLEAN_VALUES = ("inactive", "active")


def sigmoid(score: float) -> float:
    if score >= 0:
        value = math.exp(-score)
        return 1.0 / (1.0 + value)
    value = math.exp(score)
    return value / (1.0 + value)


def conformal_quantile(nonconformity: Sequence[float], alpha: float) -> float:
    """Finite-sample corrected split-conformal `higher` quantile."""
    if not nonconformity:
        raise ValueError("Conformal calibration requires at least one score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    ordered = sorted(float(value) for value in nonconformity)
    if any(not 0.0 <= value <= 1.0 for value in ordered):
        raise ValueError("Nonconformity scores must lie in [0, 1]")
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    return ordered[min(rank, len(ordered)) - 1]


def polarity_probabilities(score: float) -> dict[str, float]:
    active = sigmoid(float(score))
    return {"inactive": 1.0 - active, "active": active}


def conformal_label_set(probabilities: dict[str, float], threshold: float) -> list[str]:
    if set(probabilities) != set(BOOLEAN_VALUES):
        raise ValueError("Polarity probabilities must contain inactive and active")
    labels = [value for value in BOOLEAN_VALUES if 1.0 - probabilities[value] <= threshold]
    if not labels:
        labels = [max(BOOLEAN_VALUES, key=lambda value: (probabilities[value], value == "active"))]
    return labels


def normalized_value_probabilities(
    probabilities: dict[str, float], allowed_values: Sequence[str]
) -> dict[str, float]:
    allowed = [value for value in BOOLEAN_VALUES if value in allowed_values]
    if not allowed or any(value not in BOOLEAN_VALUES for value in allowed_values):
        raise ValueError(f"Invalid Boolean value set: {allowed_values}")
    total = sum(float(probabilities[value]) for value in allowed)
    if total <= 0.0:
        return {value: 1.0 / len(allowed) for value in allowed}
    return {value: float(probabilities[value]) / total for value in allowed}


def assignment_distribution(
    determinant_ids: Sequence[str], groundings: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {value["determinant_id"]: value for value in groundings}
    if set(by_id) != set(determinant_ids):
        raise ValueError("Grounding determinant set differs from the program determinant set")
    choices = []
    for identifier in determinant_ids:
        grounding = by_id[identifier]
        allowed = grounding["allowed_values"]
        raw_probabilities = grounding.get("value_probabilities")
        if raw_probabilities is None:
            raw_probabilities = {value: 0.5 for value in BOOLEAN_VALUES}
        probabilities = normalized_value_probabilities(raw_probabilities, allowed)
        choices.append([(value, probabilities[value]) for value in probabilities])
    assignments = []
    for values in product(*choices):
        assignment = {
            identifier: label == "active"
            for identifier, (label, _) in zip(determinant_ids, values, strict=True)
        }
        probability = math.prod(value[1] for value in values)
        assignments.append({"assignment": assignment, "probability": probability})
    total = sum(value["probability"] for value in assignments)
    if total <= 0.0:
        raise ValueError("Grounding assignment distribution has zero mass")
    for value in assignments:
        value["probability"] /= total
    return assignments


def program_posterior(
    hypotheses: Sequence[ProgramHypothesis],
    determinant_ids: Sequence[str],
    traces: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Uniform-prior posterior from uncertain assignments and exact transition codes."""
    if not hypotheses:
        return np.zeros(0, dtype=np.float64)
    assignment_rows = all_assignments(determinant_ids)
    assignment_index = {
        tuple(value[identifier] for identifier in determinant_ids): index
        for index, value in enumerate(assignment_rows)
    }
    log_weights = np.zeros(len(hypotheses), dtype=np.float64)
    for trace in traces:
        masses: dict[int, float] = {}
        for value in trace["assignments"]:
            assignment = value["assignment"]
            key = tuple(bool(assignment[identifier]) for identifier in determinant_ids)
            index = assignment_index[key]
            masses[index] = masses.get(index, 0.0) + float(value["probability"])
        likelihood = np.asarray([
            sum(
                probability
                for index, probability in masses.items()
                if hypothesis.signature[index] == trace["transition_code"]
            )
            for hypothesis in hypotheses
        ], dtype=np.float64)
        positive = likelihood > 0.0
        log_weights[~positive] = -np.inf
        still_possible = positive & np.isfinite(log_weights)
        log_weights[still_possible] += np.log(likelihood[still_possible])
    finite = np.isfinite(log_weights)
    if not finite.any():
        return np.zeros(len(hypotheses), dtype=np.float64)
    maximum = float(np.max(log_weights[finite]))
    weights = np.zeros(len(hypotheses), dtype=np.float64)
    weights[finite] = np.exp(log_weights[finite] - maximum)
    weights /= weights.sum()
    return weights


def credible_hypothesis_indices(
    hypotheses: Sequence[ProgramHypothesis], posterior: np.ndarray, mass: float
) -> list[int]:
    if len(hypotheses) != len(posterior):
        raise ValueError("Hypothesis and posterior lengths differ")
    if not 0.0 < mass <= 1.0:
        raise ValueError("Credible mass must lie in (0, 1]")
    if posterior.sum() <= 0.0:
        return []
    ordered = sorted(
        range(len(hypotheses)),
        key=lambda index: (-float(posterior[index]), hypotheses[index].signature),
    )
    selected = []
    cumulative = 0.0
    for index in ordered:
        if posterior[index] <= 0.0:
            break
        selected.append(index)
        cumulative += float(posterior[index])
        if cumulative + 1e-15 >= mass:
            break
    return selected


def posterior_diagnostics(posterior: np.ndarray, selected: Sequence[int]) -> dict[str, float | int]:
    positive = posterior[posterior > 0.0]
    entropy = float(-np.sum(positive * np.log(positive))) if len(positive) else 0.0
    effective = float(1.0 / np.sum(positive ** 2)) if len(positive) else 0.0
    return {
        "nonzero_programs": int(len(positive)),
        "credible_programs": len(selected),
        "credible_mass": float(sum(float(posterior[index]) for index in selected)),
        "posterior_entropy_nats": entropy,
        "posterior_effective_programs": effective,
    }


def posterior_answer(
    hypotheses: Sequence[ProgramHypothesis],
    selected: Sequence[int],
    query_assignments: Sequence[dict[str, Any]],
    outcome_bits: int,
) -> dict[str, Any]:
    if not selected:
        values = [f"transition_{index:0{outcome_bits}b}" for index in range(2 ** outcome_bits)]
        return {"possible_transition_codes": values, "identifiable": False}
    possible = sorted({
        evaluate_program(hypotheses[index].program, value["assignment"])
        for index in selected
        for value in query_assignments
    })
    return {"possible_transition_codes": possible, "identifiable": len(possible) == 1}
