"""Deterministic allowed-values transition evaluator used by V9 scoring."""

from __future__ import annotations

from itertools import product
from typing import Any


VALUE_ORDER = ("inactive", "active")


def evaluate_allowed_transitions(
    schema: dict[str, Any],
    determinant_values: list[dict[str, Any]],
) -> dict[str, Any]:
    determinant_ids = [value["id"] for value in schema["transition_determinants"]]
    if len(set(determinant_ids)) != len(determinant_ids):
        raise ValueError("Transition schema contains duplicate determinant ids")
    values_by_id: dict[str, tuple[str, ...]] = {}
    for grounding in determinant_values:
        determinant_id = grounding["determinant_id"]
        if determinant_id not in determinant_ids:
            raise ValueError(f"Grounding contains unknown determinant {determinant_id}")
        if determinant_id in values_by_id:
            raise ValueError(f"Grounding repeats determinant {determinant_id}")
        supplied = grounding["allowed_values"]
        if any(value not in VALUE_ORDER for value in supplied):
            raise ValueError("Allowed values must be active or inactive")
        normalized = tuple(value for value in VALUE_ORDER if value in supplied)
        if not normalized:
            raise ValueError(f"Grounding has no allowed values for {determinant_id}")
        values_by_id[determinant_id] = normalized
    for determinant_id in determinant_ids:
        if determinant_id not in values_by_id:
            raise ValueError(f"Grounding omits determinant {determinant_id}")
    cases: dict[tuple[str, ...], str] = {}
    for transition_case in schema["transition_cases"]:
        values = tuple(transition_case["values"])
        if len(values) != len(determinant_ids):
            raise ValueError("Transition case arity differs from determinant schema")
        if values in cases:
            raise ValueError(f"Transition schema repeats assignment {values}")
        cases[values] = transition_case["transition_code"]
    assignments = list(product(*(values_by_id[value] for value in determinant_ids)))
    codes = []
    for assignment in assignments:
        if assignment not in cases:
            raise ValueError(f"Transition schema omits compatible assignment {assignment}")
        codes.append(cases[assignment])
    possible = sorted(set(codes))
    return {
        "compatible_assignments": len(assignments),
        "possible_transition_codes": possible,
        "identifiable": len(possible) == 1,
    }
