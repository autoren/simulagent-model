"""Exact finite proposal and posterior utilities for V23."""

from __future__ import annotations

import heapq
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def stable_log(value: float) -> float:
    return math.log(max(float(value), 1e-12))


def solve_assignment(
    scores: np.ndarray, fixed: dict[int, int], forbidden: set[tuple[int, int]],
) -> tuple[float, tuple[int, ...]] | None:
    size = scores.shape[0]
    if scores.shape != (size, size) or len(set(fixed.values())) != len(fixed):
        return None
    if any((row, column) in forbidden for row, column in fixed.items()):
        return None
    remaining_rows = [row for row in range(size) if row not in fixed]
    used = set(fixed.values())
    remaining_columns = [column for column in range(size) if column not in used]
    if len(remaining_rows) != len(remaining_columns):
        return None
    assignment = dict(fixed)
    if remaining_rows:
        sub = scores[np.ix_(remaining_rows, remaining_columns)].copy()
        for row_index, row in enumerate(remaining_rows):
            for column_index, column in enumerate(remaining_columns):
                if (row, column) in forbidden:
                    sub[row_index, column_index] = -np.inf
        if any(not np.isfinite(sub[index]).any() for index in range(len(remaining_rows))):
            return None
        finite = sub[np.isfinite(sub)]
        penalty = float(np.min(finite) - max(1.0, np.ptp(finite) + 1.0) * (size + 1))
        cost = -np.where(np.isfinite(sub), sub, penalty)
        row_indices, column_indices = linear_sum_assignment(cost)
        if any(not np.isfinite(sub[row, column]) for row, column in zip(row_indices, column_indices, strict=True)):
            return None
        for row, column in zip(row_indices, column_indices, strict=True):
            assignment[remaining_rows[row]] = remaining_columns[column]
    values = tuple(assignment[row] for row in range(size))
    return float(sum(scores[row, column] for row, column in enumerate(values))), values


def k_best_assignments(scores: np.ndarray, count: int) -> list[tuple[float, tuple[int, ...]]]:
    """Exact deterministic Murty enumeration for a square maximum-score assignment."""

    if count <= 0:
        return []
    root = solve_assignment(scores, {}, set())
    if root is None:
        return []
    heap: list[tuple[float, tuple[int, ...], int, dict[int, int], set[tuple[int, int]]]] = []
    serial = 0

    def push(fixed: dict[int, int], forbidden: set[tuple[int, int]]) -> None:
        nonlocal serial
        solution = solve_assignment(scores, fixed, forbidden)
        if solution is None:
            return
        score, values = solution
        heapq.heappush(heap, (-score, values, serial, fixed, forbidden))
        serial += 1

    push({}, set())
    results = []
    seen_solutions = set()
    seen_subproblems = set()
    while heap and len(results) < count:
        negative, values, _, fixed, forbidden = heapq.heappop(heap)
        if values not in seen_solutions:
            seen_solutions.add(values)
            results.append((-negative, values))
        free_rows = [row for row in range(len(values)) if row not in fixed]
        child_fixed = dict(fixed)
        for row in free_rows:
            child_forbidden = set(forbidden)
            child_forbidden.add((row, values[row]))
            key = (tuple(sorted(child_fixed.items())), tuple(sorted(child_forbidden)))
            if key not in seen_subproblems:
                seen_subproblems.add(key)
                push(dict(child_fixed), child_forbidden)
            child_fixed[row] = values[row]
    return results


def k_best_independent(labels: Sequence[Sequence[tuple[str, float]]], count: int) -> list[tuple[float, tuple[str, ...]]]:
    """Exact best-first enumeration of independent label vectors."""

    if count <= 0 or not labels:
        return []
    ordered = [sorted(values, key=lambda row: (-row[1], row[0])) for values in labels]
    if any(not values for values in ordered):
        return []

    def value(indices: tuple[int, ...]) -> tuple[float, tuple[str, ...]]:
        selected = [ordered[index][choice] for index, choice in enumerate(indices)]
        return sum(stable_log(row[1]) for row in selected), tuple(row[0] for row in selected)

    initial = tuple(0 for _ in ordered)
    initial_score, initial_labels = value(initial)
    heap = [(-initial_score, initial_labels, initial)]
    seen = {initial}
    result = []
    while heap and len(result) < count:
        negative, label_vector, indices = heapq.heappop(heap)
        result.append((-negative, label_vector))
        for dimension in range(len(indices)):
            if indices[dimension] + 1 >= len(ordered[dimension]):
                continue
            neighbor = list(indices)
            neighbor[dimension] += 1
            key = tuple(neighbor)
            if key in seen:
                continue
            seen.add(key)
            score, labels_value = value(key)
            heapq.heappush(heap, (-score, labels_value, key))
    return result


def normalized_top_graphs(
    assignments: Sequence[tuple[float, tuple[int, ...]]],
    truth_vectors: Sequence[tuple[float, tuple[str, ...]]],
    budget: int, maximum_unknown: int,
) -> list[dict[str, Any]]:
    combined = []
    seen = set()
    for assignment_score, assignment in assignments:
        for truth_score, truth in truth_vectors:
            if sum(value == "unknown" for value in truth) > maximum_unknown:
                continue
            key = (assignment, truth)
            if key in seen:
                continue
            seen.add(key)
            combined.append((assignment_score + truth_score, assignment, truth))
    combined.sort(key=lambda row: (-row[0], row[1], row[2]))
    combined = combined[:budget]
    if not combined:
        return []
    maximum = combined[0][0]
    weights = np.asarray([math.exp(row[0] - maximum) for row in combined], dtype=np.float64)
    weights /= weights.sum()
    return [
        {"log_score": score, "assignment": assignment, "truth": truth, "probability": float(weight)}
        for (score, assignment, truth), weight in zip(combined, weights, strict=True)
    ]


def credible_indices(posterior: np.ndarray, keys: Sequence[str], mass: float) -> list[int]:
    if posterior.sum() <= 0:
        return []
    ordered = sorted(range(len(posterior)), key=lambda index: (-float(posterior[index]), keys[index]))
    selected = []
    cumulative = 0.0
    for index in ordered:
        if posterior[index] <= 0:
            continue
        selected.append(index)
        cumulative += float(posterior[index])
        if cumulative + 1e-15 >= mass:
            break
    return selected
