"""Marginal program MAP utilities for V28."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from v22_relational import execute_partial, rows_to_epistemic


def logsumexp(values: Sequence[float]) -> float:
    if not len(values):
        return -math.inf
    maximum = max(float(value) for value in values)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(float(value) - maximum) for value in values))


def compatibility_matrix_deduplicated(
    graphs: Sequence[dict[str, Any]], hypotheses: Sequence[Any], support: dict[str, Any],
    observed: str, v22_config: dict[str, Any], maximum_unknown: int,
) -> tuple[np.ndarray, int]:
    by_key: dict[str, np.ndarray] = {}
    rows = []
    for graph in graphs:
        key = graph["graph_key"]
        if key not in by_key:
            state = rows_to_epistemic(graph["epistemic_state"])
            compatible = np.zeros(len(hypotheses), dtype=bool)
            for index, hypothesis in enumerate(hypotheses):
                possible = execute_partial(
                    [hypothesis.program], v22_config, support["entities"], state,
                    support["action_binding"], maximum_unknown,
                )["possible_transition_codes"]
                compatible[index] = observed in possible
            by_key[key] = compatible
        rows.append(by_key[key])
    return np.stack(rows), len(by_key)


def select_marginal_episode_map(
    hypotheses: Sequence[Any], scene_graphs: Sequence[Sequence[dict[str, Any]]],
    compatibility: Sequence[np.ndarray],
) -> dict[str, Any] | None:
    if not hypotheses or not scene_graphs:
        return None
    log_weights = np.zeros(len(hypotheses), dtype=np.float64)
    for graphs, matrix in zip(scene_graphs, compatibility, strict=True):
        scores = np.asarray([row["log_score"] for row in graphs], dtype=np.float64)
        denominator = logsumexp(scores)
        for program_index in range(len(hypotheses)):
            indices = np.flatnonzero(matrix[:, program_index])
            if not len(indices) or not math.isfinite(log_weights[program_index]):
                log_weights[program_index] = -math.inf
            else:
                log_weights[program_index] += logsumexp(scores[indices]) - denominator
    finite = np.isfinite(log_weights)
    if not finite.any():
        return None
    normalizer = logsumexp(log_weights[finite])
    posterior = np.zeros(len(hypotheses), dtype=np.float64)
    posterior[finite] = np.exp(log_weights[finite] - normalizer)
    ordered = sorted(
        np.flatnonzero(finite).tolist(),
        key=lambda index: (-float(log_weights[index]), hypotheses[index].key),
    )
    selected_program = ordered[0]
    graph_indices = []
    for matrix in compatibility:
        indices = np.flatnonzero(matrix[:, selected_program])
        if not len(indices):
            raise RuntimeError("Selected marginal program lacks a compatible support graph")
        graph_indices.append(int(indices[0]))
    return {
        "program_index": selected_program,
        "program_key": hypotheses[selected_program].key,
        "graph_indices": tuple(graph_indices),
        "posterior": posterior,
        "finite_programs": int(finite.sum()),
        "maximum_posterior": float(posterior[selected_program]),
    }
