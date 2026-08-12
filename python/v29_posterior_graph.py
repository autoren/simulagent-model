"""Exact posterior-marginal support graph decoding for V29."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from v28_marginal_map import logsumexp


def posterior_marginal_decode(
    hypotheses: Sequence[Any], scene_graphs: Sequence[Sequence[dict[str, Any]]],
    compatibility: Sequence[np.ndarray],
) -> dict[str, Any] | None:
    if not hypotheses or not scene_graphs:
        return None
    graph_log_probabilities = []
    trace_log_likelihoods = []
    for graphs, matrix in zip(scene_graphs, compatibility, strict=True):
        scores = np.asarray([row["log_score"] for row in graphs], dtype=np.float64)
        log_probabilities = scores - logsumexp(scores)
        graph_log_probabilities.append(log_probabilities)
        likelihoods = np.full(len(hypotheses), -math.inf, dtype=np.float64)
        for program_index in range(len(hypotheses)):
            indices = np.flatnonzero(matrix[:, program_index])
            if len(indices):
                likelihoods[program_index] = logsumexp(log_probabilities[indices])
        trace_log_likelihoods.append(likelihoods)
    program_log_weights = np.sum(np.stack(trace_log_likelihoods), axis=0)
    finite = np.isfinite(program_log_weights)
    if not finite.any():
        return None
    normalizer = logsumexp(program_log_weights[finite])
    program_posterior = np.zeros(len(hypotheses), dtype=np.float64)
    program_posterior[finite] = np.exp(program_log_weights[finite] - normalizer)
    graph_indices = []
    graph_posteriors = []
    for trace_index, (graphs, matrix) in enumerate(zip(scene_graphs, compatibility, strict=True)):
        other_rows = [
            values for index, values in enumerate(trace_log_likelihoods)
            if index != trace_index
        ]
        other_log_weight = (
            np.sum(np.stack(other_rows), axis=0)
            if other_rows else np.zeros(len(hypotheses), dtype=np.float64)
        )
        graph_log_weights = np.full(len(graphs), -math.inf, dtype=np.float64)
        for graph_index in range(len(graphs)):
            compatible_programs = np.flatnonzero(matrix[graph_index] & np.isfinite(other_log_weight))
            if len(compatible_programs):
                graph_log_weights[graph_index] = (
                    graph_log_probabilities[trace_index][graph_index]
                    + logsumexp(other_log_weight[compatible_programs])
                )
        graph_finite = np.isfinite(graph_log_weights)
        if not graph_finite.any():
            return None
        graph_normalizer = logsumexp(graph_log_weights[graph_finite])
        posterior = np.zeros(len(graphs), dtype=np.float64)
        posterior[graph_finite] = np.exp(graph_log_weights[graph_finite] - graph_normalizer)
        selected = min(
            np.flatnonzero(graph_finite).tolist(),
            key=lambda index: (-float(posterior[index]), graphs[index]["graph_key"], index),
        )
        graph_indices.append(selected)
        graph_posteriors.append(float(posterior[selected]))
    ordered_programs = sorted(
        np.flatnonzero(finite).tolist(),
        key=lambda index: (-float(program_posterior[index]), hypotheses[index].key),
    )
    return {
        "program_posterior": program_posterior,
        "program_index": ordered_programs[0],
        "program_key": hypotheses[ordered_programs[0]].key,
        "graph_indices": tuple(graph_indices),
        "graph_posteriors": tuple(graph_posteriors),
        "finite_programs": int(finite.sum()),
    }
