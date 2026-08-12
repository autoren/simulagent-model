"""Sparse graph enumeration and outcome-constrained episode MAP for V27."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from evaluate_v22r2_relational_grounding import pair_features
from v22_relational import canonical_json, execute_partial, rows_to_epistemic
from v23_probabilistic_relational import k_best_assignments, k_best_independent


TRUTH_VALUES = {"false": [False], "true": [True], "unknown": [False, True]}
TOKEN_TRUTH = {"A": "true", "B": "false", "C": "unknown"}


def log_softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    shifted = [math.exp(float(value) - maximum) for value in values]
    denominator = math.log(sum(shifted)) + maximum
    return [float(value) - denominator for value in values]


def enumerate_scene_graphs(
    scene: dict[str, Any], proposal_pairs: Sequence[dict[str, Any]],
    feature_lookup: dict[str, np.ndarray], match_coef: np.ndarray, match_intercept: float,
    truth_logits: dict[tuple[str, str, str], dict[str, float]], config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_ids = [row["id"] for row in scene["agent_input"]["atom_candidates"]]
    evidence_ids = [row["id"] for row in scene["agent_input"]["evidence"]]
    candidate_index = {value: index for index, value in enumerate(candidate_ids)}
    pair_by_edge = {
        (row["evidence_id"], row["candidate_id"]): row for row in proposal_pairs
    }
    scores = np.full((len(evidence_ids), len(candidate_ids)), -np.inf, dtype=np.float64)
    for evidence_index, evidence_id in enumerate(evidence_ids):
        for (current_evidence, candidate_id), pair in pair_by_edge.items():
            if current_evidence != evidence_id:
                continue
            vector = feature_lookup[pair["id"]]
            scores[evidence_index, candidate_index[candidate_id]] = float(
                vector @ match_coef + match_intercept
            )
    assignments = k_best_assignments(
        scores, config["jointMap"]["maximumAssignmentsPerScene"]
    )
    candidates = []
    maximum_unknown = config["jointMap"]["maximumUnknownAtomsPerGraph"]
    for assignment_score, assignment in assignments:
        choices = []
        for evidence_index, candidate_position in enumerate(assignment):
            key = (scene["id"], evidence_ids[evidence_index], candidate_ids[candidate_position])
            logits = truth_logits[key]
            tokens = ("A", "B", "C")
            log_probabilities = log_softmax([logits[token] for token in tokens])
            choices.append([
                (TOKEN_TRUTH[token], math.exp(log_probability))
                for token, log_probability in zip(tokens, log_probabilities, strict=True)
            ])
        truth_vectors = k_best_independent(
            choices, config["jointMap"]["truthVectorsPerAssignment"]
        )
        for truth_score, truth in truth_vectors:
            if sum(value == "unknown" for value in truth) > maximum_unknown:
                continue
            prediction_rows = [
                {
                    "evidence_id": evidence_ids[evidence_index],
                    "candidate_id": candidate_ids[candidate_position],
                    "truth_label": truth[evidence_index],
                }
                for evidence_index, candidate_position in enumerate(assignment)
            ]
            atom_by_candidate = {
                row["candidate_id"]: row["atom"] for row in scene["target"]["atom_groundings"]
            }
            epistemic = sorted([
                {
                    "atom": atom_by_candidate[row["candidate_id"]],
                    "allowed_values": TRUTH_VALUES[row["truth_label"]],
                }
                for row in prediction_rows
            ], key=lambda row: row["atom"])
            candidates.append({
                "log_score": float(assignment_score + truth_score),
                "assignment_log_score": float(assignment_score),
                "truth_log_score": float(truth_score),
                "assignment": assignment,
                "truth": truth,
                "prediction_rows": prediction_rows,
                "epistemic_state": epistemic,
                "graph_key": canonical_json(epistemic),
            })
    candidates.sort(key=lambda row: (-row["log_score"], row["graph_key"], row["assignment"], row["truth"]))
    result = []
    seen = set()
    for row in candidates:
        key = (row["assignment"], row["truth"])
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) == config["jointMap"]["graphBranchesPerScene"]:
            break
    return result


def compatibility_matrix(
    graphs: Sequence[dict[str, Any]], hypotheses: Sequence[Any], support: dict[str, Any],
    observed: str, v22_config: dict[str, Any], maximum_unknown: int,
) -> np.ndarray:
    result = np.zeros((len(graphs), len(hypotheses)), dtype=bool)
    for graph_index, graph in enumerate(graphs):
        state = rows_to_epistemic(graph["epistemic_state"])
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            possible = execute_partial(
                [hypothesis.program], v22_config, support["entities"], state,
                support["action_binding"], maximum_unknown,
            )["possible_transition_codes"]
            result[graph_index, hypothesis_index] = observed in possible
    return result


def select_episode_map(
    hypotheses: Sequence[Any], scene_graphs: Sequence[Sequence[dict[str, Any]]],
    compatibility: Sequence[np.ndarray],
) -> dict[str, Any] | None:
    candidates = []
    for program_index, hypothesis in enumerate(hypotheses):
        selected = []
        total = 0.0
        feasible = True
        for graphs, matrix in zip(scene_graphs, compatibility, strict=True):
            indices = np.flatnonzero(matrix[:, program_index])
            if not len(indices):
                feasible = False
                break
            graph_index = int(indices[0])
            selected.append(graph_index)
            total += graphs[graph_index]["log_score"]
        if feasible:
            candidates.append((total, hypothesis.key, program_index, tuple(selected)))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row[0], row[1], row[3]))
    score, program_key, program_index, graph_indices = candidates[0]
    return {
        "log_score": float(score), "program_key": program_key,
        "program_index": program_index, "graph_indices": graph_indices,
        "feasible_programs": len(candidates),
    }
