"""Typed global alignment for the conditional V31/V28 replay."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from v30_language import TRUTH_VALUES, log_softmax, parse_public_candidate_statements, predicate_specs


def masked_log_probability(logits: Sequence[float], index: int, allowed: Sequence[int]) -> float:
    values = [float(logits[position]) for position in allowed]
    normalized = log_softmax(values)
    return normalized[list(allowed).index(index)]


def candidate_score(
    score: dict[str, Any], fact: dict[str, Any], entities: Sequence[dict[str, str]],
    config: dict[str, Any],
) -> float:
    predicates = config["sharedStructuredHead"]["predicateClasses"]
    predicate_index = predicates.index(fact["predicate"])
    predicate_log = log_softmax(score["mean_logits"]["predicate"])[predicate_index]
    specs = predicate_specs(config)
    spec = specs[fact["predicate"]]
    entity_ids = [entity["id"] for entity in entities]
    if spec["kind"] == "unary":
        allowed1 = [
            index for index, entity in enumerate(entities)
            if entity["entity_type"] == spec["entityType"]
        ]
        argument1_index = entity_ids.index(fact["arguments"][0])
        argument2_index = max(config["construction"]["entityCounts"])
        allowed2 = [argument2_index]
    else:
        allowed1 = [
            index for index, entity in enumerate(entities)
            if entity["entity_type"] == spec["sourceType"]
        ]
        argument1_index = entity_ids.index(fact["arguments"][0])
        argument2_index = entity_ids.index(fact["arguments"][1])
        allowed2 = [
            index for index, entity in enumerate(entities)
            if entity["entity_type"] == spec["targetType"] and index != argument1_index
        ]
    return (
        predicate_log
        + masked_log_probability(score["mean_logits"]["argument1"], argument1_index, allowed1)
        + masked_log_probability(score["mean_logits"]["argument2"], argument2_index, allowed2)
    )


def assemble_scene_prediction(
    scene: dict[str, Any], scores: Sequence[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    public = scene["agent_input"]
    by_evidence = {row["evidence_id"]: row for row in scores}
    evidence_ids = [row["id"] for row in public["evidence"]]
    candidate_ids = [row["id"] for row in public["atom_candidates"]]
    if set(by_evidence) != set(evidence_ids):
        raise ValueError("V31 integration scores do not cover every evidence clause")
    facts = parse_public_candidate_statements(
        public["atom_candidates"], public["entities"], config
    )
    matrix = np.empty((len(evidence_ids), len(candidate_ids)), dtype=np.float64)
    for evidence_index, evidence_id in enumerate(evidence_ids):
        for candidate_index, candidate_id in enumerate(candidate_ids):
            matrix[evidence_index, candidate_index] = candidate_score(
                by_evidence[evidence_id], facts[candidate_id], public["entities"], config
            )
    evidence_indices, candidate_indices = linear_sum_assignment(-matrix)
    assignment = dict(zip(evidence_indices.tolist(), candidate_indices.tolist(), strict=True))
    truths = config["sharedStructuredHead"]["truthClasses"]
    rows, epistemic = [], []
    for evidence_index, evidence_id in enumerate(evidence_ids):
        score = by_evidence[evidence_id]
        candidate_id = candidate_ids[assignment[evidence_index]]
        truth_status = truths[int(np.argmax(score["mean_logits"]["truth"]))]
        rows.append({
            "evidence_id": evidence_id, "candidate_id": candidate_id,
            "truth_label": truth_status,
            "deterministic_alignment_score": float(matrix[evidence_index, assignment[evidence_index]]),
        })
        epistemic.append({
            "atom": facts[candidate_id]["atom"], "allowed_values": TRUTH_VALUES[truth_status],
        })
    return {
        "scene_id": scene["id"], "episode_id": scene["episode_id"],
        "split": scene["split"], "role": scene["role"], "rows": rows,
        "epistemic_state": sorted(epistemic, key=lambda row: row["atom"]),
    }
