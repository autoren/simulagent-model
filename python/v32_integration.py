"""Typed global atom alignment for a conditionally selected V32 head."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from v30_language import TRUTH_VALUES, log_softmax, parse_public_candidate_statements, predicate_specs
from v32_language import compile_truth


def masked_log_probability(logits: Sequence[float], index: int, allowed: Sequence[int]) -> float:
    normalized = log_softmax([float(logits[position]) for position in allowed])
    return normalized[list(allowed).index(index)]


def candidate_score(score, fact, entities, config):
    predicate_index = config["sharedHead"]["predicateClasses"].index(fact["predicate"])
    predicate_log = log_softmax(score["mean_logits"]["predicate"])[predicate_index]
    spec, ids = predicate_specs(config)[fact["predicate"]], [entity["id"] for entity in entities]
    if spec["kind"] == "unary":
        allowed1 = [i for i, entity in enumerate(entities) if entity["entity_type"] == spec["entityType"]]
        argument1, argument2, allowed2 = ids.index(fact["arguments"][0]), max(config["construction"]["entityCounts"]), [max(config["construction"]["entityCounts"])]
    else:
        allowed1 = [i for i, entity in enumerate(entities) if entity["entity_type"] == spec["sourceType"]]
        argument1, argument2 = ids.index(fact["arguments"][0]), ids.index(fact["arguments"][1])
        allowed2 = [i for i, entity in enumerate(entities) if entity["entity_type"] == spec["targetType"] and i != argument1]
    return predicate_log + masked_log_probability(score["mean_logits"]["argument1"], argument1, allowed1) + masked_log_probability(score["mean_logits"]["argument2"], argument2, allowed2)


def truth_from_score(score, selected_system, config):
    head = config["sharedHead"]
    if selected_system in ("monolithic", "auxiliaryDirect"):
        return head["truthClasses"][int(np.argmax(score["mean_logits"]["truth"]))]
    sign = head["lexicalSignClasses"][int(np.argmax(score["mean_logits"]["lexical_sign"]))]
    operation = head["outerOperationClasses"][int(np.argmax(score["mean_logits"]["outer_operation"]))]
    return compile_truth(sign, operation, config)


def assemble_scene_prediction(scene: dict[str, Any], scores: Sequence[dict[str, Any]], selected_system: str, config: dict[str, Any]) -> dict[str, Any]:
    public, by_evidence = scene["agent_input"], {row["evidence_id"]: row for row in scores}
    evidence_ids = [row["id"] for row in public["evidence"]]
    candidate_ids = [row["id"] for row in public["atom_candidates"]]
    if set(by_evidence) != set(evidence_ids): raise ValueError("V32 scores do not cover evidence")
    facts = parse_public_candidate_statements(public["atom_candidates"], public["entities"], config)
    matrix = np.asarray([[candidate_score(by_evidence[evidence], facts[candidate], public["entities"], config) for candidate in candidate_ids] for evidence in evidence_ids])
    evidence_indices, candidate_indices = linear_sum_assignment(-matrix)
    assignment = dict(zip(evidence_indices.tolist(), candidate_indices.tolist(), strict=True))
    rows, epistemic = [], []
    for evidence_index, evidence_id in enumerate(evidence_ids):
        candidate_id, score = candidate_ids[assignment[evidence_index]], by_evidence[evidence_id]
        truth = truth_from_score(score, selected_system, config)
        rows.append({"evidence_id": evidence_id, "candidate_id": candidate_id, "truth_label": truth, "deterministic_alignment_score": float(matrix[evidence_index, assignment[evidence_index]])})
        epistemic.append({"atom": facts[candidate_id]["atom"], "allowed_values": TRUTH_VALUES[truth]})
    return {"scene_id": scene["id"], "episode_id": scene["episode_id"], "split": scene["split"], "role": scene["role"], "rows": rows, "epistemic_state": sorted(epistemic, key=lambda row: row["atom"])}
