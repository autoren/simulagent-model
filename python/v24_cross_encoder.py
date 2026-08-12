"""Proposal scoring and prompt construction for V24."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from evaluate_v22r2_relational_grounding import pair_features


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def old_match_scores(
    candidate_features: np.ndarray, evidence_feature: np.ndarray,
    atom_coef: np.ndarray, atom_intercept: float,
) -> np.ndarray:
    return sigmoid(
        pair_features(evidence_feature[None, :], candidate_features) @ atom_coef
        + atom_intercept
    )


def proposal_candidate_ids(
    candidate_ids: list[str], scores: np.ndarray, hard_candidate_id: str, top_k: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        range(len(candidate_ids)), key=lambda index: (-float(scores[index]), candidate_ids[index])
    )
    result = [
        {
            "candidate_id": candidate_ids[index],
            "proposal_sources": ["raw_top_k"],
            "raw_rank": rank,
            "old_match_score": float(scores[index]),
        }
        for rank, index in enumerate(ordered[:top_k], start=1)
    ]
    by_id = {row["candidate_id"]: row for row in result}
    if hard_candidate_id in by_id:
        by_id[hard_candidate_id]["proposal_sources"].append("global_hard_assignment")
    else:
        index = candidate_ids.index(hard_candidate_id)
        row = {
            "candidate_id": hard_candidate_id,
            "proposal_sources": ["global_hard_assignment"],
            "raw_rank": ordered.index(index) + 1,
            "old_match_score": float(scores[index]),
        }
        result.append(row)
    return result


def cross_prompt_layout(pair: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    public = pair["agent_input"]
    entities = ", ".join(
        f"{row['id']} ({row['entity_type']})" for row in public["entities"]
    )
    binding = public["action"]["binding"]
    prefix = (
        f"Typed entities: {entities}.\n"
        f"Action binding: actor={binding['actor']}, target={binding['target']}.\n"
        f"Evidence statement: {public['evidence_text']}\n"
        "Candidate fact: "
    )
    statement = public["candidate_statement"]
    return prefix + statement, (len(prefix), len(prefix) + len(statement))


def cross_prompt_text(pair: dict[str, Any]) -> str:
    return cross_prompt_layout(pair)[0]
