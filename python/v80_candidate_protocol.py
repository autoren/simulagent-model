#!/usr/bin/env python3
"""Strict parser and scorer for frozen V80 candidate-generation responses."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable
import json


FORBIDDEN_CONFIDENCE_KEYS = {"confidence", "confidences", "probability", "probabilities", "score", "scores"}
FORBIDDEN_ACTION_KEYS = {"action", "actions", "tool", "tools", "tool_call", "tool_calls"}


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def parse_candidate_response(response: str, config: dict[str, Any]) -> dict[str, Any]:
    parsed = None
    parse_error = None
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError) as error:
        parse_error = str(error)
    keys = list(recursive_keys(parsed)) if parsed is not None else []
    confidence_count = sum(key.lower() in FORBIDDEN_CONFIDENCE_KEYS for key in keys)
    action_count = sum(key.lower() in FORBIDDEN_ACTION_KEYS for key in keys)
    candidates = parsed.get("candidate_ids") if isinstance(parsed, dict) else None
    candidate_list = candidates if isinstance(candidates, list) else []
    allowed = config["candidateIdsInRequiredOrder"]
    allowed_set = set(allowed)
    string_candidates = [value for value in candidate_list if isinstance(value, str)]
    unknown_count = sum(value not in allowed_set for value in string_candidates)
    duplicate_count = len(string_candidates) - len(set(string_candidates))
    canonical = string_candidates == [
        value for value in allowed if value in string_candidates
    ]
    contract = config["outputContract"]
    schema_valid = bool(
        isinstance(parsed, dict)
        and list(parsed.keys()) == contract["exactTopLevelKeys"]
        and isinstance(candidates, list)
        and len(candidate_list) == len(string_candidates)
        and contract["minimumCandidates"] <= len(string_candidates) <= contract["maximumCandidates"]
        and unknown_count == 0
        and duplicate_count == 0
        and canonical
        and "none_of_the_above" in string_candidates
        and confidence_count == 0
        and action_count == 0
    )
    return {
        "exact_json_parse": parsed is not None,
        "parse_error": parse_error,
        "schema_valid": schema_valid,
        "candidate_ids": string_candidates,
        "candidate_count": len(string_candidates),
        "none_of_the_above_included": "none_of_the_above" in string_candidates,
        "canonical_order": canonical if string_candidates else False,
        "unknown_candidate_id_count": unknown_count,
        "duplicate_candidate_id_count": duplicate_count,
        "confidence_or_probability_field_count": confidence_count,
        "action_or_tool_field_count": action_count,
    }


def score_record(record: dict[str, Any], response: str, config: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_candidate_response(response, config)
    gold = record["goldCandidateIds"]
    predicted = parsed["candidate_ids"] if parsed["schema_valid"] else []
    recall = len(set(predicted) & set(gold)) / len(gold)
    return {
        "id": record["id"],
        "stratum": record["stratum"],
        "instruction": record["instruction"],
        "gold_candidate_ids": gold,
        "raw_response": response,
        **parsed,
        "gold_candidate_recall": float(recall),
        "exact_candidate_set": predicted == gold,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("V80 cannot aggregate an empty population")
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[row["stratum"]].append(row)
    mean = lambda values: float(sum(values) / len(values))
    return {
        "record_count": len(rows),
        "stratum_counts": dict(sorted(Counter(row["stratum"] for row in rows).items())),
        "exact_json_parse_rate": mean([row["exact_json_parse"] for row in rows]),
        "schema_validity_rate": mean([row["schema_valid"] for row in rows]),
        "none_of_the_above_inclusion_rate": mean(
            [row["none_of_the_above_included"] for row in rows]
        ),
        "mean_gold_candidate_recall": mean(
            [row["gold_candidate_recall"] for row in rows]
        ),
        "per_stratum_mean_gold_candidate_recall": {
            stratum: mean([row["gold_candidate_recall"] for row in members])
            for stratum, members in sorted(by_stratum.items())
        },
        "exact_candidate_set_accuracy": mean(
            [row["exact_candidate_set"] for row in rows]
        ),
        "clear_exact_candidate_set_accuracy": mean(
            [row["exact_candidate_set"] for row in by_stratum["clear"]]
        ),
        "out_of_ontology_exact_candidate_set_accuracy": mean(
            [row["exact_candidate_set"] for row in by_stratum["out_of_ontology"]]
        ),
        "canonical_order_rate": mean([row["canonical_order"] for row in rows]),
        "mean_candidate_count": mean([row["candidate_count"] for row in rows]),
        "confidence_or_probability_field_count": sum(
            row["confidence_or_probability_field_count"] for row in rows
        ),
        "action_or_tool_field_count": sum(
            row["action_or_tool_field_count"] for row in rows
        ),
        "unknown_candidate_id_count": sum(
            row["unknown_candidate_id_count"] for row in rows
        ),
        "duplicate_candidate_id_count": sum(
            row["duplicate_candidate_id_count"] for row in rows
        ),
    }


def evaluate_gates(
    metrics: dict[str, Any], config: dict[str, Any], access: dict[str, int]
) -> dict[str, bool]:
    gates = config["gates"]
    return {
        "complete_record_and_stratum_census": bool(
            metrics["record_count"] == gates["requiredRecordCount"]
            and metrics["stratum_counts"] == gates["requiredStratumCounts"]
        ),
        "exact_JSON_parse_rate": metrics["exact_json_parse_rate"]
        >= gates["minimumExactJSONParseRate"],
        "schema_validity_rate": metrics["schema_validity_rate"]
        >= gates["minimumSchemaValidityRate"],
        "none_of_the_above_inclusion_rate": metrics[
            "none_of_the_above_inclusion_rate"
        ]
        >= gates["minimumNoneOfTheAboveInclusionRate"],
        "mean_gold_candidate_recall": metrics["mean_gold_candidate_recall"]
        >= gates["minimumMeanGoldCandidateRecall"],
        "per_stratum_gold_candidate_recall": all(
            value >= gates["minimumPerStratumMeanGoldCandidateRecall"]
            for value in metrics["per_stratum_mean_gold_candidate_recall"].values()
        ),
        "exact_candidate_set_accuracy": metrics["exact_candidate_set_accuracy"]
        >= gates["minimumExactCandidateSetAccuracy"],
        "clear_exact_candidate_set_accuracy": metrics[
            "clear_exact_candidate_set_accuracy"
        ]
        >= gates["minimumClearExactCandidateSetAccuracy"],
        "out_of_ontology_exact_candidate_set_accuracy": metrics[
            "out_of_ontology_exact_candidate_set_accuracy"
        ]
        >= gates["minimumOutOfOntologyExactCandidateSetAccuracy"],
        "canonical_order_rate": metrics["canonical_order_rate"]
        >= gates["minimumCanonicalOrderRate"],
        "bounded_mean_candidate_count": metrics["mean_candidate_count"]
        <= gates["maximumMeanCandidateCount"],
        "zero_confidence_or_probability_fields": metrics[
            "confidence_or_probability_field_count"
        ]
        <= gates["maximumConfidenceOrProbabilityFieldCount"],
        "zero_action_or_tool_fields": metrics["action_or_tool_field_count"]
        <= gates["maximumActionOrToolFieldCount"],
        "zero_unknown_candidate_ids": metrics["unknown_candidate_id_count"]
        <= gates["maximumUnknownCandidateIdCount"],
        "zero_duplicate_candidate_ids": metrics["duplicate_candidate_id_count"]
        <= gates["maximumDuplicateCandidateIdCount"],
        "bounded_local_model_and_zero_external_access": bool(
            access["model_forward_pass_count"] <= gates["maximumModelForwardPassCount"]
            and access["API_call_count"] <= gates["maximumAPICallCount"]
            and access["adapter_training_run_count"]
            <= gates["maximumAdapterTrainingRunCount"]
            and access["human_record_access_count"]
            <= gates["maximumHumanRecordAccessCount"]
            and access["real_tool_call_count"] <= gates["maximumRealToolCallCount"]
            and access["external_side_effect_count"]
            <= gates["maximumExternalSideEffectCount"]
        ),
    }
