#!/usr/bin/env python3
"""Strict V81 Boolean-label parser, deterministic composer, scorer, and gates."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable


CONFIDENCE_KEYS = {
    "confidence",
    "confidences",
    "probability",
    "probabilities",
    "score",
    "scores",
}
ACTION_KEYS = {
    "action",
    "actions",
    "tool",
    "tools",
    "tool_call",
    "tool_calls",
}
CANDIDATE_KEYS = {"candidate", "candidates", "candidate_id", "candidate_ids"}


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def compose_candidates(labels: dict[str, bool]) -> list[str]:
    if labels.get("out_of_ontology", False):
        return ["none_of_the_above"]
    result: list[str] = []
    for recipient in ("alex_chen", "alex_kim"):
        for operation in ("schedule_review", "send_summary"):
            if labels.get(operation, False) and labels.get(recipient, False):
                result.append(f"{operation}__{recipient}")
    result.append("none_of_the_above")
    return result


def parse_factorized_response(response: str, config: dict[str, Any]) -> dict[str, Any]:
    value: Any = None
    parse_error = None
    try:
        value = json.loads(response)
    except (json.JSONDecodeError, TypeError) as error:
        parse_error = str(error)
    keys = list(recursive_keys(value)) if value is not None else []
    confidence_count = sum(key.lower() in CONFIDENCE_KEYS for key in keys)
    action_count = sum(key.lower() in ACTION_KEYS for key in keys)
    candidate_field_count = sum(key.lower() in CANDIDATE_KEYS for key in keys)
    expected_keys = config["labelKeysInRequiredOrder"]
    labels = {
        key: value[key]
        for key in expected_keys
        if isinstance(value, dict) and key in value and type(value[key]) is bool
    }
    booleans_exact = len(labels) == len(expected_keys)
    ontology_consistent = bool(
        booleans_exact
        and labels["out_of_ontology"]
        == (not labels["schedule_review"] and not labels["send_summary"])
    )
    schema_valid = bool(
        isinstance(value, dict)
        and list(value) == expected_keys
        and booleans_exact
        and ontology_consistent
        and confidence_count == 0
        and action_count == 0
        and candidate_field_count == 0
    )
    composed = compose_candidates(labels) if schema_valid else []
    canonical = composed == [
        candidate
        for candidate in config["candidateIdsInRequiredOrder"]
        if candidate in composed
    ]
    return {
        "exact_json_parse": value is not None,
        "parse_error": parse_error,
        "schema_valid": schema_valid,
        "labels": labels,
        "ontology_consistent": ontology_consistent,
        "candidate_ids": composed,
        "candidate_count": len(composed),
        "none_of_the_above_included": "none_of_the_above" in composed,
        "canonical_order": canonical if composed else False,
        "confidence_or_probability_field_count": confidence_count,
        "action_or_tool_field_count": action_count,
        "candidate_id_field_count": candidate_field_count,
    }


def score_record(record: dict[str, Any], response: str, config: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_factorized_response(response, config)
    labels = parsed["labels"] if parsed["schema_valid"] else {}
    gold_labels = record["goldLabels"]
    label_accuracy = (
        sum(labels.get(key) == value for key, value in gold_labels.items())
        / len(gold_labels)
        if parsed["schema_valid"]
        else 0.0
    )
    candidates = parsed["candidate_ids"]
    gold_candidates = record["goldCandidateIds"]
    recall = len(set(candidates) & set(gold_candidates)) / len(gold_candidates)
    return {
        "id": record["id"],
        "stratum": record["stratum"],
        "instruction": record["instruction"],
        "gold_labels": gold_labels,
        "gold_candidate_ids": gold_candidates,
        "raw_response": response,
        **parsed,
        "label_accuracy": float(label_accuracy),
        "exact_label_vector": labels == gold_labels,
        "out_of_ontology_label_correct": bool(
            labels.get("out_of_ontology") == gold_labels["out_of_ontology"]
        ),
        "gold_candidate_recall": float(recall),
        "exact_candidate_set": candidates == gold_candidates,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("V81 cannot aggregate an empty population")
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[row["stratum"]].append(row)
    mean = lambda values: float(sum(values) / len(values))
    return {
        "record_count": len(rows),
        "stratum_counts": dict(
            sorted(Counter(row["stratum"] for row in rows).items())
        ),
        "exact_json_parse_rate": mean([row["exact_json_parse"] for row in rows]),
        "schema_validity_rate": mean([row["schema_valid"] for row in rows]),
        "exact_label_vector_accuracy": mean(
            [row["exact_label_vector"] for row in rows]
        ),
        "mean_label_accuracy": mean([row["label_accuracy"] for row in rows]),
        "out_of_ontology_label_accuracy": mean(
            [row["out_of_ontology_label_correct"] for row in by_stratum["out_of_ontology"]]
        ),
        "none_of_the_above_inclusion_rate": mean(
            [row["none_of_the_above_included"] for row in rows]
        ),
        "mean_gold_candidate_recall": mean(
            [row["gold_candidate_recall"] for row in rows]
        ),
        "per_stratum_mean_gold_candidate_recall": {
            name: mean([row["gold_candidate_recall"] for row in members])
            for name, members in sorted(by_stratum.items())
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
        "candidate_id_field_count": sum(
            row["candidate_id_field_count"] for row in rows
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
        "exact_label_vector_accuracy": metrics["exact_label_vector_accuracy"]
        >= gates["minimumExactLabelVectorAccuracy"],
        "mean_label_accuracy": metrics["mean_label_accuracy"]
        >= gates["minimumMeanLabelAccuracy"],
        "out_of_ontology_label_accuracy": metrics["out_of_ontology_label_accuracy"]
        >= gates["minimumOutOfOntologyLabelAccuracy"],
        "none_of_the_above_inclusion_rate": metrics[
            "none_of_the_above_inclusion_rate"
        ]
        >= 1.0,
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
        "zero_candidate_ID_fields_from_model": metrics["candidate_id_field_count"]
        <= gates["maximumCandidateIdFieldCount"],
        "bounded_local_model_and_zero_external_access": bool(
            access["model_generation_count"] <= gates["maximumModelGenerationCount"]
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
