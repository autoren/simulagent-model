#!/usr/bin/env python3
"""Frozen scoring and paired comparison helpers for V90."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from v88_external_candidate_protocol import aggregate, format_user_prompt, score_response


QUALITY_GATE_KEYS = (
    "exact_JSON_parse",
    "ontology_conformance",
    "mandatory_NONE_inclusion",
    "permanent_non_deployable",
    "gold_active_intent_coverage",
    "intent_candidate_set_exact",
    "NONE_only_intent_exact",
    "per_service_intent_candidate_recall",
    "state_slot_key_recall",
    "state_slot_key_exact",
    "intent_exact_improvement_over_exhaustive",
    "mean_intent_candidate_count",
)


def evaluate_condition_gates(
    metrics: dict[str, Any], config: dict[str, Any], access: dict[str, Any]
) -> dict[str, bool]:
    quality = config["qualityGatesPerCondition"]
    limits = config["accessGatesPerCondition"]
    return {
        "required_record_count": metrics["record_count"] == limits["requiredRecordCount"],
        "exact_JSON_parse": metrics["exact_JSON_parse_rate"] >= quality["minimumExactJSONParseRate"],
        "ontology_conformance": metrics["ontology_conformance_rate"] >= quality["minimumOntologyConformanceRate"],
        "mandatory_NONE_inclusion": metrics["mandatory_NONE_inclusion_rate"] >= quality["minimumMandatoryNoneInclusionRate"],
        "permanent_non_deployable": metrics["permanent_non_deployable_rate"] >= quality["minimumPermanentNonDeployableRate"],
        "gold_active_intent_coverage": metrics["gold_active_intent_coverage_rate"] >= quality["minimumGoldActiveIntentCoverageRate"],
        "intent_candidate_set_exact": metrics["intent_candidate_set_exact_rate"] >= quality["minimumIntentCandidateSetExactRate"],
        "NONE_only_intent_exact": metrics["none_only_intent_exact_rate"] >= quality["minimumNoneOnlyIntentExactRate"],
        "per_service_intent_candidate_recall": all(
            value >= quality["minimumPerServiceIntentCandidateRecallRate"]
            for value in metrics["per_service_intent_candidate_recall"].values()
        ),
        "state_slot_key_recall": metrics["state_slot_key_recall"] >= quality["minimumStateSlotKeyRecallRate"],
        "state_slot_key_exact": metrics["state_slot_key_exact_rate"] >= quality["minimumStateSlotKeyExactRate"],
        "intent_exact_improvement_over_exhaustive": (
            metrics["intent_exact_improvement_over_exhaustive"]
            >= quality["minimumIntentExactImprovementOverExhaustive"]
        ),
        "mean_intent_candidate_count": metrics["mean_intent_candidate_count"] <= quality["maximumMeanIntentCandidateCount"],
        "model_load_budget": access["model_load_count"] <= limits["maximumModelLoadCount"],
        "model_generation_budget": access["model_generation_count"] <= limits["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= limits["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= limits["maximumAdapterTrainingRunCount"],
        "zero_manual_utterance_inspection": (
            access["manual_utterance_inspection_count"] <= limits["maximumManualUtteranceInspectionCount"]
        ),
        "zero_real_service_calls": access["real_service_call_count"] <= limits["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= limits["maximumExternalSideEffectCount"],
    }


def quality_gate_pass(gates: dict[str, bool]) -> bool:
    return all(gates[key] for key in QUALITY_GATE_KEYS)


def paired_correctness_transitions(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired condition fixture identities differ")
    metrics = {
        "active_intent_covered": "gold_active_intent_covered",
        "intent_exact": "intent_candidate_exact",
        "state_exact": "state_slot_key_exact",
        "ontology_conformant": "ontology_conformant",
    }
    output: dict[str, Any] = {}
    for label, key in metrics.items():
        counts = defaultdict(int)
        for record_id in sorted(left):
            pair = (bool(left[record_id][key]), bool(right[record_id][key]))
            counts[f"{str(pair[0]).lower()}_to_{str(pair[1]).lower()}"] += 1
        output[label] = dict(sorted(counts.items()))
    return output


def union_scored_row(record: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    conformant = bool(left["ontology_conformant"] and right["ontology_conformant"])
    if conformant:
        intents = sorted(set(left["intent_candidates"]) | set(right["intent_candidates"]), key=lambda x: (x == "NONE", x))
        slots = sorted(set(left["state_slot_key_candidates"]) | set(right["state_slot_key_candidates"]))
        if "NONE" not in intents:
            conformant = False
    if not conformant:
        intents = []
        slots = []
    import json

    response = json.dumps({
        "intent_candidates": intents,
        "state_slot_key_candidates": slots,
    }) if conformant else "fail-closed-invalid-union"
    row = score_response(record, response)
    row["name"] = record["id"]
    row["union_inputs_both_conformant"] = conformant
    row["permanently_non_deployable"] = True
    row["executable"] = False
    return row


__all__ = [
    "aggregate",
    "evaluate_condition_gates",
    "format_user_prompt",
    "paired_correctness_transitions",
    "quality_gate_pass",
    "score_response",
    "union_scored_row",
]
