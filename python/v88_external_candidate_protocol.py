#!/usr/bin/env python3
"""Frozen prompting, parsing, controls, and scoring for V88."""
from __future__ import annotations

from collections import defaultdict
import json
from typing import Any


def format_user_prompt(record: dict[str, Any], config: dict[str, Any]) -> str:
    schema = record["schema_context"]
    intent_lines = "\n".join(f"- {item['id']}: {item['description']}" for item in schema["intents"])
    slot_lines = "\n".join(f"- {item['id']}: {item['description']}" for item in schema["slots"])
    history = "\n".join(f"{turn['speaker']}: {turn['utterance']}" for turn in record["dialogue_history"])
    return config["userPromptTemplate"].format(
        service_name=schema["service_name"],
        service_description=schema["service_description"],
        intent_lines=intent_lines,
        slot_lines=slot_lines,
        dialogue_history=history,
    )


def set_precision(predicted: set[str], gold: set[str]) -> float:
    if not predicted:
        return 1.0 if not gold else 0.0
    return len(predicted & gold) / len(predicted)


def set_recall(predicted: set[str], gold: set[str]) -> float:
    if not gold:
        return 1.0 if not predicted else 0.0
    return len(predicted & gold) / len(gold)


def parse_candidate_response(response: str, record: dict[str, Any]) -> dict[str, Any]:
    parsed = None
    exact_json = False
    exact_keys = False
    lists_well_formed = False
    unique_lists = False
    ontology_conformant = False
    intents: list[str] = []
    slots: list[str] = []
    try:
        parsed = json.loads(response.strip())
        exact_json = isinstance(parsed, dict)
    except (json.JSONDecodeError, TypeError):
        pass
    if exact_json:
        exact_keys = set(parsed) == {"intent_candidates", "state_slot_key_candidates"}
        raw_intents = parsed.get("intent_candidates")
        raw_slots = parsed.get("state_slot_key_candidates")
        lists_well_formed = bool(
            isinstance(raw_intents, list) and raw_intents
            and isinstance(raw_slots, list)
            and all(isinstance(item, str) for item in raw_intents + raw_slots)
        )
        if lists_well_formed:
            intents = raw_intents
            slots = raw_slots
            unique_lists = len(intents) == len(set(intents)) and len(slots) == len(set(slots))
            ontology_conformant = bool(
                exact_keys and unique_lists
                and set(intents) <= set(record["allowed_intent_ids"])
                and set(slots) <= set(record["allowed_slot_ids"])
            )
    return {
        "parsed": parsed,
        "exact_json": exact_json,
        "exact_keys": exact_keys,
        "lists_well_formed": lists_well_formed,
        "unique_lists": unique_lists,
        "ontology_conformant": ontology_conformant,
        "intent_candidates": intents,
        "state_slot_key_candidates": slots,
    }


def score_response(record: dict[str, Any], response: str) -> dict[str, Any]:
    parsed = parse_candidate_response(response, record)
    predicted_intents = set(parsed["intent_candidates"]) if parsed["ontology_conformant"] else set()
    predicted_slots = set(parsed["state_slot_key_candidates"]) if parsed["ontology_conformant"] else set()
    gold_intents = set(record["gold"]["intent_candidates"])
    gold_slots = set(record["gold"]["state_slot_key_candidates"])
    active = record["gold"]["active_intent"]
    return {
        "id": record["id"],
        "source_record_id": record["source_record_id"],
        "service": record["service"],
        "active_intent": active,
        "response": response,
        **parsed,
        "mandatory_NONE_included": "NONE" in predicted_intents,
        "gold_active_intent_covered": active == "NONE" or active in predicted_intents,
        "intent_candidate_precision": set_precision(predicted_intents, gold_intents),
        "intent_candidate_recall": set_recall(predicted_intents, gold_intents),
        "intent_candidate_exact": predicted_intents == gold_intents,
        "none_only_intent_exact": active != "NONE" or predicted_intents == {"NONE"},
        "state_slot_key_precision": set_precision(predicted_slots, gold_slots),
        "state_slot_key_recall": set_recall(predicted_slots, gold_slots),
        "state_slot_key_exact": predicted_slots == gold_slots,
        "intent_candidate_count": len(predicted_intents),
        "permanently_non_deployable": True,
        "executable": False,
        "gold": record["gold"],
        "allowed_intent_ids": record["allowed_intent_ids"],
        "allowed_slot_ids": record["allowed_slot_ids"],
    }


def control_rows(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gold_intents = set(record["gold"]["intent_candidates"])
    gold_slots = set(record["gold"]["state_slot_key_candidates"])
    exhaustive_intents = set(record["allowed_intent_ids"])
    exhaustive_slots = set(record["allowed_slot_ids"])
    none_only = {"NONE"}
    return {
        "exhaustive": {
            "intent_exact": exhaustive_intents == gold_intents,
            "intent_precision": set_precision(exhaustive_intents, gold_intents),
            "intent_recall": set_recall(exhaustive_intents, gold_intents),
            "state_exact": exhaustive_slots == gold_slots,
            "state_precision": set_precision(exhaustive_slots, gold_slots),
            "state_recall": set_recall(exhaustive_slots, gold_slots),
        },
        "none_only": {
            "intent_exact": none_only == gold_intents,
            "intent_precision": set_precision(none_only, gold_intents),
            "intent_recall": set_recall(none_only, gold_intents),
            "state_exact": not gold_slots,
            "state_precision": set_precision(set(), gold_slots),
            "state_recall": set_recall(set(), gold_slots),
        },
        "empty_state_gold_intent": {
            "intent_exact": True,
            "intent_precision": 1.0,
            "intent_recall": 1.0,
            "state_exact": not gold_slots,
            "state_precision": set_precision(set(), gold_slots),
            "state_recall": set_recall(set(), gold_slots),
        },
        "oracle": {
            "intent_exact": True,
            "intent_precision": 1.0,
            "intent_recall": 1.0,
            "state_exact": True,
            "state_precision": 1.0,
            "state_recall": 1.0,
        },
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_service_recalls: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_service_recalls[row["service"]].append(row["intent_candidate_recall"])
    controls = {name: [] for name in ("exhaustive", "none_only", "empty_state_gold_intent", "oracle")}
    for row in rows:
        for name, control in control_rows(records_by_id[row["id"]]).items():
            controls[name].append(control)
    control_metrics = {
        name: {
            "intent_exact_rate": mean([float(item["intent_exact"]) for item in items]),
            "intent_precision": mean([item["intent_precision"] for item in items]),
            "intent_recall": mean([item["intent_recall"] for item in items]),
            "state_exact_rate": mean([float(item["state_exact"]) for item in items]),
            "state_precision": mean([item["state_precision"] for item in items]),
            "state_recall": mean([item["state_recall"] for item in items]),
        }
        for name, items in controls.items()
    }
    active_rows = [row for row in rows if row["active_intent"] != "NONE"]
    none_rows = [row for row in rows if row["active_intent"] == "NONE"]
    intent_exact = mean([float(row["intent_candidate_exact"]) for row in rows])
    return {
        "record_count": len(rows),
        "service_count": len({row["service"] for row in rows}),
        "active_intent_label_count": len({row["active_intent"] for row in active_rows}),
        "none_record_count": len(none_rows),
        "exact_JSON_parse_rate": mean([float(row["exact_json"]) for row in rows]),
        "ontology_conformance_rate": mean([float(row["ontology_conformant"]) for row in rows]),
        "mandatory_NONE_inclusion_rate": mean([float(row["mandatory_NONE_included"]) for row in rows]),
        "permanent_non_deployable_rate": mean([float(row["permanently_non_deployable"]) for row in rows]),
        "gold_active_intent_coverage_rate": mean([float(row["gold_active_intent_covered"]) for row in active_rows]),
        "intent_candidate_set_exact_rate": intent_exact,
        "intent_candidate_precision": mean([row["intent_candidate_precision"] for row in rows]),
        "intent_candidate_recall": mean([row["intent_candidate_recall"] for row in rows]),
        "none_only_intent_exact_rate": mean([float(row["intent_candidate_exact"]) for row in none_rows]),
        "per_service_intent_candidate_recall": {
            service: mean(values) for service, values in sorted(per_service_recalls.items())
        },
        "state_slot_key_precision": mean([row["state_slot_key_precision"] for row in rows]),
        "state_slot_key_recall": mean([row["state_slot_key_recall"] for row in rows]),
        "state_slot_key_exact_rate": mean([float(row["state_slot_key_exact"]) for row in rows]),
        "intent_exact_improvement_over_exhaustive": intent_exact - control_metrics["exhaustive"]["intent_exact_rate"],
        "mean_intent_candidate_count": mean([float(row["intent_candidate_count"]) for row in rows]),
        "controls": control_metrics,
    }


def evaluate_gates(metrics: dict[str, Any], config: dict[str, Any], access: dict[str, int]) -> dict[str, bool]:
    gates = config["gates"]
    return {
        "required_record_count": metrics["record_count"] == gates["requiredRecordCount"],
        "required_service_count": metrics["service_count"] == gates["requiredServiceCount"],
        "required_active_intent_label_count": metrics["active_intent_label_count"] == gates["requiredActiveIntentLabelCount"],
        "required_NONE_record_count": metrics["none_record_count"] == gates["requiredNoneRecordCount"],
        "exact_JSON_parse": metrics["exact_JSON_parse_rate"] >= gates["minimumExactJSONParseRate"],
        "ontology_conformance": metrics["ontology_conformance_rate"] >= gates["minimumOntologyConformanceRate"],
        "mandatory_NONE_inclusion": metrics["mandatory_NONE_inclusion_rate"] >= gates["minimumMandatoryNoneInclusionRate"],
        "permanent_non_deployable": metrics["permanent_non_deployable_rate"] >= gates["minimumPermanentNonDeployableRate"],
        "gold_active_intent_coverage": metrics["gold_active_intent_coverage_rate"] >= gates["minimumGoldActiveIntentCoverageRate"],
        "intent_candidate_set_exact": metrics["intent_candidate_set_exact_rate"] >= gates["minimumIntentCandidateSetExactRate"],
        "NONE_only_intent_exact": metrics["none_only_intent_exact_rate"] >= gates["minimumNoneOnlyIntentExactRate"],
        "per_service_intent_candidate_recall": all(
            value >= gates["minimumPerServiceIntentCandidateRecallRate"]
            for value in metrics["per_service_intent_candidate_recall"].values()
        ),
        "state_slot_key_recall": metrics["state_slot_key_recall"] >= gates["minimumStateSlotKeyRecallRate"],
        "state_slot_key_exact": metrics["state_slot_key_exact_rate"] >= gates["minimumStateSlotKeyExactRate"],
        "intent_exact_improvement_over_exhaustive": metrics["intent_exact_improvement_over_exhaustive"] >= gates["minimumIntentExactImprovementOverExhaustive"],
        "mean_intent_candidate_count": metrics["mean_intent_candidate_count"] <= gates["maximumMeanIntentCandidateCount"],
        "model_load_budget": access["model_load_count"] <= gates["maximumModelLoadCount"],
        "model_generation_budget": access["model_generation_count"] <= gates["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= gates["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= gates["maximumAdapterTrainingRunCount"],
        "zero_manual_utterance_inspection": access["manual_utterance_inspection_count"] <= gates["maximumManualUtteranceInspectionCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= gates["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"],
    }
