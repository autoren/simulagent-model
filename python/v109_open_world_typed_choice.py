from __future__ import annotations

import json
from typing import Any

from v107_open_world_local_model import aggregate_model_fixtures


def compile_choice_catalog(catalog: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = config["typedChoiceInterface"]
    width = spec["choiceWidth"]
    choices: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(catalog["intents"], key=lambda value: value["intent_id"])):
        choices.append({
            "choice_id": f"{spec['knownChoicePrefix']}{index:0{width}d}",
            "kind": "KNOWN", "scenario": row["scenario"],
            "intent_label": row["intent"], "intent_id": row["intent_id"],
            "slot_types": row["slot_types"],
        })
    for index, scenario in enumerate(sorted(catalog["scenarios"])):
        choices.append({
            "choice_id": f"{spec['novelScenarioChoicePrefix']}{index:0{width}d}",
            "kind": "NOVEL", "scenario": scenario,
            "meaning": "plausible capability in this scenario but no KNOWN choice matches",
        })
    choices.extend([
        {"choice_id": spec["unsupportedChoiceId"], "kind": "UNSUPPORTED", "meaning": "outside every visible scenario"},
        {"choice_id": spec["insufficientEvidenceChoiceId"], "kind": "ABSTAIN", "meaning": "insufficient evidence to choose"},
    ])
    ids = [row["choice_id"] for row in choices]
    if len(choices) != spec["requiredChoiceCount"] or len(ids) != len(set(ids)):
        raise ValueError("typed choice catalog is not complete and unique")
    return {"choices": choices, "choice_count": len(choices)}


def render_choice_prompt(
    choice_catalog: dict[str, Any], utterance: str | None,
    observation_available: bool, config: dict[str, Any],
) -> str:
    payload = {
        "instruction": (
            "Select exactly one choice_id from choices. Choose KNOWN only when one listed intent matches. "
            "Choose a NOVEL choice when the request is a plausible new capability inside that visible "
            "scenario but no KNOWN choice matches. Choose U00 when it is outside all visible scenarios. "
            "Choose A00 when evidence is insufficient. Return exactly one JSON object and no explanation."
        ),
        "choices": choice_catalog["choices"],
        "response_contract": {
            "required_keys": config["typedChoiceInterface"]["outputKeys"],
            "choice_id": "exactly one supplied choice_id",
            "confidence": "number from 0 through 1",
            "extra_keys_allowed": False,
        },
        "observation_available": observation_available,
        "user_utterance": (
            utterance if observation_available else config["prompt"]["missingObservationSentinel"]
        ),
    }
    if observation_available and not isinstance(utterance, str):
        raise ValueError("observed fixture requires an utterance")
    if not observation_available and utterance is not None:
        raise ValueError("missing-observation fixture cannot expose an utterance")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fallback() -> dict[str, Any]:
    return {"status": "ABSTAIN", "known_intent": None, "novel_scenario": None, "confidence": 0.0}


def validate_and_expand_choice(
    response: str | dict[str, Any], choice_catalog: dict[str, Any], config: dict[str, Any],
) -> tuple[dict[str, Any], bool, str]:
    try:
        value = json.loads(response) if isinstance(response, str) else response
    except (json.JSONDecodeError, TypeError):
        return _fallback(), False, "invalid_json"
    required = set(config["typedChoiceInterface"]["outputKeys"])
    if not isinstance(value, dict) or set(value) != required:
        return _fallback(), False, "invalid_keys"
    choice_id = value.get("choice_id")
    confidence = value.get("confidence")
    if (
        not isinstance(choice_id, str)
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return _fallback(), False, "invalid_types"
    by_id = {row["choice_id"]: row for row in choice_catalog["choices"]}
    if choice_id not in by_id:
        return _fallback(), False, "unknown_choice_id"
    choice = by_id[choice_id]
    prediction = {
        "status": choice["kind"], "known_intent": None,
        "novel_scenario": None, "confidence": float(confidence),
    }
    if choice["kind"] == "KNOWN":
        prediction["known_intent"] = choice["intent_id"]
    elif choice["kind"] == "NOVEL":
        prediction["novel_scenario"] = choice["scenario"]
    return prediction, True, "valid"


def evaluate_v109_gates(
    fixtures: dict[str, dict[str, Any]], evaluation_records: list[dict[str, Any]],
    controlled_ids: set[str], choice_catalog: dict[str, Any], access: dict[str, Any],
    baseline_config: dict[str, Any], config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, bool], dict[str, bool]]:
    metrics = aggregate_model_fixtures(fixtures, evaluation_records, controlled_ids, baseline_config)
    interface = config["interfaceGates"]
    semantic = config["semanticGates"]
    limits = config["accessGates"]
    choice_ids = [row["choice_id"] for row in choice_catalog["choices"]]
    interface_checks = {
        "structured_response_validity": metrics["structured_response_validity"] >= interface["minimumStructuredResponseValidity"],
        "observed_structured_response_validity": metrics["observed_structured_response_validity"] >= interface["minimumObservedStructuredResponseValidity"],
        "controlled_missing_observation_abstention_accuracy": metrics["controlled_missing_observation_abstention_accuracy"] >= interface["minimumControlledMissingObservationAbstentionAccuracy"],
        "required_choice_count": len(choice_ids) == interface["requiredChoiceCount"],
        "zero_ambiguous_choice_ids": len(choice_ids) - len(set(choice_ids)) <= interface["maximumAmbiguousChoiceIdCount"],
    }
    semantic_checks = {
        "observed_status_macro_f1": metrics["observed_status_macro_f1"] >= semantic["minimumObservedStatusMacroF1"],
        "observed_exact_decision_accuracy": metrics["observed_exact_decision_accuracy"] >= semantic["minimumObservedExactDecisionAccuracy"],
        "known_exact_intent_accuracy": metrics["known_exact_intent_accuracy"] >= semantic["minimumKnownExactIntentAccuracy"],
        "novel_status_recall": metrics["per_status"]["NOVEL"]["recall"] >= semantic["minimumNovelStatusRecall"],
        "novel_status_precision": metrics["per_status"]["NOVEL"]["precision"] >= semantic["minimumNovelStatusPrecision"],
        "novel_exact_scenario_accuracy": metrics["novel_exact_scenario_accuracy"] >= semantic["minimumNovelExactScenarioAccuracy"],
        "unsupported_status_recall": metrics["per_status"]["UNSUPPORTED"]["recall"] >= semantic["minimumUnsupportedStatusRecall"],
        "unsupported_status_precision": metrics["per_status"]["UNSUPPORTED"]["precision"] >= semantic["minimumUnsupportedStatusPrecision"],
        "false_known_acceptance_rate": metrics["false_known_acceptance_rate"] <= semantic["maximumFalseKnownAcceptanceRate"],
        "confidence_ece": metrics["confidence_ece_10_bin"] <= semantic["maximumConfidenceECE"],
        "top_confidence_80_percent_error": metrics["top_confidence_80_percent_error"] <= semantic["maximumTopConfidence80PercentError"],
        "mean_decision_regret": metrics["mean_regret"] <= semantic["maximumMeanDecisionRegret"],
        "regret_above_ask_always": metrics["mean_regret"] - 1.125 <= semantic["maximumRegretAboveAskAlways"],
    }
    access_checks = {
        "required_fixture_count": len(fixtures) == limits["requiredFixtureCount"],
        "development_language_read_budget": access["development_language_read_count"] <= limits["maximumDevelopmentLanguageReadCount"],
        "zero_protected_test_language_reads": access["protected_test_language_read_count"] <= limits["maximumProtectedTestLanguageReadCount"],
        "zero_manual_utterance_inspection": access["manual_utterance_inspection_count"] <= limits["maximumManualUtteranceInspectionCount"],
        "model_load_budget": access["model_load_count"] <= limits["maximumModelLoadCount"],
        "model_generation_budget": access["model_generation_count"] <= limits["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= limits["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= limits["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= limits["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= limits["maximumExternalSideEffectCount"],
    }
    return metrics, interface_checks, semantic_checks, access_checks


__all__ = [
    "compile_choice_catalog", "evaluate_v109_gates", "render_choice_prompt",
    "validate_and_expand_choice",
]
