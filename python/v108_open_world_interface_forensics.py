from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any

from v105_open_world_interface import validate_response
from v106_open_world_benchmark import evaluate_predictions
from v107_open_world_local_model import aggregate_model_fixtures


def _validate_response_safely(
    response: str | dict[str, Any], catalog: dict[str, Any], interface_config: dict[str, Any]
) -> tuple[dict[str, Any], bool, str]:
    try:
        return validate_response(response, catalog, interface_config)
    except (TypeError, ValueError):
        fallback = dict(interface_config["responseContract"]["invalidResponseFallback"])
        return fallback, False, "invalid_response_type"


def local_intent_index(catalog: dict[str, Any]) -> tuple[dict[str, str], int]:
    candidates: dict[str, list[str]] = defaultdict(list)
    for row in catalog["intents"]:
        candidates[row["intent"]].append(row["intent_id"])
    unique = {name: values[0] for name, values in candidates.items() if len(values) == 1}
    ambiguous_count = sum(len(values) > 1 for values in candidates.values())
    return unique, ambiguous_count


def classify_response_shape(raw_response: str, catalog: dict[str, Any]) -> str:
    try:
        value = json.loads(raw_response)
    except json.JSONDecodeError:
        return "non_status_invariant_failure"
    if not isinstance(value, dict) or set(value) != {"status", "known_intent", "novel_scenario", "confidence"}:
        return "non_status_invariant_failure"
    status = value["status"]
    known = value["known_intent"]
    scenario = value["novel_scenario"]
    declared = {row["intent_id"] for row in catalog["intents"]}
    local, _ = local_intent_index(catalog)
    pair_scenario = {
        row["intent_id"]: row["scenario"] for row in catalog["intents"]
    }
    if status == "KNOWN":
        resolved = known if isinstance(known, str) and known in declared else (
            local.get(known) if isinstance(known, str) else None
        )
        if known is None:
            return "known_missing_intent"
        if isinstance(known, str) and known in local and scenario is None:
            return "known_local_name_only"
        if isinstance(known, str) and known in declared and scenario == pair_scenario[known]:
            return "known_qualified_with_redundant_matching_scenario"
        if isinstance(known, str) and known in local and resolved and scenario == pair_scenario[resolved]:
            return "known_local_name_with_redundant_matching_scenario"
        if not isinstance(known, str) or (known not in declared and known not in local):
            return "known_unknown_identifier"
        return "known_other_invariant"
    if status == "NOVEL":
        return "novel_other_invariant"
    if status in {"UNSUPPORTED", "ABSTAIN"} and (known is not None or scenario is not None):
        return "unsupported_or_abstain_nonnull_fields"
    return "non_status_invariant_failure"


def diagnostic_canonicalize(
    raw_response: str, catalog: dict[str, Any], interface_config: dict[str, Any]
) -> dict[str, Any]:
    original, original_valid, original_reason = _validate_response_safely(
        raw_response, catalog, interface_config,
    )
    try:
        value = json.loads(raw_response)
    except json.JSONDecodeError:
        return {
            "parsed_response": original, "valid": original_valid,
            "original_valid": original_valid, "original_reason": original_reason,
            "transforms": [],
        }
    if not isinstance(value, dict):
        return {
            "parsed_response": original, "valid": original_valid,
            "original_valid": original_valid, "original_reason": original_reason,
            "transforms": [],
        }
    candidate = dict(value)
    transforms = []
    local, _ = local_intent_index(catalog)
    if candidate.get("status") == "KNOWN":
        known = candidate.get("known_intent")
        if isinstance(known, str) and known in local:
            candidate["known_intent"] = local[known]
            transforms.append("unique_local_intent_to_qualified")
        resolved = candidate.get("known_intent")
        pair_scenario = {
            row["intent_id"]: row["scenario"] for row in catalog["intents"]
        }
        if (
            isinstance(resolved, str)
            and resolved in pair_scenario
            and candidate.get("novel_scenario") == pair_scenario[resolved]
        ):
            candidate["novel_scenario"] = None
            transforms.append("remove_redundant_matching_known_scenario")
    parsed, valid, _ = _validate_response_safely(candidate, catalog, interface_config)
    if not valid:
        parsed = original
    return {
        "parsed_response": parsed, "valid": valid,
        "original_valid": original_valid, "original_reason": original_reason,
        "transforms": transforms,
    }


def analyze_existing_outputs(
    fixtures: dict[str, dict[str, Any]], evaluation_records: list[dict[str, Any]],
    controlled_ids: set[str], catalog: dict[str, Any], interface_config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    expected = {row["record_id"] for row in evaluation_records} | controlled_ids
    if set(fixtures) != expected:
        raise ValueError("V108 input fixture identities mismatch")
    by_id = {row["record_id"]: row for row in evaluation_records}
    original_predictions = {
        identifier: fixtures[identifier]["parsed_response"] for identifier in by_id
    }
    counterfactual_predictions = {}
    category_counts = Counter()
    category_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    transform_counts = Counter()
    invalid_observed = 0
    canonicalized_invalid = 0
    for identifier, record in by_id.items():
        fixture = fixtures[identifier]
        category = "valid" if fixture["response_valid"] else classify_response_shape(fixture["raw_response"], catalog)
        category_counts[category] += 1
        category_by_class[record["class_label"]][category] += 1
        diagnostic = diagnostic_canonicalize(fixture["raw_response"], catalog, interface_config)
        counterfactual_predictions[identifier] = diagnostic["parsed_response"]
        transform_counts.update(diagnostic["transforms"])
        if not diagnostic["original_valid"]:
            invalid_observed += 1
            canonicalized_invalid += int(diagnostic["valid"])
    original_metrics = aggregate_model_fixtures(
        fixtures, evaluation_records, controlled_ids, baseline_config,
    )
    counterfactual_metrics = evaluate_predictions(evaluation_records, counterfactual_predictions, baseline_config)
    control_categories = Counter(
        "valid" if fixtures[identifier]["response_valid"]
        else classify_response_shape(fixtures[identifier]["raw_response"], catalog)
        for identifier in controlled_ids
    )
    _, ambiguous_count = local_intent_index(catalog)
    return {
        "fixture_counts": {
            "total": len(fixtures), "observed": len(evaluation_records),
            "controlled": len(controlled_ids), "invalid_observed": invalid_observed,
            "canonicalized_invalid_observed": canonicalized_invalid,
        },
        "invalid_observed_canonicalizable_fraction": (
            canonicalized_invalid / invalid_observed if invalid_observed else 0.0
        ),
        "ambiguous_local_intent_name_count": ambiguous_count,
        "observed_category_counts": dict(sorted(category_counts.items())),
        "observed_category_counts_by_class": {
            key: dict(sorted(value.items())) for key, value in sorted(category_by_class.items())
        },
        "control_category_counts": dict(sorted(control_categories.items())),
        "transform_counts": dict(sorted(transform_counts.items())),
        "original_metrics": original_metrics,
        "counterfactual_metrics": counterfactual_metrics,
        "raw_response_or_identifier_emission_count": 0,
    }


def evaluate_forensics_gates(
    analysis: dict[str, Any], original_metrics: dict[str, Any],
    access: dict[str, int], config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["formatDominanceGates"]
    counts = analysis["fixture_counts"]
    return {
        "original_fixture_count": counts["total"] == gates["requiredOriginalFixtureCount"],
        "original_observed_fixture_count": counts["observed"] == gates["requiredOriginalObservedFixtureCount"],
        "original_control_fixture_count": counts["controlled"] == gates["requiredOriginalControlFixtureCount"],
        "original_metric_reconstruction": analysis["original_metrics"] == original_metrics,
        "minimum_originally_invalid_observed_count": counts["invalid_observed"] >= gates["minimumOriginallyInvalidObservedCount"],
        "invalid_observed_canonicalizable_fraction": analysis["invalid_observed_canonicalizable_fraction"] >= gates["minimumInvalidObservedCanonicalizableFraction"],
        "counterfactual_known_exact_intent_accuracy": analysis["counterfactual_metrics"]["known_exact_intent_accuracy"] >= gates["minimumCounterfactualKnownExactIntentAccuracy"],
        "counterfactual_observed_exact_decision_accuracy": analysis["counterfactual_metrics"]["observed_exact_decision_accuracy"] >= gates["minimumCounterfactualObservedExactDecisionAccuracy"],
        "zero_ambiguous_local_intent_names": analysis["ambiguous_local_intent_name_count"] <= gates["maximumAmbiguousLocalIntentNameCount"],
        "original_V107_outcome_unchanged": gates["requiredOriginalV107OutcomeUnchanged"],
        "zero_development_language_reads": access["development_language_read_count"] <= gates["maximumDevelopmentLanguageReadCount"],
        "zero_protected_test_language_reads": access["protected_test_language_read_count"] <= gates["maximumProtectedTestLanguageReadCount"],
        "zero_manual_raw_response_inspection": access["manual_raw_response_inspection_count"] <= gates["maximumManualRawResponseInspectionCount"],
        "zero_model_loads": access["model_load_count"] <= gates["maximumModelLoadCount"],
        "zero_model_generations": access["model_generation_count"] <= gates["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= gates["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= gates["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= gates["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"],
    }


def aggregate_only_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    output = dict(analysis)
    for key in ("original_metrics", "counterfactual_metrics"):
        output[key] = {
            metric: value for metric, value in analysis[key].items()
            if metric != "scored_rows"
        }
    return output


__all__ = [
    "aggregate_only_analysis", "analyze_existing_outputs", "classify_response_shape", "diagnostic_canonicalize",
    "evaluate_forensics_gates", "local_intent_index",
]
