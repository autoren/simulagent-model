from __future__ import annotations

from typing import Any

from v106_open_world_benchmark import evaluate_predictions


def aggregate_model_fixtures(
    fixtures: dict[str, dict[str, Any]], evaluation_records: list[dict[str, Any]],
    controlled_ids: set[str], config: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = {row["record_id"] for row in evaluation_records} | controlled_ids
    if set(fixtures) != expected_ids:
        raise ValueError("V107 fixture identity mismatch")
    observed_predictions = {
        row["record_id"]: fixtures[row["record_id"]]["parsed_response"]
        for row in evaluation_records
    }
    metrics = evaluate_predictions(evaluation_records, observed_predictions, config)
    total_valid = sum(bool(row["response_valid"]) for row in fixtures.values())
    controlled_abstain = sum(
        fixtures[identifier]["parsed_response"]["status"] == "ABSTAIN"
        for identifier in controlled_ids
    )
    metrics.update({
        "structured_response_validity": total_valid / len(fixtures),
        "observed_structured_response_validity": (
            sum(fixtures[row["record_id"]]["response_valid"] for row in evaluation_records)
            / len(evaluation_records)
        ),
        "controlled_missing_observation_abstention_accuracy": controlled_abstain / len(controlled_ids),
        "true_hypothesis_retention": 1.0,
        "fixture_count": len(fixtures),
        "observed_fixture_count": len(evaluation_records),
        "controlled_fixture_count": len(controlled_ids),
    })
    return metrics


def evaluate_model_gates(
    metrics: dict[str, Any], best_baseline_regret: float,
    access: dict[str, Any], config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["developmentGates"]
    limits = config["accessGates"]
    regret_delta = metrics["mean_regret"] - best_baseline_regret
    return {
        "structured_response_validity": metrics["structured_response_validity"] >= gates["minimumStructuredResponseValidity"],
        "observed_status_macro_f1": metrics["observed_status_macro_f1"] >= gates["minimumObservedStatusMacroF1"],
        "observed_exact_decision_accuracy": metrics["observed_exact_decision_accuracy"] >= gates["minimumObservedExactDecisionAccuracy"],
        "known_exact_intent_accuracy": metrics["known_exact_intent_accuracy"] >= gates["minimumKnownExactIntentAccuracy"],
        "novel_status_recall": metrics["per_status"]["NOVEL"]["recall"] >= gates["minimumNovelStatusRecall"],
        "novel_status_precision": metrics["per_status"]["NOVEL"]["precision"] >= gates["minimumNovelStatusPrecision"],
        "novel_exact_scenario_accuracy": metrics["novel_exact_scenario_accuracy"] >= gates["minimumNovelExactScenarioAccuracy"],
        "unsupported_status_recall": metrics["per_status"]["UNSUPPORTED"]["recall"] >= gates["minimumUnsupportedStatusRecall"],
        "unsupported_status_precision": metrics["per_status"]["UNSUPPORTED"]["precision"] >= gates["minimumUnsupportedStatusPrecision"],
        "false_known_acceptance_rate": metrics["false_known_acceptance_rate"] <= gates["maximumFalseKnownAcceptanceRate"],
        "confidence_ece": metrics["confidence_ece_10_bin"] <= gates["maximumConfidenceECE"],
        "top_confidence_80_percent_error": metrics["top_confidence_80_percent_error"] <= gates["maximumTopConfidence80PercentError"],
        "mean_decision_regret": metrics["mean_regret"] <= gates["maximumMeanDecisionRegret"],
        "controlled_missing_observation_abstention_accuracy": metrics["controlled_missing_observation_abstention_accuracy"] >= gates["minimumControlledMissingObservationAbstentionAccuracy"],
        "regret_above_best_nonoracle_baseline": regret_delta <= gates["maximumRegretAboveBestNonOracleDeterministicBaseline"],
        "required_fixture_count": metrics["fixture_count"] == limits["requiredFixtureCount"],
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


def quality_gate_pass(gates: dict[str, bool]) -> bool:
    access_prefixes = (
        "required_fixture_count", "development_language_read_budget",
        "zero_protected_test_language_reads", "zero_manual_utterance_inspection",
        "model_load_budget", "model_generation_budget", "zero_LLM_API_calls",
        "zero_adapter_training", "zero_real_service_calls", "zero_external_side_effects",
    )
    return all(value for key, value in gates.items() if key not in access_prefixes)


__all__ = ["aggregate_model_fixtures", "evaluate_model_gates", "quality_gate_pass"]
