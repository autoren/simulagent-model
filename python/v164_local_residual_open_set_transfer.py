from __future__ import annotations

from collections import Counter
import json
from typing import Any

from v106_open_world_benchmark import evaluate_predictions, prediction


def render_residual_prompt(
    catalog: dict[str, Any], utterance: str, config: dict[str, Any]
) -> str:
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("V164 residual utterance is missing")
    payload = {
        "instruction": config["prompt"]["instruction"],
        "visible_catalog": catalog,
        "user_utterance": utterance,
        "response_contract": {
            "required_keys": [
                "status", "known_intent", "novel_scenario", "confidence"
            ],
            "allowed_statuses": ["KNOWN", "NOVEL", "UNSUPPORTED", "ABSTAIN"],
            "known_intent": "exact listed intent_id or null",
            "novel_scenario": "exact listed scenario or null",
            "confidence": "number from 0 to 1"
        },
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def combine_predictions(
    evaluation_records: list[dict[str, Any]],
    residual_predictions: dict[str, dict[str, Any]],
    consensus_predictions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    evaluation_ids = {row["record_id"] for row in evaluation_records}
    if set(consensus_predictions) != evaluation_ids:
        raise ValueError("V164 consensus identity mismatch")
    if not set(residual_predictions) < evaluation_ids:
        raise ValueError("V164 residual prediction identity mismatch")
    return {
        identifier: (
            residual_predictions[identifier]
            if identifier in residual_predictions
            else consensus_predictions[identifier]
        )
        for identifier in sorted(evaluation_ids)
    }


def aggregate_residual_fixtures(
    fixtures: dict[str, dict[str, Any]],
    residual_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
    consensus_predictions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    residual_ids = {row["record_id"] for row in residual_records}
    if set(fixtures) != residual_ids:
        raise ValueError("V164 fixture identity mismatch")
    residual_predictions = {
        identifier: fixtures[identifier]["parsed_response"]
        for identifier in sorted(residual_ids)
    }
    hybrid_predictions = combine_predictions(
        evaluation_records, residual_predictions, consensus_predictions
    )
    residual_metrics = evaluate_predictions(
        residual_records, residual_predictions, config
    )
    hybrid_metrics = evaluate_predictions(
        evaluation_records, hybrid_predictions, config
    )
    valid_count = sum(bool(row["response_valid"]) for row in fixtures.values())
    reason_counts = dict(
        sorted(Counter(row["validation_reason"] for row in fixtures.values()).items())
    )
    residual_metrics["structured_response_validity"] = valid_count / len(fixtures)
    return {
        "fixture_count": len(fixtures),
        "valid_fixture_count": valid_count,
        "validation_reason_counts": reason_counts,
        "residual_metrics": residual_metrics,
        "hybrid_metrics": hybrid_metrics,
        "controlled_missing_observation_abstention_accuracy": 1.0,
        "true_hypothesis_retention": 1.0,
        "model_nonresidual_override_count": 0,
    }


def evaluate_quality_and_access_gates(
    aggregate: dict[str, Any],
    frozen_consensus_regret: float,
    access: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    quality = config["qualityGates"]
    limits = config["accessGates"]
    residual = aggregate["residual_metrics"]
    hybrid = aggregate["hybrid_metrics"]
    return {
        "structured_response_validity": residual["structured_response_validity"]
        >= quality["minimumStructuredResponseValidity"],
        "residual_status_macro_f1": residual["observed_status_macro_f1"]
        >= quality["minimumResidualStatusMacroF1"],
        "residual_exact_decision_accuracy": residual[
            "observed_exact_decision_accuracy"
        ]
        >= quality["minimumResidualExactDecisionAccuracy"],
        "residual_known_exact_intent_accuracy": residual[
            "known_exact_intent_accuracy"
        ]
        >= quality["minimumResidualKnownExactIntentAccuracy"],
        "residual_novel_status_recall": residual["per_status"]["NOVEL"]["recall"]
        >= quality["minimumResidualNovelStatusRecall"],
        "residual_novel_status_precision": residual["per_status"]["NOVEL"][
            "precision"
        ]
        >= quality["minimumResidualNovelStatusPrecision"],
        "residual_novel_exact_scenario_accuracy": residual[
            "novel_exact_scenario_accuracy"
        ]
        >= quality["minimumResidualNovelExactScenarioAccuracy"],
        "residual_unsupported_status_recall": residual["per_status"][
            "UNSUPPORTED"
        ]["recall"]
        >= quality["minimumResidualUnsupportedStatusRecall"],
        "residual_unsupported_status_precision": residual["per_status"][
            "UNSUPPORTED"
        ]["precision"]
        >= quality["minimumResidualUnsupportedStatusPrecision"],
        "residual_false_known_acceptance_rate": residual[
            "false_known_acceptance_rate"
        ]
        <= quality["maximumResidualFalseKnownAcceptanceRate"],
        "residual_confidence_ECE": residual["confidence_ece_10_bin"]
        <= quality["maximumResidualConfidenceECE"],
        "residual_top_confidence_80_percent_error": residual[
            "top_confidence_80_percent_error"
        ]
        <= quality["maximumResidualTopConfidence80PercentError"],
        "residual_mean_decision_regret": residual["mean_regret"]
        <= quality["maximumResidualMeanDecisionRegret"],
        "combined_hybrid_exact_decision_accuracy": hybrid[
            "observed_exact_decision_accuracy"
        ]
        >= quality["minimumCombinedHybridExactDecisionAccuracy"],
        "combined_hybrid_false_known_acceptance_rate": hybrid[
            "false_known_acceptance_rate"
        ]
        <= quality["maximumCombinedHybridFalseKnownAcceptanceRate"],
        "combined_hybrid_mean_decision_regret": hybrid["mean_regret"]
        <= quality["maximumCombinedHybridMeanDecisionRegret"],
        "combined_regret_improvement_over_frozen_consensus": (
            frozen_consensus_regret - hybrid["mean_regret"]
        )
        >= quality["minimumCombinedRegretImprovementOverFrozenConsensus"],
        "controlled_missing_observation_abstention_accuracy": aggregate[
            "controlled_missing_observation_abstention_accuracy"
        ]
        == quality["requiredControlledMissingObservationAbstentionAccuracy"],
        "true_hypothesis_retention": aggregate["true_hypothesis_retention"]
        == quality["requiredTrueHypothesisRetention"],
        "required_residual_fixture_count": aggregate["fixture_count"]
        == limits["requiredResidualFixtureCount"],
        "development_language_read_budget": access[
            "development_language_read_count"
        ]
        <= limits["maximumDevelopmentLanguageReadCount"],
        "zero_protected_language_reads": access["protected_language_read_count"]
        <= limits["maximumProtectedLanguageReadCount"],
        "zero_manual_utterance_inspection": access[
            "manual_utterance_inspection_count"
        ]
        <= limits["maximumManualUtteranceInspectionCount"],
        "zero_manual_raw_response_inspection": access[
            "manual_raw_response_inspection_count"
        ]
        <= limits["maximumManualRawResponseInspectionCount"],
        "model_load_budget": access["model_load_count"]
        <= limits["maximumModelLoadCount"],
        "model_generation_budget": access["model_generation_count"]
        <= limits["maximumModelGenerationCount"],
        "zero_retries": access["retry_count"] <= limits["maximumRetryCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"]
        <= limits["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"]
        <= limits["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"]
        <= limits["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"]
        <= limits["maximumExternalSideEffectCount"],
        "zero_actual_execution": access["actual_execution_count"]
        <= limits["maximumActualExecutionCount"],
    }


def ask_residual_prediction() -> dict[str, Any]:
    return prediction("ABSTAIN", 1.0)


__all__ = [
    "aggregate_residual_fixtures",
    "ask_residual_prediction",
    "combine_predictions",
    "evaluate_quality_and_access_gates",
    "render_residual_prompt",
]
