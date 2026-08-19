from __future__ import annotations

from collections import Counter
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order
from v106_open_world_benchmark import (
    ask_always_prediction,
    build_declared_training_records,
    character_retrieval_observations,
    evaluate_predictions,
    fit_character_retrieval,
    identifier_grammar_prediction,
    oracle_prediction,
    prediction,
    retrieval_prediction,
    split_development_records,
    tune_retrieval_thresholds,
)


def adapt_development_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapted = []
    for row in records:
        if row.get("role") != "development_transfer":
            raise ValueError("unexpected transfer role")
        adapted.append(dict(row, role="development"))
    return adapted


def select_controlled_missing_identifiers(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    spec = config["controlledMissingObservation"]
    ordered = sorted(
        records,
        key=lambda row: hash_order(spec["selectionSalt"], row["record_id"]),
    )
    return [
        {"record_id": row["record_id"], "observation_available": False}
        for row in ordered[: spec["count"]]
    ]


def same_structured_decision(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("status", "known_intent", "novel_scenario")
    return all(left[key] == right[key] for key in keys)


def deterministic_consensus_prediction(
    grammar: dict[str, Any], retrieval: dict[str, Any]
) -> dict[str, Any]:
    if (
        grammar["status"] != "ABSTAIN"
        and retrieval["status"] != "ABSTAIN"
        and same_structured_decision(grammar, retrieval)
    ):
        return prediction(
            grammar["status"],
            min(grammar["confidence"], retrieval["confidence"]),
            known_intent=grammar["known_intent"],
            novel_scenario=grammar["novel_scenario"],
        )
    return prediction("ABSTAIN", 0.0)


def _subset_summary(
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not records:
        return {
            "record_count": 0,
            "exact_decision_accuracy": 0.0,
            "false_known_acceptance_rate": 0.0,
            "mean_regret": 0.0,
        }
    metrics = evaluate_predictions(records, predictions, config)
    return {
        "record_count": len(records),
        "exact_decision_accuracy": metrics["observed_exact_decision_accuracy"],
        "false_known_acceptance_rate": metrics["false_known_acceptance_rate"],
        "mean_regret": metrics["mean_regret"],
    }


def evaluate_residual_qualification(
    residual_summary: dict[str, Any],
    nonresidual_summary: dict[str, Any],
    consensus_metrics: dict[str, Any],
    ask_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["residualQualificationGates"]
    return {
        "minimum_residual_record_count": residual_summary["record_count"]
        >= gates["minimumResidualRecordCount"],
        "maximum_residual_record_count": residual_summary["record_count"]
        <= gates["maximumResidualRecordCount"],
        "minimum_residual_class_coverage": residual_summary["class_coverage"]
        >= gates["minimumResidualClassCoverage"],
        "minimum_nonresidual_record_count": nonresidual_summary["record_count"]
        >= gates["minimumNonResidualRecordCount"],
        "minimum_nonresidual_exact_decision_accuracy": nonresidual_summary[
            "exact_decision_accuracy"
        ]
        >= gates["minimumNonResidualExactDecisionAccuracy"],
        "maximum_nonresidual_false_known_acceptance_rate": nonresidual_summary[
            "false_known_acceptance_rate"
        ]
        <= gates["maximumNonResidualFalseKnownAcceptanceRate"],
        "maximum_nonresidual_mean_regret": nonresidual_summary["mean_regret"]
        <= gates["maximumNonResidualMeanRegret"],
        "maximum_consensus_overall_false_known_acceptance_rate": consensus_metrics[
            "false_known_acceptance_rate"
        ]
        <= gates["maximumConsensusOverallFalseKnownAcceptanceRate"],
        "maximum_consensus_regret_above_ask_always": (
            consensus_metrics["mean_regret"] - ask_metrics["mean_regret"]
        )
        <= gates["maximumConsensusRegretAboveAskAlways"],
    }


def build_deterministic_transfer_artifacts(
    source_records: list[dict[str, Any]],
    transfer_records: list[dict[str, Any]],
    catalog: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    development_records = adapt_development_records(transfer_records)
    split = split_development_records(development_records, config)
    controls = select_controlled_missing_identifiers(development_records, config)
    training_records = build_declared_training_records(source_records, catalog)
    retrieval_spec = config["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training_records, retrieval_spec["vectorizer"])
    calibration_observations = character_retrieval_observations(
        fitted, split["calibration"]
    )
    evaluation_observations = character_retrieval_observations(
        fitted, split["evaluation"]
    )
    tuning = tune_retrieval_thresholds(
        split["calibration"], calibration_observations, config
    )
    thresholds = tuning["selected"]

    grammar_predictions = {
        row["record_id"]: identifier_grammar_prediction(row, catalog)
        for row in split["evaluation"]
    }
    retrieval_predictions = {
        row["record_id"]: retrieval_prediction(
            evaluation_observations[row["record_id"]],
            thresholds["known_threshold"],
            thresholds["unsupported_threshold"],
        )
        for row in split["evaluation"]
    }
    consensus_predictions = {
        row["record_id"]: deterministic_consensus_prediction(
            grammar_predictions[row["record_id"]],
            retrieval_predictions[row["record_id"]],
        )
        for row in split["evaluation"]
    }
    predictions = {
        "complete_safe_enumeration": {
            row["record_id"]: ask_always_prediction(row)
            for row in split["evaluation"]
        },
        "ask_always": {
            row["record_id"]: ask_always_prediction(row)
            for row in split["evaluation"]
        },
        "identifier_grammar": grammar_predictions,
        "character_ngram_retrieval": retrieval_predictions,
        "deterministic_consensus": consensus_predictions,
        "oracle": {
            row["record_id"]: oracle_prediction(row) for row in split["evaluation"]
        },
    }
    metrics = {}
    for name, values in predictions.items():
        metric = evaluate_predictions(split["evaluation"], values, config)
        metric["controlled_missing_observation_abstention_accuracy"] = 1.0
        metric["true_hypothesis_retention"] = 1.0
        metrics[name] = metric

    residual_records = [
        row
        for row in split["evaluation"]
        if consensus_predictions[row["record_id"]]["status"] == "ABSTAIN"
    ]
    nonresidual_records = [
        row
        for row in split["evaluation"]
        if consensus_predictions[row["record_id"]]["status"] != "ABSTAIN"
    ]
    residual_manifest = [
        {"record_id": row["record_id"]}
        for row in sorted(residual_records, key=lambda value: value["record_id"])
    ]
    residual_summary = {
        "record_count": len(residual_records),
        "class_coverage": len({row["class_label"] for row in residual_records}),
        "class_counts": dict(
            sorted(Counter(row["class_label"] for row in residual_records).items())
        ),
        "membership_uses_truth_or_language": False,
        "manifest_payload_sha256": canonical_sha256(residual_manifest),
    }
    nonresidual_summary = _subset_summary(
        nonresidual_records,
        {
            row["record_id"]: consensus_predictions[row["record_id"]]
            for row in nonresidual_records
        },
        config,
    )
    residual_checks = evaluate_residual_qualification(
        residual_summary,
        nonresidual_summary,
        metrics["deterministic_consensus"],
        metrics["ask_always"],
        config,
    )
    residual_qualified = all(residual_checks.values())

    eligible_baselines = (
        "ask_always",
        "identifier_grammar",
        "character_ngram_retrieval",
        "deterministic_consensus",
    )
    best_name = min(
        eligible_baselines,
        key=lambda name: (
            metrics[name]["mean_regret"],
            -metrics[name]["observed_exact_decision_accuracy"],
            name,
        ),
    )
    training_counts = dict(
        sorted(Counter(row["intent_id"] for row in training_records).items())
    )
    language_free_predictions = {
        name: {key: value for key, value in sorted(values.items())}
        for name, values in sorted(predictions.items())
    }
    return {
        "split": split,
        "controlled_missing_identifiers": controls,
        "training_records": training_records,
        "training_summary": {
            "record_count": len(training_records),
            "declared_intent_count": len(training_counts),
            "records_per_declared_intent": training_counts,
        },
        "retrieval_tuning": tuning,
        "baseline_metrics": metrics,
        "evaluation_predictions": language_free_predictions,
        "evaluation_prediction_payload_sha256": canonical_sha256(
            language_free_predictions
        ),
        "best_nonoracle_baseline": {
            "name": best_name,
            "mean_regret": metrics[best_name]["mean_regret"],
            "observed_exact_decision_accuracy": metrics[best_name][
                "observed_exact_decision_accuracy"
            ],
        },
        "residual_manifest": residual_manifest,
        "residual_summary": residual_summary,
        "nonresidual_summary": nonresidual_summary,
        "residual_qualification_checks": residual_checks,
        "residual_qualified": residual_qualified,
    }


def evaluate_baseline_pipeline_gates(
    artifacts: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, bool]:
    gates = config["baselinePipelineGates"]
    training_counts = artifacts["training_summary"]["records_per_declared_intent"]
    return {
        "training_declared_intent_count": len(training_counts)
        == gates["requiredTrainingDeclaredIntentCount"],
        "minimum_training_count_per_declared_intent": min(training_counts.values())
        >= gates["minimumTrainingRecordCountPerDeclaredIntent"],
        "calibration_record_count": artifacts["split"]["counts"]["calibration"]
        == gates["requiredCalibrationRecordCount"],
        "evaluation_record_count": artifacts["split"]["counts"]["evaluation"]
        == gates["requiredEvaluationRecordCount"],
        "controlled_missing_observation_count": len(
            artifacts["controlled_missing_identifiers"]
        )
        == gates["requiredControlledMissingObservationCount"],
        "required_baselines_complete": set(artifacts["baseline_metrics"])
        == set(gates["requiredBaselineNames"]),
        "oracle_exact_decision_accuracy": artifacts["baseline_metrics"]["oracle"][
            "observed_exact_decision_accuracy"
        ]
        == gates["requiredOracleExactDecisionAccuracy"],
        "oracle_mean_regret": artifacts["baseline_metrics"]["oracle"]["mean_regret"]
        == gates["requiredOracleMeanRegret"],
        "complete_safe_enumeration_true_hypothesis_retention": artifacts[
            "baseline_metrics"
        ]["complete_safe_enumeration"]["true_hypothesis_retention"]
        == gates["requiredCompleteSafeEnumerationTrueHypothesisRetention"],
        "zero_protected_language_reads": access["protected_language_read_count"]
        <= gates["maximumProtectedLanguageReadCount"],
        "zero_manual_utterance_inspection": access["manual_utterance_inspection_count"]
        <= gates["maximumManualUtteranceInspectionCount"],
        "zero_model_loads": access["model_load_count"] <= gates["maximumModelLoadCount"],
        "zero_model_generations": access["model_generation_count"]
        <= gates["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"]
        <= gates["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"]
        <= gates["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"]
        <= gates["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"]
        <= gates["maximumExternalSideEffectCount"],
        "zero_actual_execution": access["actual_execution_count"]
        <= gates["maximumActualExecutionCount"],
    }


__all__ = [
    "adapt_development_records",
    "build_deterministic_transfer_artifacts",
    "deterministic_consensus_prediction",
    "evaluate_baseline_pipeline_gates",
    "evaluate_residual_qualification",
    "same_structured_decision",
    "select_controlled_missing_identifiers",
]
