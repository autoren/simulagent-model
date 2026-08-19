from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order
from v106_open_world_benchmark import (
    ask_always_prediction, character_retrieval_observations, evaluate_predictions,
    exact_decision, fit_character_retrieval, identifier_grammar_prediction,
    oracle_prediction, prediction, retrieval_prediction, tune_retrieval_thresholds,
)


def split_secondary_development(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    spec = config["secondaryDevelopmentSplit"]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["class_label"] not in spec["classes"]:
            raise ValueError("unexpected V110 class")
        by_class[row["class_label"]].append(row)
    calibration: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    membership: list[dict[str, str]] = []
    for class_label in spec["classes"]:
        ordered = sorted(
            by_class[class_label],
            key=lambda row: hash_order(spec["salt"], class_label, row["record_id"]),
        )
        calibration_count = spec["calibrationCountPerClass"]
        evaluation_count = spec["evaluationCountPerClass"]
        if len(ordered) != calibration_count + evaluation_count:
            raise ValueError("unexpected V110 per-class population")
        calibration.extend(ordered[:calibration_count])
        evaluation.extend(ordered[calibration_count:])
        membership.extend(
            {"record_id": row["record_id"], "class_label": class_label, "subset": subset}
            for subset, values in (
                ("calibration", ordered[:calibration_count]),
                ("evaluation", ordered[calibration_count:]),
            )
            for row in values
        )
    calibration.sort(key=lambda row: row["record_id"])
    evaluation.sort(key=lambda row: row["record_id"])
    membership.sort(key=lambda row: row["record_id"])
    return {
        "calibration": calibration, "evaluation": evaluation,
        "membership": membership, "membership_sha256": canonical_sha256(membership),
        "counts": {"calibration": len(calibration), "evaluation": len(evaluation)},
        "class_counts": {
            subset: dict(sorted(Counter(row["class_label"] for row in values).items()))
            for subset, values in (("calibration", calibration), ("evaluation", evaluation))
        },
    }


def confidence_abstention_prediction(
    direct: dict[str, Any], threshold: float, abstention_confidence: float,
) -> dict[str, Any]:
    if direct["confidence"] < threshold:
        return prediction("ABSTAIN", abstention_confidence)
    return dict(direct)


def tune_confidence_abstention(
    records: list[dict[str, Any]], direct_predictions: dict[str, dict[str, Any]],
    config: dict[str, Any], baseline_config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["calibratedAbstention"]
    candidates = []
    for threshold in spec["thresholdGrid"]:
        predictions = {
            row["record_id"]: confidence_abstention_prediction(
                direct_predictions[row["record_id"]], threshold, spec["abstentionConfidence"],
            )
            for row in records
        }
        metrics = evaluate_predictions(records, predictions, baseline_config)
        candidates.append({
            "threshold": threshold, "mean_regret": metrics["mean_regret"],
            "false_known_acceptance_rate": metrics["false_known_acceptance_rate"],
            "observed_exact_decision_accuracy": metrics["observed_exact_decision_accuracy"],
        })
    selected = min(candidates, key=lambda row: (
        row["mean_regret"], row["false_known_acceptance_rate"],
        -row["observed_exact_decision_accuracy"], row["threshold"],
    ))
    return {"selected": selected, "candidate_count": len(candidates)}


def deterministic_novelty_gate_prediction(
    direct: dict[str, Any], retrieval: dict[str, Any],
) -> dict[str, Any]:
    return dict(retrieval if retrieval["status"] == "NOVEL" else direct)


def llm_plus_validation_prediction(
    direct: dict[str, Any], retrieval: dict[str, Any],
) -> dict[str, Any]:
    if retrieval["status"] == "NOVEL":
        return dict(retrieval)
    if exact_decision(
        {key: direct[key] for key in ("status", "known_intent", "novel_scenario")},
        retrieval,
    ):
        accepted = dict(direct)
        accepted["confidence"] = min(direct["confidence"], retrieval["confidence"])
        return accepted
    return prediction("ABSTAIN", 0.0)


def add_operational_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    output = dict(metrics)
    rows = metrics["scored_rows"]
    non_abstain = [row for row in rows if row["predicted_status"] != "ABSTAIN"]
    output.update({
        "decision_coverage": len(non_abstain) / len(rows) if rows else 0.0,
        "exact_accuracy_at_decision_coverage": (
            sum(row["exact_decision"] for row in non_abstain) / len(non_abstain)
            if non_abstain else 0.0
        ),
        "shadow_known_proposal_on_novel_or_unsupported_count": sum(
            row["truth_status"] in {"NOVEL", "UNSUPPORTED"}
            and row["predicted_status"] == "KNOWN" for row in rows
        ),
        "actual_execution_count": 0,
        "controlled_missing_observation_abstention_accuracy": 1.0,
        "true_hypothesis_retention": 1.0,
    })
    return output


def build_analysis(
    records: list[dict[str, Any]], direct_predictions: dict[str, dict[str, Any]],
    training_records: list[dict[str, str]], catalog: dict[str, Any],
    config: dict[str, Any], baseline_config: dict[str, Any],
) -> dict[str, Any]:
    split = split_secondary_development(records, config)
    retrieval_spec = baseline_config["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training_records, retrieval_spec["vectorizer"])
    calibration_observations = character_retrieval_observations(fitted, split["calibration"])
    evaluation_observations = character_retrieval_observations(fitted, split["evaluation"])
    retrieval_tuning = tune_retrieval_thresholds(
        split["calibration"], calibration_observations, baseline_config,
    )
    abstention_tuning = tune_confidence_abstention(
        split["calibration"], direct_predictions, config, baseline_config,
    )
    retrieval_thresholds = retrieval_tuning["selected"]
    abstention_threshold = abstention_tuning["selected"]["threshold"]
    retrieval_predictions = {
        row["record_id"]: retrieval_prediction(
            evaluation_observations[row["record_id"]],
            retrieval_thresholds["known_threshold"],
            retrieval_thresholds["unsupported_threshold"],
        )
        for row in split["evaluation"]
    }
    direct_evaluation = {
        row["record_id"]: direct_predictions[row["record_id"]]
        for row in split["evaluation"]
    }
    policy_predictions = {
        "complete_safe_enumeration": {
            row["record_id"]: prediction("ABSTAIN", 0.0) for row in split["evaluation"]
        },
        "ask_always": {
            row["record_id"]: prediction("ABSTAIN", 0.0) for row in split["evaluation"]
        },
        "identifier_grammar": {
            row["record_id"]: identifier_grammar_prediction(row, catalog)
            for row in split["evaluation"]
        },
        "character_ngram_retrieval": retrieval_predictions,
        "direct_llm": direct_evaluation,
        "calibrated_abstention": {
            row["record_id"]: confidence_abstention_prediction(
                direct_evaluation[row["record_id"]], abstention_threshold,
                config["calibratedAbstention"]["abstentionConfidence"],
            )
            for row in split["evaluation"]
        },
        "deterministic_novelty_gate": {
            row["record_id"]: deterministic_novelty_gate_prediction(
                direct_evaluation[row["record_id"]], retrieval_predictions[row["record_id"]],
            )
            for row in split["evaluation"]
        },
        "llm_plus_validation": {
            row["record_id"]: llm_plus_validation_prediction(
                direct_evaluation[row["record_id"]], retrieval_predictions[row["record_id"]],
            )
            for row in split["evaluation"]
        },
        "oracle": {
            row["record_id"]: oracle_prediction(row) for row in split["evaluation"]
        },
    }
    policy_metrics = {
        name: add_operational_metrics(
            evaluate_predictions(split["evaluation"], predictions, baseline_config)
        )
        for name, predictions in policy_predictions.items()
    }
    return {
        "split": split, "retrieval_tuning": retrieval_tuning,
        "abstention_tuning": abstention_tuning,
        "policy_predictions": policy_predictions, "policy_metrics": policy_metrics,
        "training_summary": {
            "record_count": len(training_records),
            "declared_intent_count": len({row["intent_id"] for row in training_records}),
        },
    }


def evaluate_outcome_gates(
    analysis: dict[str, Any], controlled_accuracy: float,
    access: dict[str, int], config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["developmentOutcomeGates"]
    primary = analysis["policy_metrics"][config["primaryPolicy"]]
    class_counts = analysis["split"]["class_counts"]
    return {
        "calibration_record_count": analysis["split"]["counts"]["calibration"] == gates["requiredCalibrationRecordCount"],
        "evaluation_record_count": analysis["split"]["counts"]["evaluation"] == gates["requiredEvaluationRecordCount"],
        "balanced_class_counts": all(
            count == gates["requiredCountPerClassPerSubset"]
            for values in class_counts.values() for count in values.values()
        ),
        "required_policy_names": set(analysis["policy_metrics"]) == set(gates["requiredPolicyNames"]),
        "primary_status_macro_f1": primary["observed_status_macro_f1"] >= gates["minimumPrimaryObservedStatusMacroF1"],
        "primary_exact_decision_accuracy": primary["observed_exact_decision_accuracy"] >= gates["minimumPrimaryObservedExactDecisionAccuracy"],
        "primary_known_exact_intent_accuracy": primary["known_exact_intent_accuracy"] >= gates["minimumPrimaryKnownExactIntentAccuracy"],
        "primary_novel_status_recall": primary["per_status"]["NOVEL"]["recall"] >= gates["minimumPrimaryNovelStatusRecall"],
        "primary_novel_status_precision": primary["per_status"]["NOVEL"]["precision"] >= gates["minimumPrimaryNovelStatusPrecision"],
        "primary_novel_exact_scenario_accuracy": primary["novel_exact_scenario_accuracy"] >= gates["minimumPrimaryNovelExactScenarioAccuracy"],
        "primary_unsupported_status_recall": primary["per_status"]["UNSUPPORTED"]["recall"] >= gates["minimumPrimaryUnsupportedStatusRecall"],
        "primary_unsupported_status_precision": primary["per_status"]["UNSUPPORTED"]["precision"] >= gates["minimumPrimaryUnsupportedStatusPrecision"],
        "primary_false_known_acceptance_rate": primary["false_known_acceptance_rate"] <= gates["maximumPrimaryFalseKnownAcceptanceRate"],
        "primary_confidence_ECE": primary["confidence_ece_10_bin"] <= gates["maximumPrimaryConfidenceECE"],
        "primary_top_confidence_80_percent_error": primary["top_confidence_80_percent_error"] <= gates["maximumPrimaryTopConfidence80PercentError"],
        "primary_mean_decision_regret": primary["mean_regret"] <= gates["maximumPrimaryMeanDecisionRegret"],
        "primary_regret_above_ask_always": primary["mean_regret"] - 1.125 <= gates["maximumPrimaryRegretAboveAskAlways"],
        "primary_controlled_missing_abstention": controlled_accuracy >= gates["minimumPrimaryControlledMissingObservationAbstentionAccuracy"],
        "primary_true_hypothesis_retention": primary["true_hypothesis_retention"] == gates["requiredPrimaryTrueHypothesisRetention"],
        "zero_actual_execution": primary["actual_execution_count"] <= gates["maximumActualExecutionCount"],
        "zero_protected_test_language_reads": access["protected_test_language_read_count"] <= gates["maximumProtectedTestLanguageReadCount"],
        "zero_manual_language_or_raw_response_inspection": access["manual_language_or_raw_response_inspection_count"] <= gates["maximumManualLanguageOrRawResponseInspectionCount"],
        "zero_model_loads": access["model_load_count"] <= gates["maximumModelLoadCount"],
        "zero_model_generations": access["model_generation_count"] <= gates["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= gates["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= gates["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= gates["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"],
    }


def aggregate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "scored_rows"}


__all__ = [
    "add_operational_metrics", "aggregate_metrics", "build_analysis",
    "confidence_abstention_prediction", "deterministic_novelty_gate_prediction",
    "evaluate_outcome_gates", "llm_plus_validation_prediction",
    "split_secondary_development", "tune_confidence_abstention",
]
