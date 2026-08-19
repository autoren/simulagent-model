from __future__ import annotations

from typing import Any

from v106_open_world_benchmark import (
    ask_always_prediction, evaluate_predictions, oracle_prediction, retrieval_prediction,
)
from v112_open_world_full_policy_transfer import (
    novelty_evidence_metrics, policy_prediction, policy_quality_gates,
)


def aggregate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "scored_rows"}


def evaluate_preserved_outputs(
    records: list[dict[str, Any]], fixtures: dict[str, dict[str, Any]],
    retrieval: dict[str, dict[str, Any]], access: dict[str, Any],
    config: dict[str, Any], baseline_config: dict[str, Any],
) -> dict[str, Any]:
    observed = {row["record_id"]: fixtures[row["record_id"]] for row in records}
    controlled = [row for row in fixtures.values() if row["kind"] == "controlled_missing_observation"]
    direct = {identifier: row["parsed_response"] for identifier, row in observed.items()}
    policy_actions: dict[str, dict[str, Any]] = {}
    policy_evidence: dict[str, dict[str, Any]] = {}
    for row in records:
        identifier = row["record_id"]
        action, evidence = policy_prediction(direct[identifier], retrieval[identifier], config)
        policy_actions[identifier] = action
        policy_evidence[identifier] = evidence
    fixed = config["fixedRetrievalThresholds"]
    comparator_predictions = {
        "ask_always": {row["record_id"]: ask_always_prediction(row) for row in records},
        "direct_llm": direct,
        "fixed_v110_character_retrieval": {
            row["record_id"]: retrieval_prediction(
                retrieval[row["record_id"]], fixed["known"], fixed["unsupported"],
            ) for row in records
        },
        "validated_novelty_evidence_policy": policy_actions,
        "oracle": {row["record_id"]: oracle_prediction(row) for row in records},
    }
    metrics = {
        name: aggregate_metrics(evaluate_predictions(records, predictions, baseline_config))
        for name, predictions in comparator_predictions.items()
    }
    novel_metrics = novelty_evidence_metrics(records, policy_evidence)
    validity = sum(row["response_valid"] for row in fixtures.values()) / len(fixtures)
    controlled_accuracy = sum(
        row["response_valid"] and row["parsed_response"]["status"] == "ABSTAIN"
        for row in controlled
    ) / len(controlled)
    quality = policy_quality_gates(
        validity, controlled_accuracy, novel_metrics,
        metrics["validated_novelty_evidence_policy"], 0, 1.0, config,
    )
    limits = config["accessGates"]
    access_gates = {
        "fresh_development_language_read_budget": access["fresh_development_language_read_count"] <= limits["maximumFreshDevelopmentLanguageReadCount"],
        "zero_protected_test_language_reads": access["protected_test_language_read_count"] <= limits["maximumProtectedTestLanguageReadCount"],
        "zero_manual_language_or_raw_response_inspection": access["manual_language_or_raw_response_inspection_count"] <= limits["maximumManualLanguageOrRawResponseInspectionCount"],
        "model_load_budget": access["model_load_count"] <= limits["maximumModelLoadCount"],
        "model_generation_budget": access["model_generation_count"] <= limits["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= limits["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= limits["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= limits["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= limits["maximumExternalSideEffectCount"],
    }
    return {
        "interface_validity": validity,
        "controlled_missing_observation_abstention_accuracy": controlled_accuracy,
        "novel_evidence_metrics": novel_metrics, "policy_metrics": metrics,
        "quality_gates": quality, "access_gates": access_gates,
        "actual_execution_count": 0, "true_hypothesis_retention": 1.0,
        "individual_evidence_emission_count": 0,
    }


__all__ = ["evaluate_preserved_outputs"]
