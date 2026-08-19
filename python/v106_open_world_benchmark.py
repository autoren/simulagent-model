from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from v93_open_set_source import canonical_sha256, hash_order, normalized_tokens


OBSERVED_STATUSES = ("KNOWN", "NOVEL", "UNSUPPORTED")


def split_development_records(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    spec = config["developmentSplit"]
    classes = tuple(spec["classes"])
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("role") != "development" or row.get("class_label") not in classes:
            raise ValueError("unexpected development record")
        by_class[row["class_label"]].append(row)
    calibration: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    membership: list[dict[str, str]] = []
    for class_label in classes:
        ordered = sorted(
            by_class[class_label],
            key=lambda row: hash_order(spec["salt"], class_label, row["record_id"]),
        )
        calibration_count = spec["calibrationCountPerClass"]
        evaluation_count = spec["evaluationCountPerClass"]
        if len(ordered) != calibration_count + evaluation_count:
            raise ValueError(f"unexpected class count for {class_label}")
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
        "calibration": calibration,
        "evaluation": evaluation,
        "membership": membership,
        "membership_sha256": canonical_sha256(membership),
        "counts": {"calibration": len(calibration), "evaluation": len(evaluation)},
        "class_counts": {
            subset: dict(sorted(Counter(row["class_label"] for row in values).items()))
            for subset, values in (("calibration", calibration), ("evaluation", evaluation))
        },
    }


def build_declared_training_records(
    source_records: list[dict[str, Any]], catalog: dict[str, Any]
) -> list[dict[str, str]]:
    declared = {row["intent_id"] for row in catalog["intents"]}
    selected = []
    for row in source_records:
        pair = f"{row['scenario']}::{row['intent']}"
        if row["partition"] == "train" and pair in declared:
            selected.append({
                "source_id": str(row["id"]),
                "intent_id": pair,
                "scenario": row["scenario"],
                "utterance": row["utt"],
            })
    selected.sort(key=lambda row: (row["intent_id"], row["source_id"]))
    if {row["intent_id"] for row in selected} != declared:
        raise ValueError("declared training intent coverage mismatch")
    return selected


def truth_for_record(record: dict[str, Any]) -> dict[str, Any]:
    label = record["class_label"]
    if label in {"known_familiar", "known_unfamiliar"}:
        return {
            "status": "KNOWN",
            "known_intent": f"{record['scenario']}::{record['intent']}",
            "novel_scenario": None,
        }
    if label == "novel_valid":
        return {"status": "NOVEL", "known_intent": None, "novel_scenario": record["scenario"]}
    if label == "unsupported":
        return {"status": "UNSUPPORTED", "known_intent": None, "novel_scenario": None}
    if label == "controlled_missing_observation":
        return {"status": "ABSTAIN", "known_intent": None, "novel_scenario": None}
    raise ValueError(f"unknown truth class: {label}")


def prediction(
    status: str, confidence: float, *, known_intent: str | None = None,
    novel_scenario: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "known_intent": known_intent,
        "novel_scenario": novel_scenario,
        "confidence": float(min(1.0, max(0.0, confidence))),
    }


def missing_observation_prediction() -> dict[str, Any]:
    return prediction("ABSTAIN", 1.0)


def ask_always_prediction(_: dict[str, Any]) -> dict[str, Any]:
    return prediction("ABSTAIN", 1.0)


def oracle_prediction(record: dict[str, Any]) -> dict[str, Any]:
    truth = truth_for_record(record)
    return prediction(
        truth["status"], 1.0,
        known_intent=truth["known_intent"], novel_scenario=truth["novel_scenario"],
    )


def identifier_grammar_prediction(
    record: dict[str, Any], catalog: dict[str, Any]
) -> dict[str, Any]:
    utterance_tokens = normalized_tokens(record["utterance"])
    intent_candidates = []
    for row in catalog["intents"]:
        label_tokens = normalized_tokens(row["intent"])
        overlap = len(utterance_tokens & label_tokens)
        union = len(utterance_tokens | label_tokens) or 1
        intent_candidates.append((overlap, overlap / union, row["intent_id"]))
    best_overlap, best_jaccard, best_intent = max(
        intent_candidates, key=lambda value: (value[0], value[1], -len(value[2]), value[2])
    )
    if best_overlap > 0:
        return prediction("KNOWN", min(0.99, 0.65 + 0.15 * best_overlap + best_jaccard), known_intent=best_intent)
    scenario_candidates = []
    for scenario in catalog["scenarios"]:
        overlap = len(utterance_tokens & normalized_tokens(scenario))
        scenario_candidates.append((overlap, scenario))
    scenario_overlap, scenario = max(scenario_candidates, key=lambda value: (value[0], value[1]))
    if scenario_overlap > 0:
        return prediction("NOVEL", 0.65, novel_scenario=scenario)
    return prediction("UNSUPPORTED", 0.55)


def fit_character_retrieval(
    training_records: list[dict[str, str]], vectorizer_spec: dict[str, Any]
) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1,
        norm="l2", dtype=np.float64,
    )
    matrix = vectorizer.fit_transform([row["utterance"] for row in training_records])
    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "training_records": training_records,
        "declared_intent_count": len({row["intent_id"] for row in training_records}),
        "vectorizer_spec": vectorizer_spec,
    }


def character_retrieval_observations(
    fitted: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    query_matrix = fitted["vectorizer"].transform([row["utterance"] for row in records])
    similarities = query_matrix @ fitted["matrix"].T
    output: dict[str, dict[str, Any]] = {}
    training = fitted["training_records"]
    for index, row in enumerate(records):
        dense = similarities.getrow(index).toarray().ravel()
        best_index = int(np.argmax(dense))
        nearest = training[best_index]
        output[row["record_id"]] = {
            "similarity": float(dense[best_index]),
            "nearest_intent": nearest["intent_id"],
            "nearest_scenario": nearest["scenario"],
        }
    return output


def retrieval_prediction(
    observation: dict[str, Any], known_threshold: float, unsupported_threshold: float,
) -> dict[str, Any]:
    if unsupported_threshold >= known_threshold:
        raise ValueError("retrieval thresholds are not ordered")
    similarity = observation["similarity"]
    if similarity >= known_threshold:
        return prediction("KNOWN", similarity, known_intent=observation["nearest_intent"])
    if similarity <= unsupported_threshold:
        return prediction("UNSUPPORTED", 1.0 - similarity)
    midpoint = (known_threshold + unsupported_threshold) / 2.0
    half_width = max((known_threshold - unsupported_threshold) / 2.0, 1e-12)
    confidence = max(0.0, 1.0 - abs(similarity - midpoint) / half_width)
    return prediction("NOVEL", confidence, novel_scenario=observation["nearest_scenario"])


def exact_decision(truth: dict[str, Any], predicted: dict[str, Any]) -> bool:
    return bool(
        truth["status"] == predicted["status"]
        and truth["known_intent"] == predicted["known_intent"]
        and truth["novel_scenario"] == predicted["novel_scenario"]
    )


def decision_cost(
    record: dict[str, Any], predicted: dict[str, Any], config: dict[str, Any]
) -> float:
    truth = truth_for_record(record)
    status = predicted["status"]
    costs = config["decisionCosts"]
    if truth["status"] == "KNOWN":
        if status == "KNOWN":
            key = "exact_known" if predicted["known_intent"] == truth["known_intent"] else "wrong_known"
        else:
            key = status.lower()
        return float(costs["known"][key])
    if truth["status"] == "NOVEL":
        if status == "NOVEL":
            key = "exact_novel" if predicted["novel_scenario"] == truth["novel_scenario"] else "wrong_novel_scenario"
        else:
            key = status.lower()
        return float(costs["novel"][key])
    if truth["status"] == "UNSUPPORTED":
        return float(costs["unsupported"][status.lower()])
    return float(costs["insufficient"][status.lower()])


def false_known_acceptance_rate(
    records: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> float:
    non_known = [row for row in records if truth_for_record(row)["status"] != "KNOWN"]
    return sum(predictions[row["record_id"]]["status"] == "KNOWN" for row in non_known) / len(non_known)


def tune_retrieval_thresholds(
    records: list[dict[str, Any]], observations: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["deterministicBaselines"]["character_ngram_retrieval"]
    candidates = []
    for known_threshold in spec["knownThresholdGrid"]:
        for unsupported_threshold in spec["unsupportedThresholdGrid"]:
            if unsupported_threshold >= known_threshold:
                continue
            predictions = {
                row["record_id"]: retrieval_prediction(
                    observations[row["record_id"]], known_threshold, unsupported_threshold,
                )
                for row in records
            }
            costs = [decision_cost(row, predictions[row["record_id"]], config) for row in records]
            exact = [exact_decision(truth_for_record(row), predictions[row["record_id"]]) for row in records]
            candidates.append({
                "known_threshold": known_threshold,
                "unsupported_threshold": unsupported_threshold,
                "mean_regret": sum(costs) / len(costs),
                "exact_decision_accuracy": sum(exact) / len(exact),
                "false_known_acceptance_rate": false_known_acceptance_rate(records, predictions),
            })
    selected = min(candidates, key=lambda row: (
        row["mean_regret"], -row["exact_decision_accuracy"],
        row["false_known_acceptance_rate"], row["known_threshold"],
        row["unsupported_threshold"],
    ))
    return {"selected": selected, "candidate_count": len(candidates)}


def evaluate_predictions(
    records: list[dict[str, Any]], predictions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if {row["record_id"] for row in records} != set(predictions):
        raise ValueError("prediction identifiers do not match records")
    scored = []
    for row in records:
        truth = truth_for_record(row)
        predicted = predictions[row["record_id"]]
        correct = exact_decision(truth, predicted)
        scored.append({
            "record_id": row["record_id"], "class_label": row["class_label"],
            "truth_status": truth["status"], "predicted_status": predicted["status"],
            "exact_decision": correct, "confidence": predicted["confidence"],
            "regret": decision_cost(row, predicted, config),
        })
    observed = [item for item in scored if item["truth_status"] in OBSERVED_STATUSES]
    per_status: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for status in OBSERVED_STATUSES:
        tp = sum(item["truth_status"] == status and item["predicted_status"] == status for item in observed)
        truth_count = sum(item["truth_status"] == status for item in observed)
        prediction_count = sum(item["predicted_status"] == status for item in observed)
        recall = tp / truth_count if truth_count else 0.0
        precision = tp / prediction_count if prediction_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_status[status] = {
            "true_positive_count": tp, "truth_count": truth_count,
            "prediction_count": prediction_count, "precision": precision,
            "recall": recall, "f1": f1,
        }
        f1_values.append(f1)
    known = [row for row in records if truth_for_record(row)["status"] == "KNOWN"]
    novel = [row for row in records if truth_for_record(row)["status"] == "NOVEL"]
    exact_values = [float(item["exact_decision"]) for item in observed]
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        members = [
            item for item in observed
            if lower <= item["confidence"] < upper or (bin_index == 9 and item["confidence"] == 1.0)
        ]
        if members:
            accuracy = sum(item["exact_decision"] for item in members) / len(members)
            mean_confidence = sum(item["confidence"] for item in members) / len(members)
            ece += len(members) / len(observed) * abs(accuracy - mean_confidence)
    ordered = sorted(observed, key=lambda item: (-item["confidence"], item["record_id"]))
    retained_count = max(1, math.ceil(0.8 * len(ordered)))
    retained = ordered[:retained_count]
    per_class: dict[str, dict[str, float | int]] = {}
    for class_label in sorted({item["class_label"] for item in scored}):
        members = [item for item in scored if item["class_label"] == class_label]
        per_class[class_label] = {
            "record_count": len(members),
            "exact_decision_accuracy": sum(item["exact_decision"] for item in members) / len(members),
            "mean_regret": sum(item["regret"] for item in members) / len(members),
        }
    return {
        "record_count": len(records),
        "structured_response_validity": 1.0,
        "observed_record_count": len(observed),
        "observed_exact_decision_accuracy": sum(exact_values) / len(exact_values) if exact_values else 0.0,
        "observed_status_macro_f1": sum(f1_values) / len(f1_values),
        "per_status": per_status,
        "known_exact_intent_accuracy": (
            sum(exact_decision(truth_for_record(row), predictions[row["record_id"]]) for row in known) / len(known)
            if known else 0.0
        ),
        "novel_exact_scenario_accuracy": (
            sum(exact_decision(truth_for_record(row), predictions[row["record_id"]]) for row in novel) / len(novel)
            if novel else 0.0
        ),
        "false_known_acceptance_rate": false_known_acceptance_rate(records, predictions),
        "confidence_ece_10_bin": ece,
        "predicted_decision_correctness_brier": (
            sum((item["confidence"] - float(item["exact_decision"])) ** 2 for item in observed) / len(observed)
            if observed else 0.0
        ),
        "top_confidence_80_percent_coverage": retained_count / len(ordered) if ordered else 0.0,
        "top_confidence_80_percent_error": (
            1.0 - sum(item["exact_decision"] for item in retained) / len(retained) if retained else 0.0
        ),
        "mean_regret": sum(item["regret"] for item in scored) / len(scored),
        "per_class": per_class,
        "scored_rows": scored,
    }


def evaluate_baseline_outcome_gates(
    training_records: list[dict[str, str]], split: dict[str, Any],
    control_count: int, baseline_metrics: dict[str, dict[str, Any]],
    access: dict[str, int], config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["baselineOutcomeGates"]
    training_counts = Counter(row["intent_id"] for row in training_records)
    names = set(baseline_metrics)
    return {
        "training_declared_intent_count": len(training_counts) == gates["requiredTrainingDeclaredIntentCount"],
        "minimum_training_count_per_declared_intent": min(training_counts.values()) >= gates["minimumTrainingRecordCountPerDeclaredIntent"],
        "calibration_record_count": split["counts"]["calibration"] == gates["requiredCalibrationRecordCount"],
        "evaluation_record_count": split["counts"]["evaluation"] == gates["requiredEvaluationRecordCount"],
        "controlled_missing_observation_count": control_count == gates["requiredControlledMissingObservationCount"],
        "required_baselines_complete": names == set(gates["requiredBaselineNames"]),
        "oracle_exact_decision_accuracy": baseline_metrics["oracle"]["observed_exact_decision_accuracy"] == gates["requiredOracleExactDecisionAccuracy"],
        "oracle_mean_regret": baseline_metrics["oracle"]["mean_regret"] == gates["requiredOracleMeanRegret"],
        "complete_safe_enumeration_true_hypothesis_retention": baseline_metrics["complete_safe_enumeration"]["true_hypothesis_retention"] == gates["requiredCompleteSafeEnumerationTrueHypothesisRetention"],
        "zero_protected_test_language_reads": access["protected_test_language_read_count"] <= gates["maximumProtectedTestLanguageReadCount"],
        "zero_manual_utterance_inspection": access["manual_utterance_inspection_count"] <= gates["maximumManualUtteranceInspectionCount"],
        "zero_model_loads": access["model_load_count"] <= gates["maximumModelLoadCount"],
        "zero_model_generations": access["model_generation_count"] <= gates["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= gates["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= gates["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= gates["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"],
    }


def build_deterministic_benchmark_artifacts(
    source_records: list[dict[str, Any]], development_records: list[dict[str, Any]],
    catalog: dict[str, Any], controlled: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    split = split_development_records(development_records, config)
    training_records = build_declared_training_records(source_records, catalog)
    retrieval_spec = config["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training_records, retrieval_spec["vectorizer"])
    calibration_observations = character_retrieval_observations(fitted, split["calibration"])
    evaluation_observations = character_retrieval_observations(fitted, split["evaluation"])
    tuning = tune_retrieval_thresholds(split["calibration"], calibration_observations, config)
    thresholds = tuning["selected"]
    evaluation_predictions = {
        "complete_safe_enumeration": {
            row["record_id"]: ask_always_prediction(row) for row in split["evaluation"]
        },
        "ask_always": {
            row["record_id"]: ask_always_prediction(row) for row in split["evaluation"]
        },
        "identifier_grammar": {
            row["record_id"]: identifier_grammar_prediction(row, catalog) for row in split["evaluation"]
        },
        "character_ngram_retrieval": {
            row["record_id"]: retrieval_prediction(
                evaluation_observations[row["record_id"]],
                thresholds["known_threshold"], thresholds["unsupported_threshold"],
            )
            for row in split["evaluation"]
        },
        "oracle": {
            row["record_id"]: oracle_prediction(row) for row in split["evaluation"]
        },
    }
    baseline_metrics = {}
    for name, predictions in evaluation_predictions.items():
        metrics = evaluate_predictions(split["evaluation"], predictions, config)
        metrics["controlled_missing_observation_abstention_accuracy"] = 1.0
        metrics["true_hypothesis_retention"] = 1.0
        baseline_metrics[name] = metrics
    eligible = ("ask_always", "identifier_grammar", "character_ngram_retrieval")
    best_name = min(
        eligible,
        key=lambda name: (
            baseline_metrics[name]["mean_regret"],
            -baseline_metrics[name]["observed_exact_decision_accuracy"], name,
        ),
    )
    training_counts = dict(sorted(Counter(row["intent_id"] for row in training_records).items()))
    language_free_predictions = {
        name: {
            record_id: predicted for record_id, predicted in sorted(predictions.items())
        }
        for name, predictions in sorted(evaluation_predictions.items())
    }
    return {
        "split": split,
        "training_records": training_records,
        "training_summary": {
            "record_count": len(training_records),
            "declared_intent_count": len(training_counts),
            "records_per_declared_intent": training_counts,
        },
        "retrieval_tuning": tuning,
        "baseline_metrics": baseline_metrics,
        "evaluation_predictions": language_free_predictions,
        "evaluation_prediction_payload_sha256": canonical_sha256(language_free_predictions),
        "best_nonoracle_baseline": {
            "name": best_name,
            "mean_regret": baseline_metrics[best_name]["mean_regret"],
            "observed_exact_decision_accuracy": baseline_metrics[best_name]["observed_exact_decision_accuracy"],
        },
        "controlled_development_count": len(controlled["role_records"]["development"]),
    }


__all__ = [
    "ask_always_prediction", "build_declared_training_records",
    "build_deterministic_benchmark_artifacts",
    "character_retrieval_observations", "decision_cost", "evaluate_baseline_outcome_gates",
    "evaluate_predictions", "fit_character_retrieval", "identifier_grammar_prediction",
    "missing_observation_prediction", "oracle_prediction", "prediction",
    "retrieval_prediction", "split_development_records", "truth_for_record",
    "tune_retrieval_thresholds",
]
