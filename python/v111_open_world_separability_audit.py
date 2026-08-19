from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def extract_features(
    fitted: dict[str, Any], records: list[dict[str, Any]],
    direct_predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    matrix = fitted["vectorizer"].transform([row["utterance"] for row in records]) @ fitted["matrix"].T
    training = fitted["training_records"]
    intent_indices: dict[str, list[int]] = defaultdict(list)
    intent_scenario: dict[str, str] = {}
    for index, row in enumerate(training):
        intent_indices[row["intent_id"]].append(index)
        intent_scenario[row["intent_id"]] = row["scenario"]
    output = []
    for row_index, record in enumerate(records):
        dense = matrix.getrow(row_index).toarray().ravel()
        intent_scores = {
            intent_id: float(np.max(dense[indices]))
            for intent_id, indices in intent_indices.items()
        }
        ranked_intents = sorted(intent_scores, key=lambda key: (-intent_scores[key], key))
        top_intent, second_intent = ranked_intents[:2]
        scenario_scores: dict[str, float] = defaultdict(float)
        for intent_id, score in intent_scores.items():
            scenario_scores[intent_scenario[intent_id]] = max(scenario_scores[intent_scenario[intent_id]], score)
        ranked_scenarios = sorted(scenario_scores, key=lambda key: (-scenario_scores[key], key))
        top_scenario, second_scenario = ranked_scenarios[:2]
        direct = direct_predictions[record["record_id"]]
        proposed = direct["known_intent"] if direct["status"] == "KNOWN" else None
        output.append({
            "record_id": record["record_id"], "class_label": record["class_label"],
            "is_novel": record["class_label"] == "novel_valid",
            "top_intent_score": intent_scores[top_intent],
            "second_intent_score": intent_scores[second_intent],
            "intent_margin": intent_scores[top_intent] - intent_scores[second_intent],
            "top_scenario_score": scenario_scores[top_scenario],
            "second_scenario_score": scenario_scores[second_scenario],
            "scenario_margin": scenario_scores[top_scenario] - scenario_scores[second_scenario],
            "proposed_intent_score": intent_scores.get(proposed, 0.0),
            "llm_confidence": direct["confidence"],
            "llm_is_known": direct["status"] == "KNOWN",
            "llm_is_abstain": direct["status"] == "ABSTAIN",
            "llm_is_unsupported": direct["status"] == "UNSUPPORTED",
            "llm_retrieval_intent_disagree": bool(proposed and proposed != top_intent),
        })
    return output


def enumerate_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    score = config["thresholdGrids"]["score"]
    margin = config["thresholdGrids"]["margin"]
    proposal = config["thresholdGrids"]["proposal"]
    bands = [(low, high) for low in score for high in score if low < high]
    rules: list[dict[str, Any]] = []
    for low, high in bands:
        rules.append({"family": "top_score_band", "low": low, "high": high, "complexity": 2})
        rules.append({"family": "llm_nonknown_top_score_band", "low": low, "high": high, "complexity": 3})
        rules.append({"family": "not_llm_unsupported_top_score_band", "low": low, "high": high, "complexity": 3})
        rules.append({"family": "top_score_band_or_known_disagreement", "low": low, "high": high, "complexity": 3})
        for maximum_margin in margin:
            rules.append({
                "family": "top_score_band_and_low_margin", "low": low, "high": high,
                "maximum_margin": maximum_margin, "complexity": 3,
            })
            rules.append({
                "family": "llm_known_top_score_band_and_low_margin", "low": low,
                "high": high, "maximum_margin": maximum_margin, "complexity": 4,
            })
    for maximum_margin in margin:
        rules.append({"family": "low_intent_margin", "maximum_margin": maximum_margin, "complexity": 1})
    for threshold in proposal:
        rules.append({"family": "llm_known_proposal_score_below", "threshold": threshold, "complexity": 2})
    rules.append({"family": "llm_known_retrieval_disagreement", "complexity": 1})
    rules.append({"family": "llm_abstain_only", "complexity": 1})
    registered = set(config["registeredRuleFamilies"])
    if {rule["family"] for rule in rules} != registered:
        raise ValueError("registered V111 rule families mismatch")
    return rules


def rule_signature(rule: dict[str, Any]) -> str:
    return "|".join(f"{key}={rule[key]}" for key in sorted(rule) if key != "complexity")


def apply_rule(rule: dict[str, Any], feature: dict[str, Any]) -> bool:
    family = rule["family"]
    in_band = rule.get("low", 0.0) <= feature["top_intent_score"] < rule.get("high", 1.01)
    if family == "top_score_band":
        return in_band
    if family == "low_intent_margin":
        return feature["intent_margin"] <= rule["maximum_margin"]
    if family == "top_score_band_and_low_margin":
        return in_band and feature["intent_margin"] <= rule["maximum_margin"]
    if family == "llm_known_proposal_score_below":
        return feature["llm_is_known"] and feature["proposed_intent_score"] < rule["threshold"]
    if family == "llm_known_retrieval_disagreement":
        return feature["llm_is_known"] and feature["llm_retrieval_intent_disagree"]
    if family == "llm_nonknown_top_score_band":
        return not feature["llm_is_known"] and in_band
    if family == "not_llm_unsupported_top_score_band":
        return not feature["llm_is_unsupported"] and in_band
    if family == "llm_known_top_score_band_and_low_margin":
        return feature["llm_is_known"] and in_band and feature["intent_margin"] <= rule["maximum_margin"]
    if family == "top_score_band_or_known_disagreement":
        return in_band or (feature["llm_is_known"] and feature["llm_retrieval_intent_disagree"])
    if family == "llm_abstain_only":
        return feature["llm_is_abstain"]
    raise ValueError(f"unknown rule family: {family}")


def binary_metrics(features: list[dict[str, Any]], rule: dict[str, Any]) -> dict[str, Any]:
    predicted = [apply_rule(rule, row) for row in features]
    truth = [row["is_novel"] for row in features]
    tp = sum(p and t for p, t in zip(predicted, truth))
    fp = sum(p and not t for p, t in zip(predicted, truth))
    fn = sum(not p and t for p, t in zip(predicted, truth))
    tn = sum(not p and not t for p, t in zip(predicted, truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "prediction_count": tp + fp, "precision": precision, "recall": recall, "f1": f1,
        "non_novel_false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def feasible(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    spec = config["calibrationSelection"]
    return bool(
        metrics["precision"] >= spec["minimumNovelPrecisionForFeasibleCandidate"]
        and metrics["recall"] >= spec["minimumNovelRecallForFeasibleCandidate"]
        and metrics["non_novel_false_positive_rate"] <= spec["maximumNonNovelFalsePositiveRateForFeasibleCandidate"]
    )


def select_rule(
    rules: list[dict[str, Any]], features: list[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for rule in rules:
        metrics = binary_metrics(features, rule)
        candidates.append({"rule": rule, "metrics": metrics, "feasible": feasible(metrics, config)})
    any_feasible = any(row["feasible"] for row in candidates)
    pool = [row for row in candidates if row["feasible"]] if any_feasible else candidates
    selected = min(pool, key=lambda row: (
        -row["metrics"]["f1"], row["metrics"]["false_positive"],
        -row["metrics"]["recall"], row["rule"]["complexity"], rule_signature(row["rule"]),
    ))
    return {
        "selected": selected, "candidate_count": len(candidates),
        "feasible_candidate_count": sum(row["feasible"] for row in candidates),
    }


def quantiles(values: list[float]) -> dict[str, float]:
    return {
        name: float(np.quantile(values, q))
        for name, q in (("minimum", 0.0), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("maximum", 1.0))
    }


def aggregate_feature_distributions(features: list[dict[str, Any]], registered: list[str]) -> dict[str, Any]:
    numeric = [name for name in registered if name not in {
        "llm_is_known", "llm_is_abstain", "llm_is_unsupported", "llm_retrieval_intent_disagree",
    }]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        by_class[row["class_label"]].append(row)
    return {
        class_label: {
            name: quantiles([float(row[name]) for row in rows]) for name in numeric
        } | {
            name: sum(bool(row[name]) for row in rows) / len(rows)
            for name in registered if name not in numeric
        }
        for class_label, rows in sorted(by_class.items())
    }


def build_separability_analysis(
    calibration_features: list[dict[str, Any]], evaluation_features: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    rules = enumerate_rules(config)
    calibration = select_rule(rules, calibration_features, config)
    selected_rule = calibration["selected"]["rule"]
    selected_evaluation = binary_metrics(evaluation_features, selected_rule)
    oracle = select_rule(rules, evaluation_features, config)
    return {
        "candidate_count": len(rules),
        "calibration": calibration,
        "selected_evaluation_metrics": selected_evaluation,
        "evaluation_oracle": oracle,
        "feature_distributions": {
            "calibration": aggregate_feature_distributions(calibration_features, config["registeredFeatures"]),
            "evaluation": aggregate_feature_distributions(evaluation_features, config["registeredFeatures"]),
        },
        "individual_feature_or_identifier_emission_count": 0,
    }


def selected_rule_passes(analysis: dict[str, Any], config: dict[str, Any]) -> bool:
    metrics = analysis["selected_evaluation_metrics"]
    gates = config["evaluationGates"]
    return bool(
        metrics["precision"] >= gates["minimumSelectedRuleNovelPrecision"]
        and metrics["recall"] >= gates["minimumSelectedRuleNovelRecall"]
        and metrics["non_novel_false_positive_rate"] <= gates["maximumSelectedRuleNonNovelFalsePositiveRate"]
    )


__all__ = [
    "apply_rule", "binary_metrics", "build_separability_analysis", "enumerate_rules",
    "extract_features", "rule_signature", "select_rule", "selected_rule_passes",
]
