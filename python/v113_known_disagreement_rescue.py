from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from v106_open_world_benchmark import evaluate_predictions, prediction
from v112_open_world_full_policy_transfer import (
    novelty_evidence_metrics, policy_prediction, policy_quality_gates,
)


def extract_rescue_features(
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
    for index, record in enumerate(records):
        dense = matrix.getrow(index).toarray().ravel()
        intent_scores = {
            intent_id: float(np.max(dense[indices]))
            for intent_id, indices in intent_indices.items()
        }
        nearest = min(intent_scores, key=lambda key: (-intent_scores[key], key))
        direct = direct_predictions[record["record_id"]]
        proposed = direct["known_intent"] if direct["status"] == "KNOWN" else None
        eligible = bool(proposed and proposed != nearest)
        proposed_score = intent_scores.get(proposed, 0.0)
        output.append({
            "record_id": record["record_id"], "class_label": record["class_label"],
            "eligible": eligible, "proposed_intent_score": proposed_score,
            "nearest_intent_score": intent_scores[nearest],
            "score_gap": intent_scores[nearest] - proposed_score,
            "llm_confidence": float(direct["confidence"]),
            "proposed_and_nearest_same_scenario": bool(
                proposed and intent_scenario.get(proposed) == intent_scenario[nearest]
            ),
        })
    return output


def enumerate_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    scores = config["thresholdGrids"]["proposalScore"]
    gaps = config["thresholdGrids"]["scoreGap"]
    confidences = config["thresholdGrids"]["confidence"]
    rules: list[dict[str, Any]] = [
        {"family": "no_rescue", "complexity": 0},
        {"family": "accept_all", "complexity": 1},
        {"family": "same_scenario_only", "complexity": 1},
    ]
    rules.extend({"family": "proposal_score_at_least", "minimum_score": score, "complexity": 1} for score in scores)
    rules.extend(
        {"family": "proposal_score_and_gap", "minimum_score": score, "maximum_gap": gap, "complexity": 2}
        for score in scores for gap in gaps
    )
    rules.extend(
        {"family": "proposal_score_and_same_scenario", "minimum_score": score, "complexity": 2}
        for score in scores
    )
    rules.extend(
        {"family": "confidence_at_least", "minimum_confidence": confidence, "complexity": 1}
        for confidence in confidences
    )
    rules.extend(
        {"family": "confidence_and_same_scenario", "minimum_confidence": confidence, "complexity": 2}
        for confidence in confidences
    )
    rules.extend(
        {
            "family": "proposal_score_confidence_and_same_scenario",
            "minimum_score": score, "minimum_confidence": confidence, "complexity": 3,
        }
        for score in scores for confidence in confidences
    )
    if {row["family"] for row in rules} != set(config["registeredRuleFamilies"]):
        raise ValueError("V113 registered rule family mismatch")
    return rules


def rule_signature(rule: dict[str, Any]) -> str:
    return "|".join(f"{key}={rule[key]}" for key in sorted(rule) if key != "complexity")


def apply_rule(rule: dict[str, Any], feature: dict[str, Any]) -> bool:
    if not feature["eligible"]:
        return False
    family = rule["family"]
    score = feature["proposed_intent_score"] >= rule.get("minimum_score", 0.0)
    gap = feature["score_gap"] <= rule.get("maximum_gap", 1.01)
    confidence = feature["llm_confidence"] >= rule.get("minimum_confidence", 0.0)
    same = feature["proposed_and_nearest_same_scenario"]
    if family == "no_rescue": return False
    if family == "accept_all": return True
    if family == "same_scenario_only": return same
    if family == "proposal_score_at_least": return score
    if family == "proposal_score_and_gap": return score and gap
    if family == "proposal_score_and_same_scenario": return score and same
    if family == "confidence_at_least": return confidence
    if family == "confidence_and_same_scenario": return confidence and same
    if family == "proposal_score_confidence_and_same_scenario": return score and confidence and same
    raise ValueError(f"unknown V113 rule: {family}")


def rescued_policy_predictions(
    rule: dict[str, Any], records: list[dict[str, Any]], features: list[dict[str, Any]],
    direct: dict[str, dict[str, Any]], retrieval: dict[str, dict[str, Any]],
    v112_config: dict[str, Any], v113_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], int]:
    by_id = {row["record_id"]: row for row in features}
    actions, evidence, rescued = {}, {}, 0
    for record in records:
        identifier = record["record_id"]
        action, item_evidence = policy_prediction(direct[identifier], retrieval[identifier], v112_config)
        if apply_rule(rule, by_id[identifier]):
            action = prediction(
                "KNOWN", v113_config["rescueActionConfidence"],
                known_intent=direct[identifier]["known_intent"],
            )
            rescued += 1
        actions[identifier], evidence[identifier] = action, item_evidence
    return actions, evidence, rescued


def normalized_violation(gates: dict[str, bool], metrics: dict[str, Any], config: dict[str, Any]) -> float:
    spec = config["qualityGates"]
    values = {
        "policy_known_exact_intent_accuracy": max(0.0, spec["minimumPolicyKnownExactIntentAccuracy"] - metrics["known_exact_intent_accuracy"]),
        "policy_top_confidence_80_percent_error": max(0.0, metrics["top_confidence_80_percent_error"] - spec["maximumPolicyTopConfidence80PercentError"]),
        "policy_false_known_acceptance_rate": max(0.0, metrics["false_known_acceptance_rate"] - spec["maximumPolicyFalseKnownAcceptanceRate"]),
        "policy_mean_decision_regret": max(0.0, metrics["mean_regret"] - spec["maximumPolicyMeanDecisionRegret"]),
    }
    return sum(values.get(name, 1.0) for name, passed in gates.items() if not passed)


def build_census(
    records: list[dict[str, Any]], features: list[dict[str, Any]],
    direct: dict[str, dict[str, Any]], retrieval: dict[str, dict[str, Any]],
    v112_config: dict[str, Any], v113_config: dict[str, Any], baseline_config: dict[str, Any],
    interface_validity: float, controlled_accuracy: float,
) -> dict[str, Any]:
    candidates = []
    for rule in enumerate_rules(v113_config):
        actions, evidence, rescued = rescued_policy_predictions(
            rule, records, features, direct, retrieval, v112_config, v113_config,
        )
        metrics = evaluate_predictions(records, actions, baseline_config)
        aggregate = {key: value for key, value in metrics.items() if key != "scored_rows"}
        novelty = novelty_evidence_metrics(records, evidence)
        gates = policy_quality_gates(
            interface_validity, controlled_accuracy, novelty, aggregate, 0, 1.0, v112_config,
        )
        candidates.append({
            "rule": rule, "rescued_count": rescued, "metrics": aggregate,
            "novel_evidence_metrics": novelty, "gates": gates,
            "passed_gate_count": sum(gates.values()), "feasible": all(gates.values()),
            "normalized_violation": normalized_violation(gates, aggregate, v112_config),
        })
    selected = min(candidates, key=lambda row: (
        not row["feasible"], -row["passed_gate_count"], row["normalized_violation"],
        row["metrics"]["mean_regret"], row["metrics"]["false_known_acceptance_rate"],
        -row["metrics"]["known_exact_intent_accuracy"], row["rule"]["complexity"],
        rule_signature(row["rule"]),
    ))
    return {
        "candidate_count": len(candidates),
        "feasible_candidate_count": sum(row["feasible"] for row in candidates),
        "selected": selected,
        "eligible_record_count": sum(row["eligible"] for row in features),
        "individual_feature_prediction_identifier_language_or_response_emission_count": 0,
    }


__all__ = [
    "apply_rule", "build_census", "enumerate_rules", "extract_rescue_features",
    "rescued_policy_predictions", "rule_signature",
]
