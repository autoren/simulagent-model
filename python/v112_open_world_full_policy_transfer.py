from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order
from v106_open_world_benchmark import prediction


def select_fresh_population(
    inventory: dict[str, Any], excluded_population: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["freshPopulation"]
    excluded = {row["candidate_id"] for row in excluded_population["selected_population"]}
    selected: list[dict[str, Any]] = []
    for class_label in spec["classes"]:
        pool = [
            row for row in inventory["candidate_index"]
            if row["partition"] == spec["sourcePartition"]
            and row["class_label"] == class_label
            and row["candidate_id"] not in excluded
        ]
        by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            by_scenario[row["scenario"]].append(row)
        if len(by_scenario) != spec["requiredScenarioCoverage"][class_label]:
            raise ValueError(f"unexpected fresh scenario coverage for {class_label}")
        chosen: dict[str, dict[str, Any]] = {}
        for scenario, rows in sorted(by_scenario.items()):
            first = min(
                rows,
                key=lambda row: hash_order(
                    spec["baseSalt"], class_label, "scenario", scenario, row["candidate_id"],
                ),
            )
            chosen[first["candidate_id"]] = first
        remainder = sorted(
            (row for row in pool if row["candidate_id"] not in chosen),
            key=lambda row: hash_order(
                spec["baseSalt"], class_label, "fill", row["candidate_id"],
            ),
        )
        needed = spec["recordCountPerClass"] - len(chosen)
        if needed < 0 or len(remainder) < needed:
            raise ValueError(f"insufficient fresh population for {class_label}")
        chosen.update((row["candidate_id"], row) for row in remainder[:needed])
        for row in chosen.values():
            selected.append({
                "population_id": f"v112::development_transfer::{row['candidate_id']}",
                "candidate_id": row["candidate_id"], "source_id": row["source_id"],
                "role": "development_transfer", "source_partition": row["partition"],
                "class_label": row["class_label"], "scenario": row["scenario"],
                "intent": row["intent"],
                "current_utterance_intent_overlap_count": row["current_utterance_intent_overlap_count"],
                "slot_type_count": row["slot_type_count"],
            })
    selected.sort(key=lambda row: row["population_id"])
    forbidden = {"utt", "utterance", "annot_utt", "tokens", "slot_values", "text", "prompt"}
    keys = set().union(*(row.keys() for row in selected)) if selected else set()
    if keys & forbidden:
        raise AssertionError("language leaked into V112 population")
    class_counts = Counter(row["class_label"] for row in selected)
    scenario_counts: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        scenario_counts[row["class_label"]].add(row["scenario"])
    return {
        "selected_record_count": len(selected),
        "class_counts": dict(sorted(class_counts.items())),
        "scenario_counts": {key: len(value) for key, value in sorted(scenario_counts.items())},
        "excluded_identifier_overlap_count": sum(row["candidate_id"] in excluded for row in selected),
        "contains_language": False,
        "selected_population_sha256": canonical_sha256(selected),
        "selected_population": selected,
    }


def population_gates(population: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    spec = config["freshPopulation"]
    return {
        "record_count": population["selected_record_count"] == len(spec["classes"]) * spec["recordCountPerClass"],
        "balanced_class_counts": all(
            population["class_counts"].get(label) == spec["recordCountPerClass"]
            for label in spec["classes"]
        ),
        "required_scenario_coverage": all(
            population["scenario_counts"].get(label) == spec["requiredScenarioCoverage"][label]
            for label in spec["classes"]
        ),
        "zero_excluded_identifier_overlap": population["excluded_identifier_overlap_count"] == 0,
        "contains_no_language": not population["contains_language"],
    }


def policy_prediction(
    direct: dict[str, Any], retrieval: dict[str, Any], config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = config["frozenPolicy"]
    novel_candidate = direct["status"] == "ABSTAIN"
    if direct["status"] == "KNOWN" and direct["known_intent"] == retrieval["nearest_intent"]:
        state = "validated_known"
        action = prediction(
            "KNOWN", spec["knownActionConfidence"], known_intent=direct["known_intent"],
        )
    elif direct["status"] == "UNSUPPORTED":
        state = "unsupported_signal"
        action = prediction("UNSUPPORTED", spec["unsupportedActionConfidence"])
    elif novel_candidate:
        state = "abstain_novel_signal"
        action = prediction("ABSTAIN", spec["abstainActionConfidence"])
    else:
        state = "other_disagreement"
        action = prediction("ABSTAIN", spec["abstainActionConfidence"])
    evidence_probability = (
        spec["positiveNovelEvidenceProbability"] if novel_candidate
        else spec["negativeNovelEvidenceProbability"]
    )
    evidence = {
        "novel_candidate": novel_candidate,
        "novel_evidence_probability": evidence_probability,
        "policy_state": state,
        "complete_safe_hypothesis_universe_retained": True,
        "capability_defined": False,
        "executable": False,
    }
    return action, evidence


def novelty_evidence_metrics(
    records: list[dict[str, Any]], evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    truth = [row["class_label"] == "novel_valid" for row in records]
    predicted = [evidence[row["record_id"]]["novel_candidate"] for row in records]
    probabilities = [evidence[row["record_id"]]["novel_evidence_probability"] for row in records]
    tp = sum(p and t for p, t in zip(predicted, truth))
    fp = sum(p and not t for p, t in zip(predicted, truth))
    fn = sum(not p and t for p, t in zip(predicted, truth))
    tn = sum(not p and not t for p, t in zip(predicted, truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ece = 0.0
    for bin_index in range(10):
        lower, upper = bin_index / 10, (bin_index + 1) / 10
        members = [
            index for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (bin_index == 9 and probability == 1.0)
        ]
        if members:
            accuracy = sum(truth[index] for index in members) / len(members)
            mean_probability = sum(probabilities[index] for index in members) / len(members)
            ece += len(members) / len(records) * abs(accuracy - mean_probability)
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "prediction_count": tp + fp, "precision": precision, "recall": recall, "f1": f1,
        "non_novel_false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "ECE_10_bin": ece,
        "Brier": sum((probability - float(target)) ** 2 for probability, target in zip(probabilities, truth)) / len(records),
    }


def policy_quality_gates(
    interface_validity: float, controlled_accuracy: float,
    novel_metrics: dict[str, Any], policy_metrics: dict[str, Any],
    actual_execution_count: int, true_hypothesis_retention: float,
    config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["qualityGates"]
    return {
        "structured_response_validity": interface_validity >= gates["minimumStructuredResponseValidity"],
        "controlled_missing_observation_abstention_accuracy": controlled_accuracy >= gates["minimumControlledMissingObservationAbstentionAccuracy"],
        "novel_evidence_precision": novel_metrics["precision"] >= gates["minimumNovelEvidencePrecision"],
        "novel_evidence_recall": novel_metrics["recall"] >= gates["minimumNovelEvidenceRecall"],
        "novel_evidence_non_novel_false_positive_rate": novel_metrics["non_novel_false_positive_rate"] <= gates["maximumNovelEvidenceNonNovelFalsePositiveRate"],
        "novel_evidence_ECE": novel_metrics["ECE_10_bin"] <= gates["maximumNovelEvidenceECE"],
        "policy_observed_exact_decision_accuracy": policy_metrics["observed_exact_decision_accuracy"] >= gates["minimumPolicyObservedExactDecisionAccuracy"],
        "policy_observed_status_macro_f1": policy_metrics["observed_status_macro_f1"] >= gates["minimumPolicyObservedStatusMacroF1"],
        "policy_known_exact_intent_accuracy": policy_metrics["known_exact_intent_accuracy"] >= gates["minimumPolicyKnownExactIntentAccuracy"],
        "policy_unsupported_status_recall": policy_metrics["per_status"]["UNSUPPORTED"]["recall"] >= gates["minimumPolicyUnsupportedStatusRecall"],
        "policy_unsupported_status_precision": policy_metrics["per_status"]["UNSUPPORTED"]["precision"] >= gates["minimumPolicyUnsupportedStatusPrecision"],
        "policy_false_known_acceptance_rate": policy_metrics["false_known_acceptance_rate"] <= gates["maximumPolicyFalseKnownAcceptanceRate"],
        "policy_confidence_ECE": policy_metrics["confidence_ece_10_bin"] <= gates["maximumPolicyConfidenceECE"],
        "policy_top_confidence_80_percent_error": policy_metrics["top_confidence_80_percent_error"] <= gates["maximumPolicyTopConfidence80PercentError"],
        "policy_mean_decision_regret": policy_metrics["mean_regret"] <= gates["maximumPolicyMeanDecisionRegret"],
        "true_hypothesis_retention": true_hypothesis_retention == gates["requiredTrueHypothesisRetention"],
        "zero_actual_execution": actual_execution_count <= gates["maximumActualExecutionCount"],
    }


__all__ = [
    "novelty_evidence_metrics", "policy_prediction", "policy_quality_gates",
    "population_gates", "select_fresh_population",
]
