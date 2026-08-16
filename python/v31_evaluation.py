"""Metrics, gates, and paired system comparison for V31."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def mean(values: Sequence[bool | float]) -> float:
    return float(np.mean(values)) if values else 0.0


def score_rows(
    records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {row["id"]: row for row in predictions}
    if set(lookup) != {row["id"] for row in records}:
        raise ValueError("V31 predictions do not exactly cover the requested population")
    result = []
    for record in records:
        target, selected = record["target"], lookup[record["id"]]["selected_fields"]
        expected2 = target["arguments"][1] if target["predicate_kind"] == "relation" else "N/A"
        predicate_ok = selected["predicate"] == target["predicate"]
        argument1_ok = selected["argument_1"] == target["arguments"][0]
        argument2_ok = selected["argument_2"] == expected2
        truth_ok = selected["truth_status"] == target["truth_status"]
        exact = predicate_ok and argument1_ok and argument2_ok and truth_ok
        result.append({
            "id": record["id"], "scene_id": record["scene_id"], "split": record["split"],
            "predicate_kind": target["predicate_kind"], "predicate": target["predicate"],
            "truth_status": target["truth_status"],
            "semantic_operator": record["oracle_metadata"]["semantic_operator"],
            "surface_family": record["oracle_metadata"]["surface_family"],
            "scene_variant": record["oracle_metadata"]["scene_variant"],
            "entity_count": record["oracle_metadata"]["entity_count"],
            "sentence_length_stratum": record["oracle_metadata"]["sentence_length_stratum"],
            "predicate_correct": predicate_ok, "argument1_correct": argument1_ok,
            "argument2_correct": argument2_ok,
            "relation_order_correct": argument1_ok and argument2_ok,
            "truth_correct": truth_ok, "exact_signed_fact": exact,
            "pairs": record["oracle_metadata"]["pairs"],
        })
    return result


def grouped(rows: Sequence[dict[str, Any]], field: str, metric: str) -> dict[str, dict[str, Any]]:
    return {
        str(value): {"records": len(selected), metric: mean([row[metric] for row in selected])}
        for value in sorted({row[field] for row in rows}, key=str)
        for selected in [[row for row in rows if row[field] == value]]
    }


def pair_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for pair in row["pairs"]: groups[(pair["kind"], pair["id"])].append(row)
    result = {}
    for kind in sorted({key[0] for key in groups}):
        selected = [members for (current, _), members in groups.items() if current == kind]
        if any(len(members) != 2 for members in selected):
            raise ValueError(f"Malformed V31 controlled pair: {kind}")
        result[kind] = {
            "pairs": len(selected),
            "pair_exact": mean([all(row["exact_signed_fact"] for row in members) for members in selected]),
        }
    return result


def summarize_seed(
    records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
    config: dict[str, Any], apply_gates: bool,
) -> dict[str, Any]:
    rows = score_rows(records, predictions)
    relations = [row for row in rows if row["predicate_kind"] == "relation"]
    scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: scenes[row["scene_id"]].append(row)
    summary = {
        "records": len(rows), "scenes": len(scenes),
        "predicate_accuracy": mean([row["predicate_correct"] for row in rows]),
        "argument1_accuracy": mean([row["argument1_correct"] for row in rows]),
        "argument2_accuracy": mean([row["argument2_correct"] for row in rows]),
        "relation_argument_order_accuracy": mean([row["relation_order_correct"] for row in relations]),
        "truth_status_accuracy": mean([row["truth_correct"] for row in rows]),
        "exact_signed_fact_accuracy": mean([row["exact_signed_fact"] for row in rows]),
        "exact_scene_accuracy": mean([all(row["exact_signed_fact"] for row in values) for values in scenes.values()]),
        "by_semantic_operator": grouped(rows, "semantic_operator", "exact_signed_fact"),
        "truth_by_semantic_operator": grouped(rows, "semantic_operator", "truth_correct"),
        "by_surface_family": grouped(rows, "surface_family", "exact_signed_fact"),
        "relation_by_surface_family": grouped(relations, "surface_family", "exact_signed_fact"),
        "by_truth_status": grouped(rows, "truth_status", "truth_correct"),
        "by_predicate": grouped(rows, "predicate", "exact_signed_fact"),
        "by_entity_count": grouped(rows, "entity_count", "exact_signed_fact"),
        "by_sentence_length": grouped(rows, "sentence_length_stratum", "exact_signed_fact"),
        "by_scene_variant": grouped(rows, "scene_variant", "exact_signed_fact"),
        "controlled_pairs": pair_metrics(rows),
    }
    if not apply_gates:
        return summary
    gates = config["gates"]["perSeedLanguage"]
    checks = {
        "predicate_accuracy": summary["predicate_accuracy"] >= gates["minimumPredicateAccuracy"],
        "argument1_accuracy": summary["argument1_accuracy"] >= gates["minimumArgument1Accuracy"],
        "relation_argument_order_accuracy": summary["relation_argument_order_accuracy"] >= gates["minimumRelationArgumentOrderAccuracy"],
        "truth_status_accuracy": summary["truth_status_accuracy"] >= gates["minimumTruthStatusAccuracy"],
        "exact_signed_fact_accuracy": summary["exact_signed_fact_accuracy"] >= gates["minimumExactSignedFactAccuracy"],
        "exact_scene_accuracy": summary["exact_scene_accuracy"] >= gates["minimumExactSceneAccuracy"],
        "worst_operator_truth_accuracy": min(row["truth_correct"] for row in summary["truth_by_semantic_operator"].values()) >= gates["minimumWorstOperatorTruthAccuracy"],
        "worst_relation_surface_exact": min(row["exact_signed_fact"] for row in summary["relation_by_surface_family"].values()) >= gates["minimumWorstRelationSurfaceExact"],
        "worst_surface_family_exact": min(row["exact_signed_fact"] for row in summary["by_surface_family"].values()) >= gates["minimumWorstSurfaceFamilyExact"],
        "affirmative_negated_pair_exact": summary["controlled_pairs"]["affirmative_negated"]["pair_exact"] >= gates["minimumAffirmativeNegatedPairExact"],
        "affirmative_double_negation_pair_exact": summary["controlled_pairs"]["affirmative_double_negation"]["pair_exact"] >= gates["minimumAffirmativeDoubleNegationPairExact"],
        "argument_reversal_pair_exact": summary["controlled_pairs"]["argument_reversal"]["pair_exact"] >= gates["minimumArgumentReversalPairExact"],
        "inverse_pair_exact": summary["controlled_pairs"]["inverse"]["pair_exact"] >= gates["minimumInversePairExact"],
        "distractor_pair_exact": summary["controlled_pairs"]["distractor"]["pair_exact"] >= gates["minimumDistractorPairExact"],
        "false_unknown_pair_exact": summary["controlled_pairs"]["false_unknown"]["pair_exact"] >= gates["minimumFalseUnknownPairExact"],
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    return summary


def system_summary(seed_results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    values = list(seed_results.values())
    gates = config["gates"]["systemStability"]
    checks = {
        "required_passing_seeds": sum(row["passed"] for row in values) >= gates["requiredPassingSeeds"],
        "mean_predicate_accuracy": mean([row["predicate_accuracy"] for row in values]) >= gates["minimumMeanPredicateAccuracy"],
        "mean_exact_signed_fact_accuracy": mean([row["exact_signed_fact_accuracy"] for row in values]) >= gates["minimumMeanExactSignedFactAccuracy"],
        "mean_exact_scene_accuracy": mean([row["exact_scene_accuracy"] for row in values]) >= gates["minimumMeanExactSceneAccuracy"],
    }
    return {
        "seeds": seed_results,
        "mean": {
            metric: mean([row[metric] for row in values])
            for metric in (
                "predicate_accuracy", "argument1_accuracy", "relation_argument_order_accuracy",
                "truth_status_accuracy", "exact_signed_fact_accuracy", "exact_scene_accuracy",
            )
        },
        "minimum": {
            metric: min(row[metric] for row in values)
            for metric in (
                "predicate_accuracy", "argument1_accuracy", "relation_argument_order_accuracy",
                "truth_status_accuracy", "exact_signed_fact_accuracy", "exact_scene_accuracy",
            )
        },
        "checks": checks, "passed": all(checks.values()),
    }


def family_bootstrap_delta(
    records: Sequence[dict[str, Any]], frozen_predictions: dict[str, Sequence[dict[str, Any]]],
    lora_predictions: dict[str, Sequence[dict[str, Any]]], config: dict[str, Any],
) -> dict[str, Any]:
    families = sorted({row["oracle_metadata"]["surface_family"] for row in records})
    record_lookup = {row["id"]: row for row in records}
    family_deltas = []
    for family in families:
        ids = {row["id"] for row in records if row["oracle_metadata"]["surface_family"] == family}
        system_values = []
        for predictions_by_seed in (frozen_predictions, lora_predictions):
            seed_values = []
            for predictions in predictions_by_seed.values():
                selected = [row for row in predictions if row["id"] in ids]
                scored = score_rows([record_lookup[row["id"]] for row in selected], selected)
                seed_values.append(mean([row["exact_signed_fact"] for row in scored]))
            system_values.append(mean(seed_values))
        family_deltas.append(system_values[1] - system_values[0])
    rng = np.random.default_rng(config["evaluation"]["bootstrapSeed"])
    replicates = []
    values = np.asarray(family_deltas, dtype=np.float64)
    for _ in range(config["evaluation"]["bootstrapReplicates"]):
        replicates.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
    return {
        "families": len(families), "mean_exact_signed_fact_delta": float(np.mean(values)),
        "family_deltas": dict(zip(families, family_deltas, strict=True)),
        "bootstrap_95_interval": [float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))],
    }
