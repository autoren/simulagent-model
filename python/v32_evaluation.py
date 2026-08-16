"""Registered V32 scoring, absolute gates, and family-paired comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def mean(values: Sequence[bool | float]) -> float:
    return float(np.mean(values)) if values else 0.0


def score_rows(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {row["id"]: row for row in predictions}
    if set(lookup) != {row["id"] for row in records}:
        raise ValueError("V32 predictions do not exactly cover the requested population")
    result = []
    for record in records:
        target, prediction = record["target"], lookup[record["id"]]
        selected, intermediate = prediction["selected_fields"], prediction["selected_intermediates"]
        expected2 = target["arguments"][1] if target["predicate_kind"] == "relation" else "N/A"
        checks = {
            "predicate_correct": selected["predicate"] == target["predicate"],
            "argument1_correct": selected["argument_1"] == target["arguments"][0],
            "argument2_correct": selected["argument_2"] == expected2,
            "truth_correct": selected["truth_status"] == target["truth_status"],
            "direct_truth_correct": intermediate["direct_truth_status"] == target["truth_status"],
            "lexical_sign_correct": intermediate["lexical_sign"] == target["factorization"]["lexical_sign"],
            "outer_operation_correct": intermediate["outer_operation"] == target["factorization"]["outer_operation"],
        }
        exact = checks["predicate_correct"] and checks["argument1_correct"] and checks["argument2_correct"] and checks["truth_correct"]
        result.append({
            "id": record["id"], "scene_id": record["scene_id"], "split": record["split"],
            "predicate_kind": target["predicate_kind"], "predicate": target["predicate"],
            "truth_status": target["truth_status"], "lexical_sign": target["factorization"]["lexical_sign"],
            "outer_operation": target["factorization"]["outer_operation"],
            "surface_family": record["oracle_metadata"]["surface_family"],
            "evaluation_stratum": record["oracle_metadata"]["evaluation_stratum"],
            "scene_variant": record["oracle_metadata"]["scene_variant"],
            "entity_count": record["oracle_metadata"]["entity_count"],
            "sentence_length_stratum": record["oracle_metadata"]["sentence_length_stratum"],
            **checks, "relation_order_correct": checks["argument1_correct"] and checks["argument2_correct"],
            "exact_signed_fact": exact, "pairs": record["oracle_metadata"]["pairs"],
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
        for pair in row["pairs"]:
            groups[(pair["kind"], pair["id"])].append(row)
    result = {}
    for kind in sorted({key[0] for key in groups}):
        selected = [members for (current, _), members in groups.items() if current == kind]
        if any(len(members) != 2 for members in selected):
            raise ValueError(f"Malformed V32 controlled pair: {kind}")
        result[kind] = {"pairs": len(selected), "pair_exact": mean([all(row["exact_signed_fact"] for row in pair) for pair in selected])}
    return result


def summarize_seed(
    records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
    config: dict[str, Any], apply_gates: bool,
) -> dict[str, Any]:
    rows = score_rows(records, predictions)
    relations = [row for row in rows if row["predicate_kind"] == "relation"]
    scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenes[row["scene_id"]].append(row)
    summary = {
        "records": len(rows), "scenes": len(scenes),
        "predicate_accuracy": mean([row["predicate_correct"] for row in rows]),
        "argument1_accuracy": mean([row["argument1_correct"] for row in rows]),
        "argument2_accuracy": mean([row["argument2_correct"] for row in rows]),
        "relation_argument_order_accuracy": mean([row["relation_order_correct"] for row in relations]),
        "truth_status_accuracy": mean([row["truth_correct"] for row in rows]),
        "direct_truth_status_accuracy": mean([row["direct_truth_correct"] for row in rows]),
        "lexical_sign_accuracy": mean([row["lexical_sign_correct"] for row in rows]),
        "outer_operation_accuracy": mean([row["outer_operation_correct"] for row in rows]),
        "exact_signed_fact_accuracy": mean([row["exact_signed_fact"] for row in rows]),
        "exact_scene_accuracy": mean([all(row["exact_signed_fact"] for row in values) for values in scenes.values()]),
        "by_stratum": grouped(rows, "evaluation_stratum", "exact_signed_fact"),
        "by_outer_operation": grouped(rows, "outer_operation", "exact_signed_fact"),
        "truth_by_outer_operation": grouped(rows, "outer_operation", "truth_correct"),
        "by_surface_family": grouped(rows, "surface_family", "exact_signed_fact"),
        "by_predicate": grouped(rows, "predicate", "exact_signed_fact"),
        "by_truth_status": grouped(rows, "truth_status", "truth_correct"),
        "controlled_pairs": pair_metrics(rows),
    }
    if not apply_gates:
        return summary
    gates = config["gates"]["absolutePerSeed"]
    scope_kinds = ["scope_assert_deny", "scope_assert_double_deny", "scope_assert_contrast"]
    pairs = summary["controlled_pairs"]
    checks = {
        "predicate_accuracy": summary["predicate_accuracy"] >= gates["minimumPredicateAccuracy"],
        "argument1_accuracy": summary["argument1_accuracy"] >= gates["minimumArgument1Accuracy"],
        "relation_argument_order_accuracy": summary["relation_argument_order_accuracy"] >= gates["minimumRelationArgumentOrderAccuracy"],
        "truth_status_accuracy": summary["truth_status_accuracy"] >= gates["minimumTruthStatusAccuracy"],
        "exact_signed_fact_accuracy": summary["exact_signed_fact_accuracy"] >= gates["minimumExactSignedFactAccuracy"],
        "exact_scene_accuracy": summary["exact_scene_accuracy"] >= gates["minimumExactSceneAccuracy"],
        "worst_surface_family_exact": min(value["exact_signed_fact"] for value in summary["by_surface_family"].values()) >= gates["minimumWorstSurfaceFamilyExact"],
        "worst_outer_operation_truth": min(value["truth_correct"] for value in summary["truth_by_outer_operation"].values()) >= gates["minimumWorstOuterOperationTruth"],
        "distractor_pair_exact": pairs["distractor"]["pair_exact"] >= gates["minimumDistractorPairExact"],
        "inverse_pair_exact": pairs["inverse"]["pair_exact"] >= gates["minimumInversePairExact"],
        "argument_reversal_pair_exact": pairs["argument_reversal"]["pair_exact"] >= gates["minimumArgumentReversalPairExact"],
        "lexical_sign_pair_exact": pairs["lexical_sign_assert"]["pair_exact"] >= gates["minimumLexicalSignPairExact"],
        "scope_pair_exact": min(pairs[kind]["pair_exact"] for kind in scope_kinds) >= gates["minimumScopePairExact"],
    }
    summary["checks"], summary["passed"] = checks, all(checks.values())
    return summary


def system_summary(seed_results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    values, gates = list(seed_results.values()), config["gates"]["absoluteSystem"]
    metrics = (
        "predicate_accuracy", "argument1_accuracy", "relation_argument_order_accuracy",
        "truth_status_accuracy", "exact_signed_fact_accuracy", "exact_scene_accuracy",
        "lexical_sign_accuracy", "outer_operation_accuracy",
    )
    checks = {
        "required_passing_seeds": sum(row["passed"] for row in values) >= gates["requiredPassingSeeds"],
        "mean_predicate_accuracy": mean([row["predicate_accuracy"] for row in values]) >= gates["minimumMeanPredicateAccuracy"],
        "mean_exact_signed_fact_accuracy": mean([row["exact_signed_fact_accuracy"] for row in values]) >= gates["minimumMeanExactSignedFactAccuracy"],
        "mean_exact_scene_accuracy": mean([row["exact_scene_accuracy"] for row in values]) >= gates["minimumMeanExactSceneAccuracy"],
    }
    return {
        "seeds": seed_results, "mean": {key: mean([row[key] for row in values]) for key in metrics},
        "minimum": {key: min(row[key] for row in values) for key in metrics},
        "checks": checks, "passed": all(checks.values()),
    }


def family_bootstrap_delta(
    records: Sequence[dict[str, Any]], baseline: dict[str, Sequence[dict[str, Any]]],
    challenger: dict[str, Sequence[dict[str, Any]]], config: dict[str, Any],
) -> dict[str, Any]:
    families = sorted({row["oracle_metadata"]["surface_family"] for row in records})
    lookup = {row["id"]: row for row in records}
    deltas = []
    for family in families:
        ids = {row["id"] for row in records if row["oracle_metadata"]["surface_family"] == family}
        systems = []
        for predictions_by_seed in (baseline, challenger):
            seed_values = []
            for predictions in predictions_by_seed.values():
                chosen = [row for row in predictions if row["id"] in ids]
                scored = score_rows([lookup[row["id"]] for row in chosen], chosen)
                seed_values.append(mean([row["exact_signed_fact"] for row in scored]))
            systems.append(mean(seed_values))
        deltas.append(systems[1] - systems[0])
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(config["evaluation"]["bootstrapSeed"])
    replicates = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(config["evaluation"]["bootstrapReplicates"])]
    return {
        "families": len(families), "mean_exact_signed_fact_delta": float(np.mean(values)),
        "family_deltas": dict(zip(families, deltas, strict=True)),
        "bootstrap_95_interval": [float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))],
    }
