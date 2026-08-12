"""Grouped, paired, directional, worst-stratum, and gate metrics for V7."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from evaluate_v5_challenge_mlx import safe_metrics


Row = dict[str, Any]


def grouped_context_metrics(rows: list[Row], threshold: float) -> dict[str, Any]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[row["split_group"]].append(row)
    summaries = []
    for group, values in sorted(groups.items()):
        correct = [
            (value["score"] > threshold) == value["gold_ambiguous"] for value in values
        ]
        metric = safe_metrics(values, threshold)
        summaries.append(
            {
                "split_group": group,
                "examples": len(values),
                "accuracy": float(np.mean(correct)),
                "balanced_accuracy": metric.get("balanced_accuracy"),
                "both_labels": metric["metrics_available"],
            }
        )
    balanced = [
        value["balanced_accuracy"] for value in summaries if value["both_labels"]
    ]
    return {
        "groups": len(summaries),
        "groups_with_both_labels": len(balanced),
        "macro_accuracy": float(np.mean([value["accuracy"] for value in summaries])),
        "macro_balanced_accuracy_on_two_label_groups": (
            float(np.mean(balanced)) if balanced else None
        ),
        "worst_group_accuracy": min(value["accuracy"] for value in summaries),
        "group_summaries": summaries,
    }


def paired_evidence_metrics(rows: list[Row], threshold: float) -> dict[str, Any]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if (
            row["surface_variant"] == "canonical"
            and row["evidence_intervention_kind"] == "oracle_label_change"
        ):
            groups[row["evidence_pair_id"]].append(row)
    score_direction: list[bool] = []
    classification_direction: list[bool] = []
    complete: list[bool] = []
    summaries = []
    for pair_id, values in sorted(groups.items()):
        ambiguous = [value for value in values if value["gold_ambiguous"]]
        identifiable = [value for value in values if not value["gold_ambiguous"]]
        if not ambiguous or not identifiable:
            continue
        score_comparisons = [
            left["score"] > right["score"]
            for left in ambiguous
            for right in identifiable
        ]
        classification_comparisons = [
            left["score"] > threshold and right["score"] <= threshold
            for left in ambiguous
            for right in identifiable
        ]
        group_complete = all(
            (value["score"] > threshold) == value["gold_ambiguous"]
            for value in values
        )
        score_direction.extend(score_comparisons)
        classification_direction.extend(classification_comparisons)
        complete.append(group_complete)
        summaries.append(
            {
                "evidence_pair_id": pair_id,
                "records": len(values),
                "evidence_variants": sorted({value["evidence_variant"] for value in values}),
                "cross_label_comparisons": len(score_comparisons),
                "paired_score_directional_accuracy": float(np.mean(score_comparisons)),
                "evidence_directional_accuracy": float(
                    np.mean(classification_comparisons)
                ),
                "complete_classification": group_complete,
            }
        )
    return {
        "groups": len(summaries),
        "cross_label_comparisons": len(score_direction),
        "paired_score_directional_accuracy": (
            float(np.mean(score_direction)) if score_direction else 0.0
        ),
        "evidence_directional_accuracy": (
            float(np.mean(classification_direction))
            if classification_direction
            else 0.0
        ),
        "complete_group_accuracy": float(np.mean(complete)) if complete else 0.0,
        "group_summaries": summaries,
    }


def worst_stratum_metrics(
    rows: list[Row], threshold: float, minimum_support: int = 4
) -> dict[str, Any]:
    definitions = {
        "evidence_variant": lambda row: row["evidence_variant"],
        "action_template": lambda row: row["action_template"],
        "evidence_x_action": lambda row: (
            f"{row['evidence_variant']}|{row['action_template']}"
        ),
    }
    eligible = []
    by_dimension = {}
    for dimension, key in definitions.items():
        groups: dict[str, list[Row]] = defaultdict(list)
        for row in rows:
            groups[str(key(row))].append(row)
        summaries = {}
        for name, values in sorted(groups.items()):
            metric = safe_metrics(values, threshold)
            metric["eligible_for_worst_stratum"] = (
                len(values) >= minimum_support and metric["metrics_available"]
            )
            summaries[name] = metric
            if metric["eligible_for_worst_stratum"]:
                eligible.append(
                    {
                        "dimension": dimension,
                        "stratum": name,
                        "examples": len(values),
                        "balanced_accuracy": metric["balanced_accuracy"],
                    }
                )
        by_dimension[dimension] = summaries
    if not eligible:
        raise RuntimeError("No V7 two-label stratum meets the minimum support")
    worst = min(eligible, key=lambda value: (value["balanced_accuracy"], value["dimension"], value["stratum"]))
    return {
        "minimum_support": minimum_support,
        "eligible_strata": len(eligible),
        "worst": worst,
        "by_dimension": by_dimension,
    }


def gate_report(
    lock: dict[str, Any],
    canonical: dict[str, Any],
    bootstrap: dict[str, Any],
    surfaces: dict[str, dict[str, Any]],
    invariance: dict[str, Any],
    paired: dict[str, Any],
    worst: dict[str, Any],
) -> dict[str, Any]:
    gates = lock["gates"]
    checks = [
        {
            "name": "development_calibration_balanced_accuracy",
            "value": lock["calibration_gate"]["value"],
            "minimum": gates["minimumCalibrationCanonicalBalancedAccuracy"],
        },
        {
            "name": "holdout_canonical_balanced_accuracy",
            "value": canonical["balanced_accuracy"],
            "minimum": gates["minimumHoldoutCanonicalBalancedAccuracy"],
        },
        {
            "name": "holdout_bootstrap_lower_bound",
            "value": bootstrap["balanced_accuracy_95_percentile_interval"][0],
            "minimum": gates["minimumHoldoutBootstrapLowerBound"],
        },
        *[
            {
                "name": f"surface_{surface}_balanced_accuracy",
                "value": surfaces[surface]["balanced_accuracy"],
                "minimum": gates["minimumSurfaceBalancedAccuracy"],
            }
            for surface in ("entity_renamed", "paraphrased")
        ],
        *[
            {
                "name": f"surface_{surface}_prediction_agreement",
                "value": invariance["transformations"][surface]["prediction_agreement"],
                "minimum": gates["minimumSurfacePredictionAgreement"],
            }
            for surface in ("entity_renamed", "paraphrased")
        ],
        {
            "name": "complete_surface_triplet_accuracy",
            "value": invariance["complete_triplet_accuracy"],
            "minimum": gates["minimumCompleteTripletAccuracy"],
        },
        {
            "name": "evidence_directional_accuracy",
            "value": paired["evidence_directional_accuracy"],
            "minimum": gates["minimumEvidenceDirectionalAccuracy"],
        },
        {
            "name": "paired_score_directional_accuracy",
            "value": paired["paired_score_directional_accuracy"],
            "minimum": gates["minimumPairedScoreDirectionalAccuracy"],
        },
        {
            "name": "worst_stratum_balanced_accuracy",
            "value": worst["worst"]["balanced_accuracy"],
            "minimum": gates["minimumWorstStratumBalancedAccuracy"],
        },
    ]
    for check in checks:
        check["passed"] = check["value"] >= check["minimum"]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}
