#!/usr/bin/env python3
"""Frozen model-level aggregation and diagnostics for V70 confirmation."""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np


CONTROLS = (
    "map",
    "posterior_sampling",
    "open_loop",
    "myopic_reward",
    "information_only",
)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _affected(row: dict[str, Any], control: str) -> bool:
    return bool(
        int(row[control]["off_support_branch_count"]) > 0
        or float(row[control]["expected_off_support_entry_probability"]) > 0.0
    )


def _fallback_diagnostics(
    rows: list[dict[str, Any]], control: str, threshold: float
) -> dict[str, Any]:
    affected = [row for row in rows if _affected(row, control)]
    fallback_free = [row for row in rows if not _affected(row, control)]
    return {
        "affected_record_count": len(affected),
        "maximum_expected_entry_probability": max(
            (
                float(row[control]["expected_off_support_entry_probability"])
                for row in rows
            ),
            default=0.0,
        ),
        "material_regret_overlap_count": sum(
            row["normalized_regrets"][control] >= threshold for row in affected
        ),
        "qualifying_MAP_overlap_count": (
            sum(
                row["exact_ba_map_root_action_disagreement"]
                and row["normalized_regrets"]["map"] >= threshold
                for row in affected
            )
            if control == "map"
            else None
        ),
        "fallback_free_subset": {
            "record_count": len(fallback_free),
            "normalized_regret": _summary(
                [row["normalized_regrets"][control] for row in fallback_free]
            ),
        },
    }


def aggregate_confirmatory_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    expected_record_count: int,
    source_validation: dict[str, dict[str, bool]],
    record_selection_or_rejection_count: int = 0,
    development_models_rescored: int = 0,
) -> dict[str, Any]:
    if not rows or expected_record_count <= 0:
        raise ValueError("V70 confirmatory aggregation requires sealed rows")
    model_specs = {row["file"]: row for row in config["confirmatoryModels"]}
    expected_models = set(model_specs)
    observed_models = {row["model_file"] for row in rows}
    if not observed_models <= expected_models:
        raise ValueError("V70 rows contain a non-confirmatory model")
    threshold = float(config["normalization"]["materialNormalizedRegret"])
    gates = config["confirmatoryGates"]
    completed_fraction = len(rows) / expected_record_count
    finite_rate = sum(bool(row["all_metrics_finite"]) for row in rows) / len(rows)
    belief_rate = sum(
        row["primary_belief_sum_error"] <= 1e-10
        and row["convergence_belief_sum_error"] <= 1e-10
        for row in rows
    ) / len(rows)
    convergence_error = max(
        row["primary_vs_convergence_normalized_value_error"] for row in rows
    )
    convergence_action_rate = sum(
        bool(row["primary_action_in_convergence_optimal_set"]) for row in rows
    ) / len(rows)
    source_validation_rate = sum(
        bool(checks) and all(checks.values()) for checks in source_validation.values()
    ) / len(expected_models)

    by_model: dict[str, Any] = {}
    qualified_models: list[str] = []
    posterior_sampling_models: list[str] = []
    for model in sorted(expected_models):
        subset = [row for row in rows if row["model_file"] == model]
        disagreement_rows = [
            row for row in subset if row["exact_ba_map_root_action_disagreement"]
        ]
        qualifying_rows = [
            row
            for row in subset
            if row["exact_ba_map_root_action_disagreement"]
            and row["normalized_regrets"]["map"] >= threshold
        ]
        material_counts = {
            control: sum(
                row["normalized_regrets"][control] >= threshold for row in subset
            )
            for control in CONTROLS
        }
        if qualifying_rows:
            qualified_models.append(model)
        if material_counts["posterior_sampling"] > 0:
            posterior_sampling_models.append(model)
        by_model[model] = {
            "stratum": model_specs[model]["stratum"],
            "retained_record_count": len(subset),
            "root_action_disagreement_count": len(disagreement_rows),
            "qualifying_MAP_record_count": len(qualifying_rows),
            "qualifies_paired_MAP_replication": bool(qualifying_rows),
            "material_regret_count": material_counts,
            "normalized_regret": {
                control: _summary(
                    [row["normalized_regrets"][control] for row in subset]
                )
                for control in CONTROLS
            },
            "first_action_disagreements": [
                {
                    "record_id": row["record_id"],
                    "prefix_depth": row["prefix_depth"],
                    "exact_action": row["exact_bayes_adaptive"][
                        "selected_action_name"
                    ],
                    "MAP_action": row["map"]["selected_action_name"],
                    "normalized_MAP_regret": row["normalized_regrets"]["map"],
                    "material": row["normalized_regrets"]["map"] >= threshold,
                    "MAP_fallback_affected": _affected(row, "map"),
                }
                for row in disagreement_rows
            ],
            "fallback_diagnostics": {
                "map": _fallback_diagnostics(subset, "map", threshold),
                "posterior_sampling": _fallback_diagnostics(
                    subset, "posterior_sampling", threshold
                ),
            },
        }

    qualified_by_stratum = Counter(
        model_specs[model]["stratum"] for model in qualified_models
    )
    model_strata = Counter(spec["stratum"] for spec in model_specs.values())
    maximum_map_regret = max(row["normalized_regrets"]["map"] for row in rows)
    metrics = {
        "confirmatory_model_count": len(observed_models),
        "structurally_related_model_count": model_strata["structurally_related"],
        "novel_model_count": model_strata["novel"],
        "retained_record_count": len(rows),
        "completed_record_fraction": completed_fraction,
        "source_validation_rate": source_validation_rate,
        "belief_normalization_rate": belief_rate,
        "finite_metric_rate": finite_rate,
        "maximum_primary_vs_convergence_normalized_value_error": convergence_error,
        "primary_action_in_convergence_optimal_set_rate": convergence_action_rate,
        "paired_MAP_qualifying_model_count": len(qualified_models),
        "paired_MAP_qualifying_structurally_related_model_count": qualified_by_stratum[
            "structurally_related"
        ],
        "paired_MAP_qualifying_novel_model_count": qualified_by_stratum["novel"],
        "material_posterior_sampling_model_count": len(posterior_sampling_models),
        "maximum_normalized_MAP_regret": maximum_map_regret,
        "record_selection_or_rejection_count": record_selection_or_rejection_count,
        "development_models_rescored": development_models_rescored,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    qualified_count = len(qualified_models)
    related_qualified = qualified_by_stratum["structurally_related"]
    novel_qualified = qualified_by_stratum["novel"]
    gate_results = {
        "minimumConfirmatoryModels": len(observed_models)
        >= gates["minimumConfirmatoryModels"],
        "minimumStructurallyRelatedModels": model_strata["structurally_related"]
        >= gates["minimumStructurallyRelatedModels"],
        "minimumNovelModels": model_strata["novel"] >= gates["minimumNovelModels"],
        "minimumCompletedRecordFraction": completed_fraction
        >= gates["minimumCompletedRecordFraction"],
        "minimumSourceValidationRate": source_validation_rate
        >= gates["minimumSourceValidationRate"],
        "minimumBeliefNormalizationRate": belief_rate
        >= gates["minimumBeliefNormalizationRate"],
        "minimumFiniteMetricRate": finite_rate >= gates["minimumFiniteMetricRate"],
        "maximumPrimaryVsConvergenceNormalizedValueError": convergence_error
        <= gates["maximumPrimaryVsConvergenceNormalizedValueError"],
        "minimumPrimaryActionInConvergenceOptimalSetRate": convergence_action_rate
        >= gates["minimumPrimaryActionInConvergenceOptimalSetRate"],
        "minimumModelsWithExactBAMAPRootActionDisagreement": qualified_count
        >= gates["minimumModelsWithExactBAMAPRootActionDisagreement"],
        "minimumStructurallyRelatedModelsWithExactBAMAPRootActionDisagreement": related_qualified
        >= gates[
            "minimumStructurallyRelatedModelsWithExactBAMAPRootActionDisagreement"
        ],
        "minimumNovelModelsWithExactBAMAPRootActionDisagreement": novel_qualified
        >= gates["minimumNovelModelsWithExactBAMAPRootActionDisagreement"],
        "minimumModelsWithMaterialMAPRegret": qualified_count
        >= gates["minimumModelsWithMaterialMAPRegret"],
        "minimumStructurallyRelatedModelsWithMaterialMAPRegret": related_qualified
        >= gates["minimumStructurallyRelatedModelsWithMaterialMAPRegret"],
        "minimumNovelModelsWithMaterialMAPRegret": novel_qualified
        >= gates["minimumNovelModelsWithMaterialMAPRegret"],
        "minimumModelsWithMaterialPosteriorSamplingRegret": len(
            posterior_sampling_models
        )
        >= gates["minimumModelsWithMaterialPosteriorSamplingRegret"],
        "minimumMaximumNormalizedMAPRegret": maximum_map_regret
        >= gates["minimumMaximumNormalizedMAPRegret"],
        "maximumRecordSelectionOrRejectionCount": record_selection_or_rejection_count
        <= gates["maximumRecordSelectionOrRejectionCount"],
        "maximumDevelopmentModelsRescored": development_models_rescored
        <= gates["maximumDevelopmentModelsRescored"],
        "maximumHumanRecordAccessCount": gates["maximumHumanRecordAccessCount"] == 0,
        "maximumModelForwardPassCount": gates["maximumModelForwardPassCount"] == 0,
        "maximumAdapterTrainingRunCount": gates["maximumAdapterTrainingRunCount"]
        == 0,
    }
    passed = all(gate_results.values())
    cheese_names = ("cheese.95.POMDP", "cheese.95_nonterminating.POMDP")
    tier_b = {name: by_model[name] for name in cheese_names}
    return {
        "passed": passed,
        "decision": (
            "confirm_multi_environment_replication_for_project_authored_V69_family"
            if passed
            else "report_complete_negative_or_mixed_confirmatory_result_without_tuning"
        ),
        "metrics": metrics,
        "gate_results": gate_results,
        "qualified_models": qualified_models,
        "posterior_sampling_material_models": posterior_sampling_models,
        "by_model": by_model,
        "by_stratum": {
            stratum: {
                "model_count": model_strata[stratum],
                "paired_MAP_qualifying_model_count": qualified_by_stratum[stratum],
            }
            for stratum in ("structurally_related", "novel")
        },
        "full_census_normalized_regret": {
            control: _summary([row["normalized_regrets"][control] for row in rows])
            for control in CONTROLS
        },
        "Tier_B_cheese_pair": tier_b,
    }
