#!/usr/bin/env python3
"""Synthetic gate and reporting tests for V70 confirmatory aggregation."""
from __future__ import annotations

import copy
import unittest

from v70_confirmatory_aggregation import aggregate_confirmatory_rows


def config() -> dict:
    models = [
        {"file": "related.POMDP", "stratum": "structurally_related"},
        {"file": "novel-a.POMDP", "stratum": "novel"},
        {"file": "novel-b.POMDP", "stratum": "novel"},
        {"file": "cheese.95.POMDP", "stratum": "novel"},
        {"file": "cheese.95_nonterminating.POMDP", "stratum": "novel"},
    ]
    return {
        "confirmatoryModels": models,
        "normalization": {"materialNormalizedRegret": 0.005},
        "confirmatoryGates": {
            "minimumConfirmatoryModels": 5,
            "minimumStructurallyRelatedModels": 1,
            "minimumNovelModels": 4,
            "minimumCompletedRecordFraction": 1.0,
            "minimumSourceValidationRate": 1.0,
            "minimumBeliefNormalizationRate": 1.0,
            "minimumFiniteMetricRate": 1.0,
            "maximumPrimaryVsConvergenceNormalizedValueError": 1e-8,
            "minimumPrimaryActionInConvergenceOptimalSetRate": 1.0,
            "minimumModelsWithExactBAMAPRootActionDisagreement": 3,
            "minimumStructurallyRelatedModelsWithExactBAMAPRootActionDisagreement": 1,
            "minimumNovelModelsWithExactBAMAPRootActionDisagreement": 2,
            "minimumModelsWithMaterialMAPRegret": 3,
            "minimumStructurallyRelatedModelsWithMaterialMAPRegret": 1,
            "minimumNovelModelsWithMaterialMAPRegret": 2,
            "minimumModelsWithMaterialPosteriorSamplingRegret": 2,
            "minimumMaximumNormalizedMAPRegret": 0.01,
            "maximumRecordSelectionOrRejectionCount": 0,
            "maximumDevelopmentModelsRescored": 0,
            "maximumHumanRecordAccessCount": 0,
            "maximumModelForwardPassCount": 0,
            "maximumAdapterTrainingRunCount": 0,
        },
    }


def row(model: str, *, qualifies: bool, ps_material: bool = True) -> dict:
    map_regret = 0.02 if qualifies else 0.0
    return {
        "record_id": f"{model}-root",
        "model_file": model,
        "prefix_depth": 0,
        "all_metrics_finite": True,
        "primary_belief_sum_error": 0.0,
        "convergence_belief_sum_error": 0.0,
        "primary_vs_convergence_normalized_value_error": 0.0,
        "primary_action_in_convergence_optimal_set": True,
        "exact_ba_map_root_action_disagreement": qualifies,
        "exact_bayes_adaptive": {"selected_action_name": "a"},
        "map": {
            "selected_action_name": "b" if qualifies else "a",
            "off_support_branch_count": 0,
            "expected_off_support_entry_probability": 0.0,
        },
        "posterior_sampling": {
            "off_support_branch_count": 0,
            "expected_off_support_entry_probability": 0.0,
        },
        "normalized_regrets": {
            "map": map_regret,
            "posterior_sampling": 0.01 if ps_material else 0.0,
            "open_loop": 0.01,
            "myopic_reward": 0.01,
            "information_only": 0.01,
        },
    }


class V70ConfirmatoryAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = config()
        self.rows = [
            row("related.POMDP", qualifies=True),
            row("novel-a.POMDP", qualifies=True),
            row("novel-b.POMDP", qualifies=True),
            row("cheese.95.POMDP", qualifies=False),
            row("cheese.95_nonterminating.POMDP", qualifies=False),
        ]
        self.validation = {
            spec["file"]: {"valid": True}
            for spec in self.config["confirmatoryModels"]
        }

    def aggregate(self, rows: list[dict] | None = None) -> dict:
        selected = self.rows if rows is None else rows
        return aggregate_confirmatory_rows(
            selected,
            self.config,
            expected_record_count=len(selected),
            source_validation=self.validation,
        )

    def test_model_level_stratified_fixture_passes(self) -> None:
        result = self.aggregate()
        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["paired_MAP_qualifying_model_count"], 3)
        self.assertEqual(
            result["metrics"]["paired_MAP_qualifying_novel_model_count"], 2
        )

    def test_disagreement_and_regret_must_cooccur_on_same_record(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["normalized_regrets"]["map"] = 0.0
        split = copy.deepcopy(rows[0])
        split["record_id"] = "related-split"
        split["exact_ba_map_root_action_disagreement"] = False
        split["normalized_regrets"]["map"] = 0.02
        rows.append(split)
        result = aggregate_confirmatory_rows(
            rows,
            self.config,
            expected_record_count=len(rows),
            source_validation=self.validation,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["metrics"]["paired_MAP_qualifying_structurally_related_model_count"],
            0,
        )

    def test_many_records_from_one_model_cannot_compensate(self) -> None:
        rows = [row("related.POMDP", qualifies=True) for _ in range(20)]
        for index, item in enumerate(rows):
            item["record_id"] = f"related-{index}"
        result = aggregate_confirmatory_rows(
            rows,
            self.config,
            expected_record_count=len(rows),
            source_validation=self.validation,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["paired_MAP_qualifying_model_count"], 1)

    def test_fallback_diagnostics_do_not_change_primary(self) -> None:
        baseline = self.aggregate()
        changed_rows = copy.deepcopy(self.rows)
        changed_rows[0]["map"]["off_support_branch_count"] = 2
        changed_rows[0]["map"]["expected_off_support_entry_probability"] = 0.2
        changed = self.aggregate(changed_rows)
        self.assertEqual(changed["passed"], baseline["passed"])
        self.assertEqual(changed["gate_results"], baseline["gate_results"])
        self.assertEqual(
            changed["by_model"]["related.POMDP"]["fallback_diagnostics"]["map"][
                "affected_record_count"
            ],
            1,
        )

    def test_cheese_pair_is_reported_separately(self) -> None:
        result = self.aggregate()
        self.assertEqual(
            set(result["Tier_B_cheese_pair"]),
            {"cheese.95.POMDP", "cheese.95_nonterminating.POMDP"},
        )


if __name__ == "__main__":
    unittest.main()
