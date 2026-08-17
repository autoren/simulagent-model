#!/usr/bin/env python3
"""Synthetic record-evaluator compatibility tests for V70."""
from __future__ import annotations

import unittest

import evaluate_v68_development_screen as original
from evaluate_v70_confirmatory import evaluate_record
from test_v68_multi_environment_exact import synthetic_model
from v69_dominant_remapping import build_dominant_remapping_family


class V70ConfirmatoryEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        model = synthetic_model()
        self.primary = build_dominant_remapping_family(
            model, ("a", "b", "c"), quadrature_nodes=9
        )
        self.convergence = build_dominant_remapping_family(
            model, ("a", "b", "c"), quadrature_nodes=17
        )
        self.record = {
            "record_id": "synthetic-root",
            "model_file": "synthetic.POMDP",
            "model_name": "synthetic",
            "stratum": "novel",
            "prefix_depth": 0,
            "actions": [],
            "observations": [],
            "history_probability": 1.0,
            "log_evidence": 0.0,
        }

    def evaluate(self) -> dict:
        return evaluate_record(
            "synthetic.POMDP",
            self.primary,
            self.convergence,
            self.record,
            horizon=2,
            tie_tolerance=1e-12,
            posterior_sampling_points=5,
            posterior_sampling_offset=0.1,
        )

    def test_record_contains_totalized_point_control_diagnostics(self) -> None:
        row = self.evaluate()
        self.assertTrue(row["all_metrics_finite"])
        self.assertIn("off_support_branch_count", row["map"])
        self.assertIn("off_support_branch_count", row["posterior_sampling"])
        self.assertFalse(row["map"]["epsilon_smoothing"])
        self.assertFalse(row["posterior_sampling"]["epsilon_smoothing"])

    def test_base_control_functions_are_restored(self) -> None:
        before_map = original.map_model_policy
        before_ps = original.persistent_posterior_sampling_mixture
        self.evaluate()
        self.assertIs(original.map_model_policy, before_map)
        self.assertIs(original.persistent_posterior_sampling_mixture, before_ps)


if __name__ == "__main__":
    unittest.main()
