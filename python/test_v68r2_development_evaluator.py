#!/usr/bin/env python3
"""Synthetic evaluator tests for V68r2 point-control totalization."""
from __future__ import annotations

import unittest

import numpy as np

import evaluate_v68_development_screen as original
from evaluate_v68r2_development_screen import evaluate_record
from test_v68_multi_environment_exact import synthetic_model
from v68_multi_environment_exact import build_command_channel_family


class V68r2DevelopmentEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        model = synthetic_model()
        self.primary = build_command_channel_family(
            model, ("a", "b", "c"), quadrature_nodes=9
        )
        self.convergence = build_command_channel_family(
            model, ("a", "b", "c"), quadrature_nodes=17
        )
        self.record = {
            "record_id": "synthetic-root",
            "model_file": "synthetic.POMDP",
            "model_name": "synthetic",
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

    def test_full_support_preserves_original_metrics(self) -> None:
        repaired = self.evaluate()
        baseline = original.evaluate_record(
            "synthetic.POMDP",
            self.primary,
            self.convergence,
            self.record,
            horizon=2,
            tie_tolerance=1e-12,
            posterior_sampling_points=5,
            posterior_sampling_offset=0.1,
        )
        self.assertEqual(repaired["exact_bayes_adaptive"], baseline["exact_bayes_adaptive"])
        self.assertEqual(repaired["map"]["static_index"], baseline["map"]["static_index"])
        self.assertAlmostEqual(
            repaired["map"]["exact_environment_value"],
            baseline["map"]["exact_environment_value"],
            places=14,
        )
        self.assertAlmostEqual(
            repaired["posterior_sampling"]["value"],
            baseline["posterior_sampling"]["value"],
            places=14,
        )

    def test_both_diagnostics_are_complete(self) -> None:
        row = self.evaluate()
        for name in ("map", "posterior_sampling"):
            self.assertEqual(
                row[name]["fallback_action"], self.primary.kernel.canonical_actions[0]
            )
            self.assertTrue(row[name]["off_support_fallback_is_open_loop"])
            self.assertFalse(row[name]["epsilon_smoothing"])
            self.assertTrue(np.isfinite(row[name]["expected_off_support_entry_probability"]))
        self.assertFalse(row["map"]["off_support_model_reselection"])
        self.assertFalse(row["posterior_sampling"]["off_support_model_resampling"])

    def test_both_monkey_patches_are_restored(self) -> None:
        before_map = original.map_model_policy
        before_ps = original.persistent_posterior_sampling_mixture
        self.evaluate()
        self.assertIs(original.map_model_policy, before_map)
        self.assertIs(original.persistent_posterior_sampling_mixture, before_ps)


if __name__ == "__main__":
    unittest.main()
