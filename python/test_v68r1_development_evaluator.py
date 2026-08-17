#!/usr/bin/env python3
"""Synthetic evaluator tests for the repaired V68r1 control."""
from __future__ import annotations

import unittest

import numpy as np

import evaluate_v68_development_screen as original
from evaluate_v68r1_development_screen import evaluate_record
from test_v68_multi_environment_exact import synthetic_model
from v68_multi_environment_exact import build_command_channel_family


class V68r1DevelopmentEvaluatorTests(unittest.TestCase):
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

    def test_full_support_preserves_original_evaluator_metrics(self) -> None:
        repaired = evaluate_record(
            "synthetic.POMDP",
            self.primary,
            self.convergence,
            self.record,
            horizon=2,
            tie_tolerance=1e-12,
            posterior_sampling_points=5,
            posterior_sampling_offset=0.1,
        )
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
        self.assertEqual(
            repaired["exact_bayes_adaptive"], baseline["exact_bayes_adaptive"]
        )
        self.assertEqual(repaired["map"], baseline["map"])
        self.assertAlmostEqual(
            repaired["posterior_sampling"]["value"],
            baseline["posterior_sampling"]["value"],
            places=14,
        )
        self.assertEqual(repaired["posterior_sampling"]["off_support_branch_count"], 0)

    def test_repair_diagnostics_are_complete_and_frozen(self) -> None:
        row = evaluate_record(
            "synthetic.POMDP",
            self.primary,
            self.convergence,
            self.record,
            horizon=2,
            tie_tolerance=1e-12,
            posterior_sampling_points=5,
            posterior_sampling_offset=0.1,
        )
        posterior = row["posterior_sampling"]
        self.assertEqual(posterior["fallback_action"], self.primary.kernel.canonical_actions[0])
        self.assertTrue(posterior["sampled_model_persists_on_supported_histories"])
        self.assertTrue(posterior["off_support_fallback_is_open_loop"])
        self.assertFalse(posterior["off_support_model_resampling"])
        self.assertFalse(posterior["epsilon_smoothing"])
        self.assertTrue(np.isfinite(posterior["expected_off_support_entry_probability"]))

    def test_monkey_patch_is_restored_after_each_record(self) -> None:
        before = original.persistent_posterior_sampling_mixture
        evaluate_record(
            "synthetic.POMDP",
            self.primary,
            self.convergence,
            self.record,
            horizon=2,
            tie_tolerance=1e-12,
            posterior_sampling_points=5,
            posterior_sampling_offset=0.1,
        )
        self.assertIs(original.persistent_posterior_sampling_mixture, before)


if __name__ == "__main__":
    unittest.main()
