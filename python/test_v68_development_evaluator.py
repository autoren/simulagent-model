#!/usr/bin/env python3
"""Synthetic tests for the V68 development evaluator and gate aggregation."""
from __future__ import annotations

import copy
import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from evaluate_v68_development_screen import aggregate_rows, evaluate_record
from test_v68_multi_environment_exact import synthetic_model
from v68_multi_environment_exact import build_command_channel_family


def synthetic_config() -> dict:
    config = json.loads(
        (PROJECT_ROOT / "configs/v68-development-screening.json").read_text()
    )
    config["gates"].update(
        {
            "minimumDevelopmentModels": 1,
            "minimumRetainedRecords": 1,
            "minimumExactBAMinusMAPRootActionDisagreementRecords": 0,
            "minimumExactBAMinusMAPMaterialRegretRecords": 0,
            "minimumMaximumNormalizedMAPRegret": -1.0,
            "minimumExactBAMinusOpenLoopMaterialRegretRecords": 0,
            "minimumExactBAMinusPosteriorSamplingMaterialRegretRecords": 0,
        }
    )
    return config


class V68DevelopmentEvaluatorTests(unittest.TestCase):
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

    def test_record_evaluation_is_finite_and_control_complete(self) -> None:
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
        self.assertTrue(row["all_metrics_finite"])
        self.assertEqual(
            set(row["normalized_regrets"]),
            {"map", "posterior_sampling", "open_loop", "myopic_reward", "information_only"},
        )
        self.assertGreaterEqual(min(row["normalized_regrets"].values()), -1e-12)

    def test_aggregate_passes_relaxed_synthetic_gates(self) -> None:
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
        result = aggregate_rows([row], synthetic_config(), expected_record_count=1)
        self.assertTrue(result["passed"])

    def test_aggregate_detects_missing_record(self) -> None:
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
        result = aggregate_rows([row], synthetic_config(), expected_record_count=2)
        self.assertFalse(result["gate_results"]["minimumCompletedRecordFraction"])

    def test_aggregate_detects_convergence_and_confirmatory_mutants(self) -> None:
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
        mutant = copy.deepcopy(row)
        mutant["primary_vs_convergence_normalized_value_error"] = 1.0
        result = aggregate_rows(
            [mutant], synthetic_config(), expected_record_count=1, confirmatory_models_scored=1
        )
        self.assertFalse(result["gate_results"]["maximumPrimaryVsConvergenceNormalizedValueError"])
        self.assertFalse(result["gate_results"]["maximumConfirmatoryModelsScored"])


if __name__ == "__main__":
    unittest.main()
