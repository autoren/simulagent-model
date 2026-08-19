#!/usr/bin/env python3
"""Outcome-blind structural tests for the V79 terminal-utility successor."""
from __future__ import annotations

import json
import unittest

import numpy as np

from evaluate_v79_terminal_utility_benchmark import evaluate_gates
from test_v78_clarification_benchmark import fake_fixture, tiny_terminal_kernel
from v22r2_grounding import PROJECT_ROOT
from v79_terminal_utility_planning import (
    ACTIVE_UNRESOLVED_TERMINAL_UTILITY,
    TERMINAL_STATE_TERMINAL_UTILITY,
    evaluate_policy_exact,
    plan_exact,
    terminal_utility,
)


class V79TerminalUtilityBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads(
            (PROJECT_ROOT / "configs/v79-terminal-utility-design-lock.json").read_text()
        )
        cls.config = cls.lock["resolved_config_payload"]

    def test_terminal_utility_semantic_ordering_is_frozen(self) -> None:
        self.assertEqual(ACTIVE_UNRESOLVED_TERMINAL_UTILITY, -6.0)
        self.assertEqual(TERMINAL_STATE_TERMINAL_UTILITY, 0.0)
        positive = self.config["positiveRewardProfile"]
        self.assertLess(ACTIVE_UNRESOLVED_TERMINAL_UTILITY, positive["abstain"])
        self.assertLess(positive["abstain"], positive["safePreview"])

    def test_horizon_one_synthetic_policy_resolves_instead_of_expiring(self) -> None:
        kernel, belief = tiny_terminal_kernel()
        policy = plan_exact(kernel, belief, 1)
        self.assertEqual(kernel.action_names[policy["selected_action"]], "safe_preview")
        self.assertEqual(policy["value"], 2.0)
        self.assertEqual(evaluate_policy_exact(kernel, belief, policy, 1), 2.0)

    def test_terminal_utility_distinguishes_active_and_terminal_beliefs(self) -> None:
        kernel, belief = tiny_terminal_kernel()
        terminal = np.zeros_like(belief)
        terminal[:, 1] = belief[:, 0]
        self.assertEqual(terminal_utility(kernel, belief), -6.0)
        self.assertEqual(terminal_utility(kernel, terminal), 0.0)

    def test_augmented_gate_aggregation_is_full_path_boolean(self) -> None:
        fixtures = {
            row["name"]: fake_fixture(row["name"])
            for row in self.config["fixtures"]
        }
        fixtures["clear_tool_intent"]["exact"][
            "root_action"
        ] = "execute_schedule_chen"
        fixtures["unknown_heavy_tool_intent"]["exact"][
            "root_action"
        ] = "ask_full_details"
        fixtures["dominant_safe_preview"]["exact"]["root_action"] = "safe_preview"
        fixtures["dominant_safe_preview"]["map"]["normalized_regret"] = 0.0
        fixtures["dominant_safe_preview"]["posterior_sampling"][
            "normalized_regret"
        ] = 0.0
        for row in fixtures.values():
            row["terminal_utility"] = {
                "initial_active_belief_at_horizon_zero": -6.0,
                "matched_terminal_belief_at_horizon_zero": 0.0,
                "exact_policy_replay_agrees": True,
            }
        access = {
            "model_forward_pass_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        }
        gates = evaluate_gates(fixtures, self.config, access)
        self.assertTrue(all(isinstance(value, bool) for value in gates.values()))
        self.assertTrue(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
