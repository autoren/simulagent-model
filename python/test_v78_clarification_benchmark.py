#!/usr/bin/env python3
"""Outcome-blind structural and full-path tests for V78."""
from __future__ import annotations

import json
import unittest

import numpy as np

from evaluate_v78_clarification_benchmark import control_summary, evaluate_gates
from v22r2_grounding import PROJECT_ROOT
from v77_clarification_benchmark import ClarificationKernel, certified_actions
from v78_clarification_benchmark import (
    ACTION_NAMES,
    EXECUTION_ACTIONS,
    HYPOTHESIS_NAMES,
    NONE_HYPOTHESIS,
    OBSERVATION_NAMES,
    SAFE_PREVIEW_ACTION,
    build_fixture,
    structural_diagnostics,
)


def tiny_terminal_kernel() -> tuple[ClarificationKernel, np.ndarray]:
    transition = np.zeros((2, 2, 2))
    transition[:, 1, 1] = 1.0
    transition[0, 0, 0] = 1.0
    transition[1, 0, 1] = 1.0
    observation = np.zeros((2, 2, 2, 2))
    observation[..., 1] = 1.0
    observation[:, 0, 0] = (0.5, 0.5)
    reward = np.zeros((2, 2, 2, 2))
    reward[:, 1, 0, 1] = 2.0
    kernel = ClarificationKernel(
        hypothesis_names=("candidate", "none"),
        action_names=("inspect", "safe_preview"),
        observation_names=("signal", "done"),
        state_names=("active", "terminal"),
        transition=transition,
        observation=observation,
        reward=reward,
        discount=1.0,
        send_minimum_matching_posterior=0.9,
        send_maximum_none_posterior=0.1,
        send_action_to_hypothesis=(),
        none_hypothesis=1,
        always_certified_actions=(0, 1),
    )
    belief = np.zeros((2, 2))
    belief[:, 0] = 0.5
    return kernel, belief


def fake_fixture(name: str) -> dict:
    return {
        "name": name,
        "structural": {
            "hypothesis_count": 5,
            "transition_normalization_rate": 1.0,
            "observation_normalization_rate": 1.0,
            "identical_hypothesis_support_rate": 1.0,
            "belief_normalizes": True,
        },
        "resource": {"belief_normalization_rate": 1.0},
        "exact": {
            "root_action": "ask_operation",
            "reachable_information_actions": ["ask_operation", "ask_recipient"],
            "safe_unknown_continuation_count": 1,
            "unknown_branch_irreversible_execution_count": 0,
            "complete_belief_certificate_violation_count": 0,
        },
        "map": {"normalized_regret": 0.06, "off_support_fallback_count": 0},
        "posterior_sampling": {
            "normalized_regret": 0.01,
            "off_support_fallback_count": 0,
        },
        "act_immediately": {"normalized_regret": 0.06},
        "ask_always": {"normalized_regret": 0.03},
    }


class V78ClarificationBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lock = json.loads(
            (PROJECT_ROOT / "configs/v78-clarification-design-lock.json").read_text()
        )
        cls.config = lock["config_payload"]

    def test_registered_population_is_structurally_complete(self) -> None:
        for row in self.config["fixtures"]:
            fixture = build_fixture(self.config, row["name"])
            kernel = fixture.kernel
            self.assertEqual(kernel.hypothesis_names, HYPOTHESIS_NAMES)
            self.assertEqual(kernel.action_names, ACTION_NAMES)
            self.assertEqual(kernel.observation_names, OBSERVATION_NAMES)
            self.assertEqual(kernel.transition.shape, (13, 2, 2))
            self.assertEqual(kernel.observation.shape, (5, 13, 2, 14))
            self.assertEqual(kernel.reward.shape, (5, 13, 2, 2))
            self.assertFalse(kernel.transition.flags.writeable)
            self.assertFalse(kernel.observation.flags.writeable)
            self.assertFalse(kernel.reward.flags.writeable)

    def test_channels_normalize_with_shared_support_and_operational_none(self) -> None:
        fixture = build_fixture(self.config, "ambiguous_tool_intent")
        diagnostics = structural_diagnostics(fixture)
        self.assertEqual(diagnostics["transition_normalization_rate"], 1.0)
        self.assertEqual(diagnostics["observation_normalization_rate"], 1.0)
        self.assertEqual(diagnostics["identical_hypothesis_support_rate"], 1.0)
        full_other = OBSERVATION_NAMES.index("full_other")
        self.assertAlmostEqual(
            fixture.kernel.observation[NONE_HYPOTHESIS, 2, 0, full_other], 0.97
        )

    def test_complete_belief_execution_certificate_is_fail_closed(self) -> None:
        ambiguous = build_fixture(self.config, "ambiguous_tool_intent")
        clear = build_fixture(self.config, "clear_tool_intent")
        ambiguous_actions = certified_actions(
            ambiguous.kernel, ambiguous.initial_belief
        )
        clear_actions = certified_actions(clear.kernel, clear.initial_belief)
        self.assertTrue(
            all(action not in ambiguous_actions for action in EXECUTION_ACTIONS)
        )
        self.assertIn(ACTION_NAMES.index("execute_schedule_chen"), clear_actions)
        self.assertIn(SAFE_PREVIEW_ACTION, ambiguous_actions)

    def test_terminal_shadow_policy_is_evaluable_at_longer_horizon(self) -> None:
        kernel, belief = tiny_terminal_kernel()
        policy = {
            "terminal": False,
            "horizon": 2,
            "selected_action": 1,
            "branches": {},
        }
        summary = control_summary(kernel, belief, policy, 2)
        self.assertEqual(summary["value"], 2.0)
        self.assertEqual(summary["complete_belief_certificate_violation_count"], 0)

    def test_gate_aggregation_retains_fixture_identity_and_returns_booleans(self) -> None:
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
        access = {
            "model_forward_pass_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        }
        gates = evaluate_gates(fixtures, self.config, access)
        self.assertTrue(gates)
        self.assertTrue(all(isinstance(value, bool) for value in gates.values()))
        self.assertTrue(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
