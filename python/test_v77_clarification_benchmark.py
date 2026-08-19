#!/usr/bin/env python3
"""Model-free structural and synthetic-unit tests for the V77 benchmark."""
from __future__ import annotations

import json
import unittest

import numpy as np

from v22r2_grounding import PROJECT_ROOT
from v77_clarification_benchmark import (
    ACTION_NAMES,
    HYPOTHESIS_NAMES,
    OBSERVATION_NAMES,
    SAFE_DRAFT_ACTION,
    SEND_ACTIONS,
    ClarificationKernel,
    build_fixture,
    certified_actions,
    exact_step,
    plan_exact,
    structural_diagnostics,
)


class V77ClarificationBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        design = json.loads(
            (PROJECT_ROOT / "configs/v77-clarification-design-lock.json").read_text()
        )
        cls.config = design["config_payload"]

    def test_registered_fixture_shapes_and_immutability(self) -> None:
        for row in self.config["fixtures"]:
            fixture = build_fixture(self.config, row["name"])
            kernel = fixture.kernel
            self.assertEqual(kernel.transition.shape, (13, 2, 2))
            self.assertEqual(kernel.observation.shape, (5, 13, 2, 14))
            self.assertEqual(kernel.reward.shape, (5, 13, 2, 2))
            self.assertEqual(fixture.initial_belief.shape, (5, 2))
            self.assertFalse(kernel.transition.flags.writeable)
            self.assertFalse(kernel.observation.flags.writeable)
            self.assertFalse(kernel.reward.flags.writeable)
            self.assertFalse(fixture.initial_belief.flags.writeable)

    def test_registered_observation_channels_normalize_with_identical_support(self) -> None:
        fixture = build_fixture(self.config, "ambiguous_send")
        observation = fixture.kernel.observation
        np.testing.assert_allclose(observation.sum(axis=-1), 1.0, atol=1e-12)
        for hypothesis in range(1, len(HYPOTHESIS_NAMES)):
            self.assertTrue(
                np.array_equal(observation[0] > 0.0, observation[hypothesis] > 0.0)
            )
        diagnostics = structural_diagnostics(fixture)
        self.assertEqual(diagnostics["observation_normalization_rate"], 1.0)
        self.assertEqual(diagnostics["identical_hypothesis_support_rate"], 1.0)

    def test_question_semantics_and_none_of_above_are_explicit(self) -> None:
        kernel = build_fixture(self.config, "ambiguous_send").kernel
        active = 0
        report_q2 = OBSERVATION_NAMES.index("report_q2")
        report_annual = OBSERVATION_NAMES.index("report_annual")
        report_other = OBSERVATION_NAMES.index("report_other")
        self.assertAlmostEqual(kernel.observation[0, 0, active, report_q2], 0.96)
        self.assertAlmostEqual(kernel.observation[1, 0, active, report_annual], 0.96)
        self.assertAlmostEqual(kernel.observation[4, 0, active, report_other], 0.96)
        full_other = OBSERVATION_NAMES.index("full_other")
        self.assertAlmostEqual(kernel.observation[4, 2, active, full_other], 0.96)

    def test_irreversible_send_certification_uses_complete_belief(self) -> None:
        ambiguous = build_fixture(self.config, "ambiguous_send")
        clear = build_fixture(self.config, "clear_send")
        ambiguous_allowed = certified_actions(ambiguous.kernel, ambiguous.initial_belief)
        clear_allowed = certified_actions(clear.kernel, clear.initial_belief)
        self.assertTrue(all(action not in ambiguous_allowed for action in SEND_ACTIONS))
        self.assertIn(ACTION_NAMES.index("send_q2_lee"), clear_allowed)
        self.assertNotIn(ACTION_NAMES.index("send_annual_lee"), clear_allowed)
        self.assertIn(SAFE_DRAFT_ACTION, ambiguous_allowed)

    def test_exact_step_updates_none_mass_without_fallback(self) -> None:
        fixture = build_fixture(self.config, "unknown_heavy")
        step = exact_step(
            fixture.kernel,
            fixture.initial_belief,
            ACTION_NAMES.index("ask_full_details"),
        )
        full_other = OBSERVATION_NAMES.index("full_other")
        posterior = step["posteriors"][full_other]
        self.assertGreater(float(posterior[4].sum()), float(fixture.initial_belief[4].sum()))
        self.assertAlmostEqual(float(sum(step["probabilities"])), 1.0)

    def test_planner_on_tiny_synthetic_kernel_not_registered_population(self) -> None:
        transition = np.zeros((2, 2, 2))
        transition[:, 1, 1] = 1.0
        transition[0, 0, 0] = 1.0
        transition[1, 0, 1] = 1.0
        observation = np.zeros((2, 2, 2, 2))
        observation[..., 1] = 1.0
        observation[:, 0, 0] = (0.5, 0.5)
        reward = np.zeros((2, 2, 2, 2))
        reward[:, 0, 0, 0] = -1.0
        reward[:, 1, 0, 1] = 2.0
        kernel = ClarificationKernel(
            hypothesis_names=("candidate", "none"),
            action_names=("inspect", "safe_draft"),
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
        policy = plan_exact(kernel, belief, 1)
        self.assertEqual(policy["selected_action"], 1)
        self.assertEqual(policy["value"], 2.0)


if __name__ == "__main__":
    unittest.main()
