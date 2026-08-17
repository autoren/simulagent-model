#!/usr/bin/env python3
"""Structural tests for the outcome-free V72 engineered fixtures."""
from __future__ import annotations

import unittest

import numpy as np

from v72_active_sensing_oracles import (
    ACTION_NAMES,
    STATE_NAMES,
    build_oracle,
    structural_diagnostics,
)


class V72ActiveSensingOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positive = build_oracle("positive")
        self.negative = build_oracle("negative_control")

    def test_initial_belief_is_independent_and_supported_on_ready_states(self) -> None:
        expected = np.zeros((2, 7))
        expected[:, 0:2] = 0.25
        np.testing.assert_allclose(self.positive.initial_belief, expected)

    def test_phase_transitions_are_deterministic(self) -> None:
        transition = self.positive.kernel.transition
        calibrate = ACTION_NAMES.index("calibrate")
        inspect = ACTION_NAMES.index("inspect")
        terminal = STATE_NAMES.index("terminal")
        self.assertEqual(int(np.argmax(transition[calibrate, 0])), 2)
        self.assertEqual(int(np.argmax(transition[calibrate, 1])), 3)
        self.assertEqual(int(np.argmax(transition[inspect, 0])), 4)
        self.assertEqual(int(np.argmax(transition[inspect, 2])), 4)
        self.assertEqual(int(np.argmax(transition[inspect, 3])), 5)
        for action in (2, 3):
            for state in range(terminal):
                self.assertEqual(int(np.argmax(transition[action, state])), terminal)

    def test_observation_models_have_identical_full_support(self) -> None:
        observation = self.positive.kernel.observation
        self.assertTrue(np.all(observation > 0.0))
        self.assertTrue(np.array_equal(observation[0] > 0.0, observation[1] > 0.0))
        self.assertAlmostEqual(float(observation.min()), 0.1)
        self.assertAlmostEqual(float(observation.max()), 0.9)

    def test_calibration_and_inspection_channels_swap_with_codebook(self) -> None:
        observation = self.positive.kernel.observation
        np.testing.assert_allclose(observation[0, 0, 2], (0.9, 0.1))
        np.testing.assert_allclose(observation[1, 0, 2], (0.1, 0.9))
        np.testing.assert_allclose(observation[0, 1, 4], (0.9, 0.1))
        np.testing.assert_allclose(observation[0, 1, 5], (0.1, 0.9))
        np.testing.assert_allclose(observation[:, 0, 4:6], 0.5)

    def test_positive_reward_depends_on_condition_and_repair(self) -> None:
        reward = self.positive.kernel.reward
        terminal = STATE_NAMES.index("terminal")
        self.assertEqual(float(reward[2, 0, terminal]), 10.0)
        self.assertEqual(float(reward[2, 1, terminal]), -20.0)
        self.assertEqual(float(reward[3, 0, terminal]), -20.0)
        self.assertEqual(float(reward[3, 1, terminal]), 10.0)

    def test_negative_control_has_dominant_repair(self) -> None:
        reward = self.negative.kernel.reward
        terminal = STATE_NAMES.index("terminal")
        np.testing.assert_allclose(reward[2, :terminal, terminal], 5.0)
        np.testing.assert_allclose(reward[3, :terminal, terminal], 4.0)

    def test_structural_information_and_support_diagnostics(self) -> None:
        diagnostics = structural_diagnostics(self.positive)
        self.assertGreater(diagnostics["calibration_mutual_information_nats"], 0.1)
        self.assertGreater(
            diagnostics["inspection_state_mutual_information_given_codebook_nats"],
            0.1,
        )
        self.assertTrue(diagnostics["point_model_supports_identical"])
        self.assertEqual(diagnostics["point_model_on_support_rate"], 1.0)
        self.assertEqual(diagnostics["fallback_count"], 0)
        self.assertTrue(diagnostics["calibration_after_inspection_is_uninformative"])

    def test_arrays_are_immutable(self) -> None:
        self.assertFalse(self.positive.kernel.transition.flags.writeable)
        self.assertFalse(self.positive.kernel.observation.flags.writeable)
        self.assertFalse(self.positive.kernel.reward.flags.writeable)
        self.assertFalse(self.positive.initial_belief.flags.writeable)


if __name__ == "__main__":
    unittest.main()
