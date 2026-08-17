#!/usr/bin/env python3
"""Outcome-free source and structure tests for the V72 RockSample export."""
from __future__ import annotations

import math
import unittest

import numpy as np

from v72_rocksample_source import (
    ACTION_NAMES,
    RockSampleState,
    build_family,
    enumerate_states,
    source_check_distribution,
    source_reward,
    state_index,
    structural_resource_metrics,
    successor,
)


class V72RockSampleSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.family = build_family()

    def test_dimensions_match_frozen_blueprint(self) -> None:
        self.assertEqual(self.family.kernel.transition.shape, (7, 17, 17))
        self.assertEqual(self.family.kernel.observation.shape, (2, 7, 17, 3))
        self.assertEqual(self.family.kernel.reward.shape, (7, 17, 17))

    def test_state_order_matches_source_index_formula(self) -> None:
        states = enumerate_states()
        self.assertEqual(states[0], RockSampleState(1, 1, False, False))
        self.assertEqual(states[3], RockSampleState(2, 2, False, False))
        self.assertEqual(states[4], RockSampleState(1, 1, True, False))
        self.assertEqual(states[12], RockSampleState(1, 1, True, True))
        self.assertIsNone(states[16])
        for index, state in enumerate(states):
            self.assertEqual(state_index(state), index)

    def test_initial_belief_fixes_reference_and_splits_target_and_codebook(self) -> None:
        belief = self.family.initial_belief
        self.assertEqual(int(np.count_nonzero(belief)), 4)
        self.assertTrue(np.allclose(belief[belief > 0.0], 0.25))
        for latent, state in zip(*np.nonzero(belief), strict=True):
            self.assertTrue(self.family.states[state].reference_good)
            self.assertEqual((self.family.states[state].x, self.family.states[state].y), (2, 1))

    def test_movement_clamps_and_east_exits(self) -> None:
        state = RockSampleState(2, 1, True, False)
        self.assertEqual(successor(state, 1), RockSampleState(2, 2, True, False))
        self.assertEqual(successor(state, 3), state)
        self.assertIsNone(successor(state, 2))
        self.assertIsNone(successor(None, 4))

    def test_sampling_changes_only_colocated_rock_to_bad(self) -> None:
        reference = RockSampleState(1, 1, True, True)
        target = RockSampleState(2, 2, True, True)
        self.assertEqual(successor(reference, 0), RockSampleState(1, 1, False, True))
        self.assertEqual(successor(target, 0), RockSampleState(2, 2, True, False))

    def test_source_rewards_match_fixed_parameters(self) -> None:
        self.assertEqual(source_reward(RockSampleState(2, 2, True, True), 0), 10.0)
        self.assertEqual(source_reward(RockSampleState(2, 2, True, False), 0), -10.0)
        self.assertEqual(source_reward(RockSampleState(2, 1, True, False), 2), 5.0)
        self.assertEqual(source_reward(RockSampleState(2, 1, True, False), 6), -0.5)

    def test_source_sensor_formula_is_distance_and_state_dependent(self) -> None:
        state = RockSampleState(2, 1, True, False)
        expected = 0.5 * (1.0 + math.exp(-math.log(2.0)))
        np.testing.assert_allclose(source_check_distribution(state, 5), (expected, 1-expected, 0.0))
        np.testing.assert_allclose(source_check_distribution(state, 6), (1-expected, expected, 0.0))

    def test_codebooks_share_support_and_swap_check_labels(self) -> None:
        observation = self.family.kernel.observation
        self.assertTrue(np.array_equal(observation[0] > 0.0, observation[1] > 0.0))
        for action in (5, 6):
            np.testing.assert_allclose(observation[0, action, :, 0], observation[1, action, :, 1])
            np.testing.assert_allclose(observation[0, action, :, 1], observation[1, action, :, 0])
        self.assertGreaterEqual(float(observation[:, 5:, :-1, :2].min()), 0.1)
        np.testing.assert_allclose(observation[:, 5:, -1, :2], 0.0)
        np.testing.assert_allclose(observation[:, 5:, -1, 2], 1.0)

    def test_noncheck_actions_preserve_none_observation(self) -> None:
        observation = self.family.kernel.observation
        np.testing.assert_allclose(observation[:, :5, :, :2], 0.0)
        np.testing.assert_allclose(observation[:, :5, :, 2], 1.0)

    def test_resource_metrics_are_bounded_without_planning(self) -> None:
        metrics = structural_resource_metrics(self.family)
        self.assertEqual(metrics["states"], 17)
        self.assertEqual(metrics["actions"], 7)
        self.assertEqual(metrics["observations"], 3)
        self.assertLess(metrics["dense_kernel_bytes"], 100000)
        self.assertEqual(metrics["exact_bellman_node_upper_bound"], 820)


if __name__ == "__main__":
    unittest.main()
