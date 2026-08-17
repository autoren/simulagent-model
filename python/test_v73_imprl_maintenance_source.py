#!/usr/bin/env python3
"""Structural tests for the frozen V73 IMPRL maintenance adapter."""
from __future__ import annotations

import unittest

import numpy as np

from v73_imprl_maintenance_source import (
    ACTION_NAMES,
    OBSERVATION_NAMES,
    PROJECTED_FAILURE_REWARD,
    SOURCE_DETERIORATION,
    SOURCE_INITIAL_BELIEF,
    SOURCE_INSPECTION_REWARD,
    SOURCE_MOBILISATION_REWARD,
    SOURCE_REPLACEMENT_REWARD,
    STATE_NAMES,
    build_family,
    fixed_structural_policy,
    known_reward_diagnostics,
    source_inspection_model,
    source_replacement_transition,
    structural_resource_metrics,
)


class V73MaintenanceSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.family = build_family()
        self.kernel = self.family.kernel

    def test_01_dimensions_and_normalization(self) -> None:
        self.assertEqual(self.kernel.transition.shape, (4, 3, 3))
        self.assertEqual(self.kernel.observation.shape, (2, 4, 3, 4))
        self.assertTrue(np.allclose(self.kernel.transition.sum(axis=-1), 1.0))
        self.assertTrue(np.allclose(self.kernel.observation.sum(axis=-1), 1.0))

    def test_02_source_deterioration_rows_are_bound(self) -> None:
        for action_name in ("do_nothing", "inspect_target", "calibrate_beacon"):
            self.assertTrue(
                np.array_equal(
                    self.kernel.transition[ACTION_NAMES.index(action_name)],
                    SOURCE_DETERIORATION,
                )
            )

    def test_03_source_replacement_transition_is_bound(self) -> None:
        expected = np.tile(SOURCE_DETERIORATION[0], (3, 1))
        self.assertTrue(np.array_equal(source_replacement_transition(), expected))
        self.assertTrue(
            np.array_equal(
                self.kernel.transition[ACTION_NAMES.index("replace_target")], expected
            )
        )

    def test_04_source_and_projected_rewards_are_bound(self) -> None:
        failed = STATE_NAMES.index("failed")
        replace = ACTION_NAMES.index("replace_target")
        inspect = ACTION_NAMES.index("inspect_target")
        self.assertEqual(
            self.kernel.reward[replace, 0, 0],
            SOURCE_REPLACEMENT_REWARD + SOURCE_MOBILISATION_REWARD,
        )
        self.assertEqual(
            self.kernel.reward[inspect, 0, 0],
            SOURCE_INSPECTION_REWARD + SOURCE_MOBILISATION_REWARD,
        )
        self.assertEqual(
            self.kernel.reward[ACTION_NAMES.index("do_nothing"), failed, failed],
            PROJECTED_FAILURE_REWARD,
        )

    def test_05_canonical_inspection_matches_source_rows(self) -> None:
        inspect = ACTION_NAMES.index("inspect_target")
        self.assertTrue(
            np.array_equal(self.kernel.observation[0, inspect, :, :3], source_inspection_model())
        )

    def test_06_reversed_codebook_swaps_only_labels_zero_and_one(self) -> None:
        inspect = ACTION_NAMES.index("inspect_target")
        canonical = self.kernel.observation[0, inspect]
        reversed_model = self.kernel.observation[1, inspect]
        self.assertTrue(np.array_equal(reversed_model[:, 0], canonical[:, 1]))
        self.assertTrue(np.array_equal(reversed_model[:, 1], canonical[:, 0]))
        self.assertTrue(np.array_equal(reversed_model[:, 2:], canonical[:, 2:]))

    def test_07_initial_joint_belief_is_source_prior_times_latent_prior(self) -> None:
        self.assertTrue(
            np.array_equal(self.family.initial_belief[0], 0.5 * SOURCE_INITIAL_BELIEF)
        )
        self.assertTrue(
            np.array_equal(self.family.initial_belief[1], 0.5 * SOURCE_INITIAL_BELIEF)
        )
        self.assertAlmostEqual(float(self.family.initial_belief.sum()), 1.0)

    def test_08_point_model_supports_are_identical(self) -> None:
        support = self.kernel.observation > 0.0
        self.assertTrue(np.array_equal(support[0], support[1]))

    def test_09_beacon_is_nonharvestable_and_noncontrolling(self) -> None:
        diagnostics = known_reward_diagnostics()
        self.assertEqual(diagnostics["strictly_positive_immediate_reward_count"], 0)
        self.assertFalse(diagnostics["calibration_beacon_harvestable"])
        self.assertTrue(diagnostics["calibration_transition_matches_source_do_nothing"])
        self.assertLess(diagnostics["calibration_maximum_immediate_reward"], 0.0)

    def test_10_fixed_policy_and_resource_envelope_are_finite(self) -> None:
        policy = fixed_structural_policy()
        self.assertEqual(
            ACTION_NAMES[policy["selected_action"]], "calibrate_beacon"
        )
        self.assertEqual(set(policy["branches"]), {0, 1})
        self.assertEqual(structural_resource_metrics()["exact_bellman_node_upper_bound"], 2801)
        self.assertEqual(OBSERVATION_NAMES[-1], "none")


if __name__ == "__main__":
    unittest.main()
