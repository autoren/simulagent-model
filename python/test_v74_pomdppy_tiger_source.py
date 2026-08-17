#!/usr/bin/env python3
"""Structural tests for the frozen V74 pomdp-py Tiger adapter."""
from __future__ import annotations

import unittest

import numpy as np

from v74_pomdppy_tiger_source import (
    ACTION_NAMES,
    OBSERVATION_NAMES,
    PROJECT_BEACON_REWARD,
    SOURCE_OBSERVATION_ACCURACY,
    SOURCE_SAFE_OPEN_REWARD,
    SOURCE_TARGET_LISTEN_REWARD,
    SOURCE_TIGER_OPEN_REWARD,
    build_family,
    fixed_structural_policy,
    source_listen_transition,
    source_open_transition,
)


class V74TigerSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.family = build_family()
        self.kernel = self.family.kernel

    def test_01_shapes(self) -> None:
        self.assertEqual(self.kernel.transition.shape, (4, 2, 2))
        self.assertEqual(self.kernel.observation.shape, (2, 4, 2, 3))
        self.assertEqual(self.kernel.reward.shape, (4, 2, 2))

    def test_02_probability_normalization(self) -> None:
        self.assertTrue(np.allclose(self.kernel.transition.sum(axis=-1), 1.0))
        self.assertTrue(np.allclose(self.kernel.observation.sum(axis=-1), 1.0))

    def test_03_source_transitions(self) -> None:
        for name in ("calibrate_beacon", "listen_target"):
            self.assertTrue(
                np.array_equal(
                    self.kernel.transition[ACTION_NAMES.index(name)],
                    source_listen_transition(),
                )
            )
        for name in ("open_left", "open_right"):
            self.assertTrue(
                np.array_equal(
                    self.kernel.transition[ACTION_NAMES.index(name)],
                    source_open_transition(),
                )
            )

    def test_04_source_and_beacon_rewards(self) -> None:
        self.assertTrue(
            np.all(
                self.kernel.reward[ACTION_NAMES.index("calibrate_beacon")]
                == PROJECT_BEACON_REWARD
            )
        )
        self.assertTrue(
            np.all(
                self.kernel.reward[ACTION_NAMES.index("listen_target")]
                == SOURCE_TARGET_LISTEN_REWARD
            )
        )
        self.assertEqual(
            self.kernel.reward[ACTION_NAMES.index("open_left"), 0, 0],
            SOURCE_TIGER_OPEN_REWARD,
        )
        self.assertEqual(
            self.kernel.reward[ACTION_NAMES.index("open_left"), 1, 0],
            SOURCE_SAFE_OPEN_REWARD,
        )

    def test_05_configured_target_observation(self) -> None:
        listen = ACTION_NAMES.index("listen_target")
        self.assertAlmostEqual(
            self.kernel.observation[0, listen, 0, 0], SOURCE_OBSERVATION_ACCURACY
        )
        self.assertAlmostEqual(
            self.kernel.observation[1, listen, 0, 1], SOURCE_OBSERVATION_ACCURACY
        )

    def test_06_common_point_model_support(self) -> None:
        self.assertTrue(
            np.array_equal(
                self.kernel.observation[0] > 0.0,
                self.kernel.observation[1] > 0.0,
            )
        )

    def test_07_nonharvestable_beacon(self) -> None:
        calibrate = ACTION_NAMES.index("calibrate_beacon")
        self.assertLess(float(self.kernel.reward[calibrate].max()), 0.0)
        self.assertTrue(
            np.array_equal(self.kernel.transition[calibrate], source_listen_transition())
        )

    def test_08_uniform_joint_initial_belief(self) -> None:
        self.assertTrue(np.array_equal(self.family.initial_belief, np.full((2, 2), 0.25)))

    def test_09_open_observation_collapse(self) -> None:
        none = OBSERVATION_NAMES.index("none")
        for name in ("open_left", "open_right"):
            action = ACTION_NAMES.index(name)
            self.assertTrue(np.all(self.kernel.observation[:, action, :, none] == 1.0))
            self.assertTrue(np.all(self.kernel.observation[:, action, :, :2] == 0.0))

    def test_10_fixed_policy_schema(self) -> None:
        policy = fixed_structural_policy()
        self.assertEqual(ACTION_NAMES[policy["selected_action"]], "calibrate_beacon")
        self.assertEqual(len(policy["branches"]), 2)
        final = set()
        for child in policy["branches"].values():
            self.assertEqual(ACTION_NAMES[child["selected_action"]], "listen_target")
            final.update(ACTION_NAMES[row["selected_action"]] for row in child["branches"].values())
        self.assertEqual(final, {"open_left", "open_right"})


if __name__ == "__main__":
    unittest.main()
