#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from v71_exact_planning import (
    best_open_loop_sequence,
    evaluate_policy_exact,
    finite_horizon_return_scale,
)
from v75_nova_paint_source import (
    ACTION_NAMES,
    OBSERVATION_NAMES,
    SOURCE_ACTION_NAMES,
    SOURCE_COMMIT,
    SOURCE_INSPECTION_ACCURACY,
    SOURCE_SHA256,
    STATE_NAMES,
    build_family,
    fixed_structural_policy,
    load_source_model,
    structural_diagnostics,
    structural_resource_metrics,
)


class V75NovaPaintSourceTests(unittest.TestCase):
    def test_01_pinned_source_metadata(self) -> None:
        model = load_source_model()
        self.assertEqual(len(SOURCE_COMMIT), 40)
        self.assertEqual(len(SOURCE_SHA256), 64)
        self.assertEqual(model.states, STATE_NAMES)
        self.assertEqual(model.actions, SOURCE_ACTION_NAMES)
        self.assertEqual(model.discount, 0.95)

    def test_02_source_arrays_normalize(self) -> None:
        model = load_source_model()
        self.assertAlmostEqual(float(model.initial.sum()), 1.0)
        np.testing.assert_allclose(model.transition.sum(axis=2), 1.0)
        np.testing.assert_allclose(model.observation.sum(axis=2), 1.0)

    def test_03_source_transitions_are_preserved(self) -> None:
        model = load_source_model()
        family = build_family()
        for target, source in zip(ACTION_NAMES[1:], SOURCE_ACTION_NAMES, strict=True):
            np.testing.assert_array_equal(
                family.kernel.transition[ACTION_NAMES.index(target)],
                model.transition[model.actions.index(source)],
            )

    def test_04_source_rewards_are_preserved(self) -> None:
        model = load_source_model()
        family = build_family()
        for target, source in zip(ACTION_NAMES[1:], SOURCE_ACTION_NAMES, strict=True):
            np.testing.assert_array_equal(
                family.kernel.reward[ACTION_NAMES.index(target)],
                model.reward[model.actions.index(source)],
            )

    def test_05_sensor_codebooks_use_source_accuracy(self) -> None:
        kernel = build_family().kernel
        calibrate = ACTION_NAMES.index("calibrate_beacon")
        inspect = ACTION_NAMES.index("inspect_target")
        self.assertEqual(kernel.observation[0, calibrate, 0, 0], SOURCE_INSPECTION_ACCURACY)
        self.assertEqual(kernel.observation[1, calibrate, 0, 1], SOURCE_INSPECTION_ACCURACY)
        self.assertEqual(kernel.observation[0, inspect, 0, 0], SOURCE_INSPECTION_ACCURACY)
        self.assertEqual(kernel.observation[1, inspect, 0, 1], SOURCE_INSPECTION_ACCURACY)

    def test_06_kernel_is_normalized_with_common_support(self) -> None:
        kernel = build_family().kernel
        np.testing.assert_allclose(kernel.transition.sum(axis=2), 1.0)
        np.testing.assert_allclose(kernel.observation.sum(axis=3), 1.0)
        np.testing.assert_array_equal(
            kernel.observation[0] > 0.0, kernel.observation[1] > 0.0
        )

    def test_07_initial_joint_belief_matches_source(self) -> None:
        family = build_family()
        model = load_source_model()
        np.testing.assert_array_equal(family.initial_belief.sum(axis=0), model.initial)
        np.testing.assert_array_equal(family.initial_belief.sum(axis=1), (0.5, 0.5))

    def test_08_beacon_is_nonharvestable_identity(self) -> None:
        family = build_family()
        action = ACTION_NAMES.index("calibrate_beacon")
        np.testing.assert_array_equal(family.kernel.transition[action], np.eye(4))
        np.testing.assert_array_equal(family.kernel.reward[action], np.zeros((4, 4)))

    def test_09_fixed_policy_reproduces_economic_lower_bound(self) -> None:
        family = build_family()
        value = evaluate_policy_exact(
            family.kernel, family.initial_belief, fixed_structural_policy(), 4
        )
        self.assertAlmostEqual(value, 0.1663984375, places=12)

    def test_10_open_loop_and_resources_match_registration(self) -> None:
        family = build_family()
        open_loop = best_open_loop_sequence(family.kernel, family.initial_belief, 4)
        self.assertAlmostEqual(open_loop["value"], 0.0, places=12)
        self.assertEqual(open_loop["sequence_count"], 625)
        self.assertAlmostEqual(finite_horizon_return_scale(family.kernel, 4), 7.41975)
        self.assertEqual(structural_resource_metrics()["exact_bellman_node_upper_bound"], 400)
        diagnostics = structural_diagnostics()
        self.assertTrue(diagnostics["source_array_parity"])
        self.assertTrue(diagnostics["point_model_supports_identical"])
        self.assertEqual(OBSERVATION_NAMES[-1], "none")


if __name__ == "__main__":
    unittest.main()
