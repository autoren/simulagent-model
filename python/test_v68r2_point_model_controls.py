#!/usr/bin/env python3
"""Synthetic tests for V68r2 all-point-model-control totalization."""
from __future__ import annotations

import unittest

import numpy as np

from test_v68_multi_environment_exact import synthetic_model
from test_v68r1_posterior_sampling import support_mismatch_kernel
from v66_bayes_adaptive_reward import map_model_policy
from v68_multi_environment_exact import build_command_channel_family
from v68r2_point_model_controls import totalized_map_model_policy


class V68r2PointModelControlTests(unittest.TestCase):
    def test_support_mismatch_map_is_total_and_diagnostic(self) -> None:
        kernel, belief = support_mismatch_kernel()
        result = totalized_map_model_policy(kernel, belief, 3)
        self.assertTrue(np.isfinite(result["exact_environment_value"]))
        self.assertGreater(result["off_support_branch_count"], 0)
        self.assertGreater(result["expected_off_support_entry_probability"], 0.0)
        self.assertEqual(result["fallback_action"], kernel.canonical_actions[0])

    def test_full_support_matches_original_map_control(self) -> None:
        family = build_command_channel_family(
            synthetic_model(), ("a", "b", "c"), quadrature_nodes=9
        )
        expected = map_model_policy(family.kernel, family.initial_belief, 3)
        actual = totalized_map_model_policy(family.kernel, family.initial_belief, 3)
        self.assertEqual(actual["static_index"], expected["static_index"])
        self.assertEqual(
            actual["policy"]["selected_action"], expected["policy"]["selected_action"]
        )
        self.assertEqual(
            actual["policy"]["optimal_actions"], expected["policy"]["optimal_actions"]
        )
        self.assertAlmostEqual(
            actual["policy"]["value"], expected["policy"]["value"], places=14
        )
        self.assertTrue(
            np.allclose(
                actual["policy"]["q_values"],
                expected["policy"]["q_values"],
                atol=1e-14,
                rtol=0.0,
            )
        )
        self.assertAlmostEqual(
            actual["exact_environment_value"],
            expected["exact_environment_value"],
            places=14,
        )
        self.assertEqual(actual["off_support_branch_count"], 0)

    def test_map_tie_uses_unchanged_first_argmax(self) -> None:
        kernel, belief = support_mismatch_kernel()
        self.assertEqual(belief.sum(axis=1).tolist(), [0.5, 0.5])
        result = totalized_map_model_policy(kernel, belief, 2)
        self.assertEqual(result["static_index"], 0)
        self.assertEqual(result["static_mass"], 0.5)

    def test_fallback_semantics_are_frozen(self) -> None:
        kernel, belief = support_mismatch_kernel()
        result = totalized_map_model_policy(kernel, belief, 3)
        self.assertTrue(result["point_model_persists_on_supported_histories"])
        self.assertTrue(result["off_support_fallback_is_open_loop"])
        self.assertFalse(result["off_support_model_reselection"])
        self.assertFalse(result["epsilon_smoothing"])


if __name__ == "__main__":
    unittest.main()
