#!/usr/bin/env python3
"""Synthetic tests for the V69 dominant-remapping family."""
from __future__ import annotations

import unittest

import numpy as np

from test_v68_multi_environment_exact import synthetic_model
from v66_bayes_adaptive_reward import plan_bayes_adaptive
from v68_multi_environment_exact import build_command_channel_family
from v69_dominant_remapping import build_dominant_remapping_family


class V69DominantRemappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = synthetic_model()
        self.family = build_dominant_remapping_family(
            self.model, ("a", "b", "c"), quadrature_nodes=9
        )

    def test_transition_formula_places_theta_on_remapped_action(self) -> None:
        node = 4
        theta = float(self.family.theta[node])
        action = self.model.actions.index("a")
        for identity in range(2):
            remapped = int(self.family.permutations[identity, action])
            expected = (
                theta * self.model.transition[remapped]
                + (1.0 - theta) * self.model.transition[action]
            )
            actual = self.family.kernel.transitions[identity * 9 + node, action]
            self.assertTrue(np.allclose(actual, expected, atol=1e-14, rtol=0.0))

    def test_family_is_not_the_V68_nominal_dominant_family(self) -> None:
        old = build_command_channel_family(
            self.model, ("a", "b", "c"), quadrature_nodes=9
        )
        self.assertFalse(
            np.allclose(
                self.family.kernel.transitions,
                old.kernel.transitions,
                atol=1e-14,
                rtol=0.0,
            )
        )

    def test_probabilities_and_prior_normalize(self) -> None:
        self.assertTrue(
            np.allclose(self.family.kernel.transitions.sum(axis=-1), 1.0, atol=1e-12)
        )
        self.assertAlmostEqual(float(self.family.initial_belief.sum()), 1.0, places=14)
        self.assertTrue(np.all(self.family.theta >= 0.6))
        self.assertTrue(np.all(self.family.theta <= 0.95))

    def test_arrays_are_immutable(self) -> None:
        self.assertFalse(self.family.kernel.transitions.flags.writeable)
        self.assertFalse(self.family.initial_belief.flags.writeable)
        self.assertFalse(self.family.permutations.flags.writeable)

    def test_exact_planner_returns_finite_policy(self) -> None:
        result = plan_bayes_adaptive(
            self.family.kernel, self.family.initial_belief, 3
        )
        self.assertTrue(np.isfinite(result["value"]))
        self.assertIn(result["selected_action"], self.family.kernel.canonical_actions)


if __name__ == "__main__":
    unittest.main()
