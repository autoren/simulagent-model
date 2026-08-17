#!/usr/bin/env python3
"""Synthetic tests for V68r1 posterior-sampling union-support totalization."""
from __future__ import annotations

import unittest

import numpy as np

from test_v68_multi_environment_exact import synthetic_model
from v66_bayes_adaptive_reward import (
    StaticKernel,
    persistent_posterior_sampling_mixture,
    point_model_kernel_and_belief,
)
from v68_multi_environment_exact import build_command_channel_family
from v68r1_posterior_sampling import (
    evaluate_point_policy_with_total_fallback,
    totalized_persistent_posterior_sampling_mixture,
)


def support_mismatch_kernel() -> tuple[StaticKernel, np.ndarray]:
    transitions = np.asarray(
        [
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ],
            [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ],
        ]
    )
    observations = np.broadcast_to(
        np.asarray([[1.0, 0.0], [0.0, 1.0]])[None, :, :], (2, 2, 2)
    ).copy()
    rewards = np.zeros((2, 2, 2))
    rewards[0, :, 0] = 1.0
    rewards[1, :, 1] = 2.0
    kernel = StaticKernel(
        action_names=("fallback", "other"),
        observation_names=("zero", "one"),
        state_names=("zero", "one"),
        canonical_actions=(0, 1),
        transitions=transitions,
        observations=observations,
        rewards=rewards,
        discount=0.9,
        identities=np.asarray([0, 1]),
        thetas=np.asarray([0.8, 0.8]),
    )
    belief = np.asarray([[0.5, 0.0], [0.5, 0.0]])
    return kernel, belief


class V68r1PosteriorSamplingTests(unittest.TestCase):
    def test_support_mismatch_is_total_and_reports_entry_probability(self) -> None:
        kernel, belief = support_mismatch_kernel()
        result = totalized_persistent_posterior_sampling_mixture(
            kernel, belief, 3, points=2, offset=0.25
        )
        self.assertTrue(np.isfinite(result["value"]))
        self.assertGreater(result["off_support_branch_count"], 0)
        self.assertGreater(result["expected_off_support_entry_probability"], 0.0)
        self.assertEqual(result["fallback_action"], 0)
        self.assertFalse(result["off_support_model_resampling"])
        self.assertFalse(result["epsilon_smoothing"])

    def test_point_evaluator_uses_first_canonical_action_as_fallback(self) -> None:
        kernel, belief = support_mismatch_kernel()
        point_kernel, point_belief, _ = point_model_kernel_and_belief(kernel, belief, 0)
        result = evaluate_point_policy_with_total_fallback(
            kernel,
            belief,
            point_kernel,
            point_belief,
            2,
            fallback_action=0,
        )
        self.assertEqual(result["fallback_action"], kernel.canonical_actions[0])
        with self.assertRaisesRegex(ValueError, "first frozen canonical action"):
            evaluate_point_policy_with_total_fallback(
                kernel,
                belief,
                point_kernel,
                point_belief,
                2,
                fallback_action=1,
            )

    def test_no_policy_branch_is_needed_after_final_action(self) -> None:
        kernel, belief = support_mismatch_kernel()
        point_kernel, point_belief, _ = point_model_kernel_and_belief(kernel, belief, 0)
        result = evaluate_point_policy_with_total_fallback(
            kernel,
            belief,
            point_kernel,
            point_belief,
            1,
            fallback_action=0,
        )
        self.assertEqual(result["off_support_branch_count"], 0)
        self.assertEqual(result["expected_off_support_entry_probability"], 0.0)

    def test_full_support_matches_original_persistent_control(self) -> None:
        family = build_command_channel_family(
            synthetic_model(), ("a", "b", "c"), quadrature_nodes=7
        )
        expected = persistent_posterior_sampling_mixture(
            family.kernel,
            family.initial_belief,
            3,
            points=5,
            offset=0.1,
        )
        actual = totalized_persistent_posterior_sampling_mixture(
            family.kernel,
            family.initial_belief,
            3,
            points=5,
            offset=0.1,
        )
        self.assertAlmostEqual(actual["value"], expected["value"], places=14)
        self.assertTrue(
            np.allclose(
                actual["root_action_distribution"],
                expected["root_action_distribution"],
                atol=1e-14,
                rtol=0.0,
            )
        )
        self.assertEqual(actual["off_support_branch_count"], 0)
        self.assertEqual(actual["expected_off_support_entry_probability"], 0.0)

    def test_point_and_exact_alphabets_must_match(self) -> None:
        kernel, belief = support_mismatch_kernel()
        point_kernel, point_belief, _ = point_model_kernel_and_belief(kernel, belief, 0)
        wrong = StaticKernel(
            action_names=point_kernel.action_names,
            observation_names=("x", "y"),
            state_names=point_kernel.state_names,
            canonical_actions=point_kernel.canonical_actions,
            transitions=point_kernel.transitions,
            observations=point_kernel.observations,
            rewards=point_kernel.rewards,
            discount=point_kernel.discount,
            identities=point_kernel.identities,
            thetas=point_kernel.thetas,
        )
        with self.assertRaisesRegex(ValueError, "observation alphabets differ"):
            evaluate_point_policy_with_total_fallback(
                kernel, belief, wrong, point_belief, 2, fallback_action=0
            )


if __name__ == "__main__":
    unittest.main()
