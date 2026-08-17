#!/usr/bin/env python3
"""Synthetic implementation tests for exact V71 planning and controls."""
from __future__ import annotations

import unittest

import numpy as np

from v71_exact_planning import (
    SensorCodebookKernel,
    best_open_loop_sequence,
    evaluate_policy_exact,
    map_control,
    plan_exact,
    plan_myopic,
    posterior_sampling_control,
)


def synthetic_kernel() -> SensorCodebookKernel:
    transition = np.asarray(
        [
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.4, 0.6], [0.9, 0.1]],
        ]
    )
    source = np.asarray(
        [
            [[0.85, 0.15], [0.25, 0.75]],
            [[0.65, 0.35], [0.1, 0.9]],
        ]
    )
    observation = np.stack((source, source[..., ::-1]))
    reward = np.asarray(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.4, 0.0], [0.4, 0.0]],
        ]
    )
    return SensorCodebookKernel(
        action_names=("a", "b"),
        observation_names=("x", "y"),
        state_names=("s0", "s1"),
        transition=transition,
        observation=observation,
        reward=reward,
        discount=0.9,
    )


def scalar_value(kernel: SensorCodebookKernel, belief: np.ndarray, horizon: int) -> float:
    if horizon == 0:
        return 0.0
    values = []
    for action in range(len(kernel.action_names)):
        immediate = 0.0
        joint = np.zeros((2, 2, 2))
        for latent in range(2):
            for state in range(2):
                for successor in range(2):
                    edge = belief[latent, state] * kernel.transition[action, state, successor]
                    immediate += edge * kernel.reward[action, state, successor]
                    for observation in range(2):
                        joint[latent, successor, observation] += (
                            edge * kernel.observation[latent, action, successor, observation]
                        )
        continuation = 0.0
        if horizon > 1:
            for observation in range(2):
                probability = float(joint[:, :, observation].sum())
                if probability > 0.0:
                    continuation += probability * scalar_value(
                        kernel, joint[:, :, observation] / probability, horizon - 1
                    )
        values.append(immediate + kernel.discount * continuation)
    return float(max(values))


class V71ExactPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = synthetic_kernel()
        self.belief = np.asarray([[0.35, 0.15], [0.1, 0.4]])

    def test_vectorized_exact_value_matches_scalar_reference(self) -> None:
        decision = plan_exact(self.kernel, self.belief, 3)
        self.assertAlmostEqual(decision["value"], scalar_value(self.kernel, self.belief, 3))

    def test_exact_value_dominates_every_frozen_control(self) -> None:
        exact = plan_exact(self.kernel, self.belief, 3)
        mapped = map_control(self.kernel, self.belief, 3)
        sampled = posterior_sampling_control(self.kernel, self.belief, 3)
        open_loop = best_open_loop_sequence(self.kernel, self.belief, 3)
        myopic = plan_myopic(self.kernel, self.belief, 3)
        myopic_value = evaluate_policy_exact(self.kernel, self.belief, myopic, 3)
        for value in (mapped["value"], sampled["value"], open_loop["value"], myopic_value):
            self.assertGreaterEqual(exact["value"] + 1e-12, value)

    def test_point_controls_are_fallback_free(self) -> None:
        mapped = map_control(self.kernel, self.belief, 3)
        sampled = posterior_sampling_control(self.kernel, self.belief, 3)
        self.assertTrue(mapped["on_support"])
        self.assertTrue(sampled["on_support"])
        self.assertAlmostEqual(sum(sampled["root_action_distribution"]), 1.0)

    def test_map_tie_uses_canonical_latent(self) -> None:
        tied = np.asarray([[0.3, 0.2], [0.1, 0.4]])
        mapped = map_control(self.kernel, tied, 2)
        self.assertEqual(mapped["latent"], 0)

    def test_action_tie_uses_source_order(self) -> None:
        zero_reward = SensorCodebookKernel(
            action_names=self.kernel.action_names,
            observation_names=self.kernel.observation_names,
            state_names=self.kernel.state_names,
            transition=self.kernel.transition,
            observation=self.kernel.observation,
            reward=np.zeros_like(self.kernel.reward),
            discount=self.kernel.discount,
        )
        self.assertEqual(plan_exact(zero_reward, self.belief, 2)["selected_action"], 0)


if __name__ == "__main__":
    unittest.main()
