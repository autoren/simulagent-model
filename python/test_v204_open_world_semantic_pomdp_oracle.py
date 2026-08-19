from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from v204_open_world_semantic_pomdp_oracle import (
    ACTION_NAMES,
    LATENT_NAMES,
    OBSERVATION_NAMES,
    SemanticKernel,
    STATE_NAMES,
    build_kernel,
    exact_step,
    plan_joint,
)


CONFIG = json.loads(Path("configs/v204-open-world-semantic-pomdp-oracle.json").read_text())


class V204KernelTest(unittest.TestCase):
    def test_kernel_and_prior_are_normalized_with_common_support(self) -> None:
        kernel, belief = build_kernel(CONFIG)
        self.assertAlmostEqual(float(belief.sum()), 1.0)
        self.assertTrue(np.allclose(kernel.transition.sum(axis=-1), 1.0))
        self.assertTrue(np.allclose(kernel.observation.sum(axis=-1), 1.0))
        support = kernel.observation > 0.0
        self.assertTrue(all(np.array_equal(support[0], support[index]) for index in range(1, len(LATENT_NAMES))))

    def test_repair_is_rewardless_until_settlement(self) -> None:
        kernel, belief = build_kernel(CONFIG)
        for action_name in ("repair_A", "repair_B"):
            self.assertEqual(exact_step(kernel, belief, ACTION_NAMES.index(action_name))["reward"], 0.0)
        settle = ACTION_NAMES.index("settle")
        terminal = STATE_NAMES.index("terminal")
        self.assertEqual(kernel.reward[settle, STATE_NAMES.index("pending_good"), terminal], 10.0)
        self.assertEqual(kernel.reward[settle, STATE_NAMES.index("pending_bad"), terminal], -30.0)

    def test_green_is_supported_by_every_semantic_hypothesis(self) -> None:
        kernel, _ = build_kernel(CONFIG)
        green = OBSERVATION_NAMES.index("green")
        calibrate = ACTION_NAMES.index("calibrate")
        ready_a = STATE_NAMES.index("ready_A")
        self.assertTrue(np.all(kernel.observation[:, calibrate, ready_a, green] > 0.0))

    def test_locked_point_and_closed_world_comparator_shapes_are_supported(self) -> None:
        kernel, belief = build_kernel(CONFIG)
        for latent_count in (1, 2):
            comparator = SemanticKernel(
                transition=kernel.transition.copy(),
                observation=kernel.observation[:latent_count].copy(),
                reward=kernel.reward.copy(),
                discount=kernel.discount,
            )
            comparator_belief = belief[:latent_count].copy()
            comparator_belief /= comparator_belief.sum()
            policy = plan_joint(comparator, comparator_belief, 1)
            self.assertIn(policy["selected_action"], range(len(ACTION_NAMES)))


if __name__ == "__main__":
    unittest.main()
