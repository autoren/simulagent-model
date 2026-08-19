from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from v205_terminally_proper_open_world_semantic_pomdp import (
    FixedStageKernel,
    STAGE_NAMES,
    allowed_actions,
    build_kernel,
    repair_return,
    sensing_step,
)


CONFIG = json.loads(Path("configs/v205-terminally-proper-open-world-semantic-pomdp.json").read_text())


class V205KernelTest(unittest.TestCase):
    def test_channels_and_prior_are_normalized_with_positive_common_support(self) -> None:
        kernel, belief = build_kernel(CONFIG)
        self.assertAlmostEqual(float(belief.sum()), 1.0)
        self.assertTrue(np.allclose(kernel.calibration.sum(axis=-1), 1.0))
        self.assertTrue(np.allclose(kernel.inspection.sum(axis=-1), 1.0))
        self.assertTrue(np.all(kernel.calibration > 0.0))
        self.assertTrue(np.all(kernel.inspection > 0.0))

    def test_sensing_posteriors_are_normalized(self) -> None:
        kernel, belief = build_kernel(CONFIG)
        for action in ("calibrate", "inspect"):
            step = sensing_step(kernel, belief, action)
            self.assertAlmostEqual(float(step["probabilities"].sum()), 1.0)
            self.assertTrue(all(np.isclose(posterior.sum(), 1.0) for posterior in step["posteriors"].values()))

    def test_repair_has_zero_immediate_reward_and_mandatory_settlement(self) -> None:
        kernel, belief = build_kernel(CONFIG)
        for action in ("repair_A", "repair_B"):
            result = repair_return(kernel, belief, action)
            self.assertEqual(result["immediate_reward"], 0.0)
            self.assertTrue(result["mandatory_automatic_settlement"])
            self.assertAlmostEqual(result["total_return"], -10.0)

    def test_final_controllable_stage_cannot_sense_or_escape_settlement(self) -> None:
        stage = STAGE_NAMES.index("POST_INSPECTION")
        actions = allowed_actions(CONFIG, stage)
        self.assertEqual(actions, ("repair_A", "repair_B", "defer"))
        self.assertNotIn("calibrate", actions)
        self.assertNotIn("inspect", actions)

    def test_locked_one_and_two_hypothesis_comparator_shapes_are_valid(self) -> None:
        kernel, _ = build_kernel(CONFIG)
        for latent_count in (1, 2):
            comparator = FixedStageKernel(
                calibration=kernel.calibration[:latent_count].copy(),
                inspection=kernel.inspection[:latent_count].copy(),
                sensing_cost=kernel.sensing_cost,
                deferral_reward=kernel.deferral_reward,
                repair_immediate_reward=kernel.repair_immediate_reward,
                correct_settlement_reward=kernel.correct_settlement_reward,
                wrong_settlement_reward=kernel.wrong_settlement_reward,
                discount=kernel.discount,
            )
            self.assertEqual(comparator.calibration.shape[0], latent_count)


if __name__ == "__main__":
    unittest.main()
