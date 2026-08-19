from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from v209_controlled_language_observation_pomdp import (
    CLARIFICATION_ACTIONS,
    CONTROL_ACTIONS,
    OBSERVATION_NAMES,
    STAGE_NAMES,
    allowed_actions,
    build_kernel,
    clarification_cost_matrix,
    clarification_likelihood,
    clarification_step,
    control_return,
    recover_semantic_policy,
    render_policy,
)


CONFIG = json.loads(Path("configs/v209-controlled-language-observation-pomdp.json").read_text())


class V209LanguageKernelTest(unittest.TestCase):
    def test_prior_and_every_frozen_channel_are_normalized_with_common_support(self) -> None:
        kernel, belief = build_kernel(CONFIG)
        self.assertAlmostEqual(float(belief.sum()), 1.0)
        for channel in (kernel.reference, kernel.target, kernel.history_anchors):
            self.assertTrue(np.allclose(channel.sum(axis=-1), 1.0))
            self.assertTrue(np.all(channel > 0.0))
        for previous in range(len(OBSERVATION_NAMES)):
            value = clarification_likelihood(kernel, "ask_target", STAGE_NAMES.index("POST_REFERENCE"), (previous,))
            self.assertTrue(np.allclose(value.sum(axis=-1), 1.0))
            self.assertTrue(np.all(value > 0.0))

    def test_target_likelihood_and_cost_change_after_reference_history(self) -> None:
        kernel, _ = build_kernel(CONFIG)
        stage = STAGE_NAMES.index("POST_REFERENCE")
        alpha = clarification_likelihood(kernel, "ask_target", stage, (0,))
        beta = clarification_likelihood(kernel, "ask_target", stage, (1,))
        self.assertFalse(np.array_equal(alpha, beta))
        alpha_cost = clarification_cost_matrix(kernel, "ask_target", stage, (0,))
        beta_cost = clarification_cost_matrix(kernel, "ask_target", stage, (1,))
        self.assertFalse(np.array_equal(alpha_cost, beta_cost))

    def test_all_clarification_posteriors_normalize(self) -> None:
        kernel, belief = build_kernel(CONFIG)
        cases = [
            ("ask_reference", STAGE_NAMES.index("PRE_REFERENCE"), ()),
            ("ask_target", STAGE_NAMES.index("PRE_REFERENCE"), ()),
            ("ask_target", STAGE_NAMES.index("POST_REFERENCE"), (0,)),
            ("ask_target", STAGE_NAMES.index("POST_REFERENCE"), (1,)),
            ("ask_target", STAGE_NAMES.index("POST_REFERENCE"), (2,)),
        ]
        for action, stage, history in cases:
            step = clarification_step(kernel, belief, action, stage, history)
            self.assertAlmostEqual(float(step["probabilities"].sum()), 1.0)
            self.assertTrue(all(np.isclose(posterior.sum(), 1.0) for posterior in step["posteriors"].values()))

    def test_control_has_zero_immediate_reward_and_mandatory_settlement(self) -> None:
        kernel, belief = build_kernel(CONFIG)
        for action in CONTROL_ACTIONS:
            result = control_return(kernel, belief, action)
            self.assertEqual(result["immediate_reward"], 0.0)
            self.assertTrue(result["mandatory_automatic_settlement"])
            self.assertAlmostEqual(result["total_return"], -10.0)

    def test_final_stage_cannot_clarify_or_escape(self) -> None:
        actions = allowed_actions(CONFIG, STAGE_NAMES.index("POST_TARGET"))
        self.assertEqual(actions, ("act_A", "act_B", "defer"))
        self.assertTrue(set(actions).isdisjoint(CLARIFICATION_ACTIONS))

    def test_each_surface_family_round_trips_semantic_branch_ids(self) -> None:
        policy = {
            "selected_action": "ask_reference",
            "branches": {
                0: {"selected_action": "act_A", "branches": {}},
                1: {"selected_action": "act_B", "branches": {}},
                2: {"selected_action": "defer", "branches": {}},
            },
        }
        for family in CONFIG["grammar"]["surfaceFamilies"]:
            recovered = recover_semantic_policy(render_policy(policy, CONFIG, family), CONFIG, family)
            self.assertEqual(recovered, policy)


if __name__ == "__main__":
    unittest.main()
