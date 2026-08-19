#!/usr/bin/env python3
"""Outcome-blind regression tests for the narrow V77r1 execution repair."""
from __future__ import annotations

import unittest

import numpy as np

from v77_clarification_benchmark import ClarificationKernel, evaluate_policy_exact
from v77r1_execution_repair import complete_terminal_branches


def tiny_kernel() -> tuple[ClarificationKernel, np.ndarray]:
    transition = np.zeros((2, 2, 2))
    transition[:, 1, 1] = 1.0
    transition[0, 0, 0] = 1.0
    transition[1, 0, 1] = 1.0
    observation = np.zeros((2, 2, 2, 2))
    observation[..., 1] = 1.0
    observation[:, 0, 0] = (0.5, 0.5)
    reward = np.zeros((2, 2, 2, 2))
    reward[:, 1, 0, 1] = 2.0
    kernel = ClarificationKernel(
        hypothesis_names=("candidate", "none"),
        action_names=("inspect", "safe_draft"),
        observation_names=("signal", "done"),
        state_names=("active", "terminal"),
        transition=transition,
        observation=observation,
        reward=reward,
        discount=1.0,
        send_minimum_matching_posterior=0.9,
        send_maximum_none_posterior=0.1,
        send_action_to_hypothesis=(),
        none_hypothesis=1,
        always_certified_actions=(0, 1),
    )
    belief = np.zeros((2, 2))
    belief[:, 0] = 0.5
    return kernel, belief


class V77r1ExecutionRepairTests(unittest.TestCase):
    def test_materializes_only_terminal_successors(self) -> None:
        kernel, belief = tiny_kernel()
        policy = {
            "terminal": False,
            "horizon": 2,
            "selected_action": 1,
            "branches": {},
        }
        with self.assertRaisesRegex(RuntimeError, "omits a reachable"):
            evaluate_policy_exact(kernel, belief, policy, 2)
        repaired = complete_terminal_branches(kernel, belief, policy, 2)
        self.assertEqual(evaluate_policy_exact(kernel, belief, repaired, 2), 2.0)
        self.assertTrue(repaired["branches"][1]["terminal"])

    def test_refuses_to_invent_nonterminal_continuation(self) -> None:
        kernel, belief = tiny_kernel()
        policy = {
            "terminal": False,
            "horizon": 2,
            "selected_action": 0,
            "branches": {},
        }
        with self.assertRaisesRegex(RuntimeError, "refuses to synthesize"):
            complete_terminal_branches(kernel, belief, policy, 2)


if __name__ == "__main__":
    unittest.main()
