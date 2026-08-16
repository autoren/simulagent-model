#!/usr/bin/env python3
"""Analytic fixtures for the V62r1 terminal-aware residual checker."""
from __future__ import annotations

import unittest

import numpy as np

from v62_external_pomdp import Decision, ExactPlanner, POMDPModel, bellman_residual
from v62r1_terminal_residual import (
    support_is_all_action_absorbing,
    terminal_aware_bellman_residual,
)


def model(
    transition: list,
    reward: list,
    observation: list,
    initial: list[float],
    discount: float = 0.8,
) -> POMDPModel:
    t = np.asarray(transition, dtype=np.float64)
    o = np.asarray(observation, dtype=np.float64)
    return POMDPModel(
        "fixture",
        tuple(f"s{i}" for i in range(t.shape[1])),
        tuple(f"a{i}" for i in range(t.shape[0])),
        tuple(f"o{i}" for i in range(o.shape[2])),
        discount,
        np.asarray(initial, dtype=np.float64),
        t,
        o,
        np.asarray(reward, dtype=np.float64),
    )


def terminal_reward_fixture() -> tuple[POMDPModel, np.ndarray]:
    fixture = model(
        transition=[[[1.0]], [[1.0]]],
        reward=[[[7.0]], [[-3.0]]],
        observation=[[[1.0]], [[1.0]]],
        initial=[1.0],
    )
    return fixture, np.asarray([1.0])


def one_step_fixture() -> tuple[POMDPModel, np.ndarray]:
    fixture = model(
        transition=[
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        reward=[
            [[0.0, 2.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        observation=[
            [[1.0], [1.0]],
            [[1.0], [1.0]],
        ],
        initial=[1.0, 0.0],
    )
    return fixture, np.asarray([1.0, 0.0])


def two_step_observation_fixture() -> tuple[POMDPModel, np.ndarray]:
    fixture = model(
        transition=[
            [
                [0.25, 0.75, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
        ],
        reward=[
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0],
            ],
        ],
        observation=[
            [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ],
        initial=[0.6, 0.4, 0.0],
        discount=0.8,
    )
    return fixture, np.asarray([0.6, 0.4, 0.0])


class FixedPlanner:
    def __init__(self, decision: Decision):
        self.fixed = decision

    def decision(self, belief: np.ndarray, horizon: int) -> Decision:
        return self.fixed


class TerminalResidualTests(unittest.TestCase):
    def test_positive_horizon_terminal_ignores_counterfactual_reward(self) -> None:
        fixture, belief = terminal_reward_fixture()
        planner = ExactPlanner(fixture)
        self.assertEqual(terminal_aware_bellman_residual(fixture, planner, belief, 3), 0.0)
        self.assertEqual(bellman_residual(fixture, planner, belief, 3), 7.0)

    def test_zero_horizon_nonterminal(self) -> None:
        fixture, belief = one_step_fixture()
        planner = ExactPlanner(fixture)
        self.assertEqual(terminal_aware_bellman_residual(fixture, planner, belief, 0), 0.0)

    def test_nonterminal_one_step_reward(self) -> None:
        fixture, belief = one_step_fixture()
        planner = ExactPlanner(fixture)
        self.assertFalse(support_is_all_action_absorbing(fixture, belief))
        self.assertLessEqual(
            terminal_aware_bellman_residual(fixture, planner, belief, 1), 1e-12
        )

    def test_nonterminal_two_step_observation_branch(self) -> None:
        fixture, belief = two_step_observation_fixture()
        planner = ExactPlanner(fixture)
        self.assertLessEqual(
            terminal_aware_bellman_residual(fixture, planner, belief, 2), 1e-12
        )

    def test_all_actions_are_required_for_terminal_status(self) -> None:
        fixture, belief = one_step_fixture()
        self.assertFalse(support_is_all_action_absorbing(fixture, belief))
        planner = ExactPlanner(fixture)
        self.assertGreater(planner.decision(belief, 1).value, 0.0)

    def test_terminal_value_and_every_q_value_are_checked(self) -> None:
        fixture, belief = terminal_reward_fixture()
        value_bad = FixedPlanner(Decision(0, 0.25, (0.0, 0.0), (0, 1)))
        q_bad = FixedPlanner(Decision(0, 0.0, (0.0, -0.5), (0,)))
        self.assertEqual(
            terminal_aware_bellman_residual(fixture, value_bad, belief, 2), 0.25
        )
        self.assertEqual(
            terminal_aware_bellman_residual(fixture, q_bad, belief, 2), 0.5
        )


if __name__ == "__main__":
    unittest.main()
