#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from v62_external_pomdp import (
    ExactPlanner,
    POMDPModel,
    condition_initial,
    fully_observed_oracle_value,
    parse_pomdp_text,
    public_policy_value,
    terminal_mask,
    update_belief,
    validate_model,
)


def make_tmaze(length: int) -> POMDPModel:
    states = tuple(f"p{position}g{goal}" for position in range(length + 2) for goal in range(2)) + ("terminal",)
    actions = ("north", "south", "east", "west")
    observations = ("goal0", "goal1", "hallway", "junction", "terminal")
    s_count = len(states)
    transition = np.zeros((4, s_count, s_count))
    observation = np.zeros((4, s_count, 5))
    reward = np.zeros((4, s_count, s_count))
    terminal = s_count - 1
    junction = length + 1
    for position in range(length + 2):
        for goal in range(2):
            state = 2 * position + goal
            for action in (0, 1):
                successor = terminal if position == junction else state
                transition[action, state, successor] = 1.0
                if position == junction:
                    reward[action, state, successor] = 4.0 if action == goal else -0.1
            transition[2, state, 2 * min(position + 1, junction) + goal] = 1.0
            transition[3, state, 2 * max(position - 1, 0) + goal] = 1.0
            obs = goal if position == 0 else (3 if position == junction else 2)
            observation[:, state, obs] = 1.0
    transition[:, terminal, terminal] = 1.0
    observation[:, terminal, 4] = 1.0
    initial = np.zeros(s_count)
    initial[:2] = 0.5
    return POMDPModel(
        f"fixture_tmaze{length}", states, actions, observations, 0.9,
        initial, transition, observation, reward,
    )


def make_tiger() -> POMDPModel:
    states = ("left-start", "right-start", "left", "right", "terminal")
    actions = ("listen", "open-left", "open-right")
    observations = ("init", "hear-left", "hear-right", "terminal")
    transition = np.zeros((3, 5, 5))
    transition[0, 0, 2] = transition[0, 2, 2] = 1.0
    transition[0, 1, 3] = transition[0, 3, 3] = 1.0
    transition[0, 4, 4] = 1.0
    transition[1:, :4, 4] = 1.0
    transition[1:, 4, 4] = 1.0
    observation = np.zeros((3, 5, 4))
    observation[:, 0:2, 0] = 1.0
    observation[:, 2, 1:3] = (0.85, 0.15)
    observation[:, 3, 1:3] = (0.15, 0.85)
    observation[:, 4, 3] = 1.0
    reward = np.zeros((3, 5, 5))
    reward[0] = -1.0
    reward[1, (0, 2), :] = -100.0
    reward[1, (1, 3), :] = 10.0
    reward[2, (0, 2), :] = 10.0
    reward[2, (1, 3), :] = -100.0
    initial = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    return POMDPModel(
        "fixture_tiger", states, actions, observations, 0.95,
        initial, transition, observation, reward,
    )


class V62ExternalPOMDPTests(unittest.TestCase):
    def test_parser_matrix_and_wildcard_reward(self) -> None:
        text = """
        discount: 0.9
        values: reward
        states: s0 terminal
        actions: wait commit
        observations: seen done
        start: 1 0
        T: wait
        1 0
        0 1
        T: commit
        0 1
        0 1
        O: *
        1 0
        0 1
        R: * : * : * : * -1
        R: commit : s0 : terminal : * 5
        """
        model = parse_pomdp_text(text)
        self.assertTrue(all(validate_model(model).values()))
        self.assertEqual(model.reward[1, 0, 1], 5.0)
        self.assertEqual(model.reward[0, 0, 0], -1.0)
        self.assertTrue(np.array_equal(terminal_mask(model), (False, True)))

    def test_binary_noisy_sensor_bayes_update(self) -> None:
        model = make_tiger()
        initial, mass = condition_initial(model, 0)
        self.assertAlmostEqual(mass, 1.0)
        posterior, probability = update_belief(model, initial, 0, 1)
        self.assertAlmostEqual(probability, 0.5)
        self.assertTrue(np.allclose(posterior, (0, 0, 0.85, 0.15, 0)))

    def test_tiger_values_information(self) -> None:
        model = make_tiger()
        planner = ExactPlanner(model)
        initial = condition_initial(model, 0)[0]
        self.assertEqual(planner.decision(initial, 1).action, 0)
        self.assertEqual(planner.decision(initial, 3).action, 0)
        self.assertGreater(
            public_policy_value(model, 3, "exact_history")
            - public_policy_value(model, 3, "map_collapse"),
            1.0,
        )

    def test_delayed_tmaze_memory_control(self) -> None:
        for length in (2, 5):
            model = make_tmaze(length)
            horizon = length + 2
            exact = public_policy_value(model, horizon, "exact_history")
            observation_only = public_policy_value(model, horizon, "observation_only")
            self.assertAlmostEqual(exact, 4.0 * 0.9 ** (horizon - 1))
            self.assertAlmostEqual(observation_only, 1.95 * 0.9 ** (horizon - 1))
            self.assertGreater(exact - observation_only, 0.9)
            self.assertAlmostEqual(fully_observed_oracle_value(model, horizon), exact)

    def test_all_actions_required_for_terminal(self) -> None:
        model = make_tmaze(2)
        mask = terminal_mask(model)
        self.assertEqual(mask.sum(), 1)
        self.assertTrue(mask[-1])
        self.assertFalse(mask[0])

    def test_zero_probability_observations_are_not_reached(self) -> None:
        model = make_tmaze(2)
        planner = ExactPlanner(model)
        reachable = planner.reachable_decisions(4)
        self.assertGreater(len(reachable), 0)
        self.assertTrue(all(abs(sum(belief) - 1.0) < 1e-12 for belief, _ in reachable))


if __name__ == "__main__":
    unittest.main()
