#!/usr/bin/env python3
"""Synthetic-only tests for the V67 independent verifier."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from v22r2_grounding import PROJECT_ROOT
from v67_verification import (
    IndependentModel,
    compile_policy_dtmc,
    condition_public_history,
    construct_independent_family,
    dtmc_statistics,
    execute_policy_scalar,
    independent_parse_pomdp_file,
    independent_scaled_beta_2_2_quadrature,
    run_storm_properties,
    scalar_step,
    storm_version,
    validate_source_model,
    vector_step,
    write_explicit_dtmc,
)


def fixture_model(*, two_observations: bool = False) -> IndependentModel:
    states = ("0", "1")
    actions = ("n", "s", "e", "w")
    observations = ("zero", "one") if two_observations else ("only",)
    transition = np.empty((4, 2, 2), dtype=np.float64)
    transition[0] = [[0.0, 1.0], [0.0, 1.0]]
    transition[1] = [[1.0, 0.0], [1.0, 0.0]]
    transition[2] = [[0.75, 0.25], [0.25, 0.75]]
    transition[3] = [[0.25, 0.75], [0.75, 0.25]]
    if two_observations:
        observation = np.broadcast_to(
            np.asarray([[1.0, 0.0], [0.0, 1.0]])[None, :, :], (4, 2, 2)
        ).copy()
    else:
        observation = np.ones((4, 2, 1), dtype=np.float64)
    reward = np.empty((4, 2, 2), dtype=np.float64)
    for action in range(4):
        reward[action, 0, :] = 0.2 + 0.1 * action
        reward[action, 1, :] = 0.8 - 0.05 * action
    return IndependentModel(
        states=states,
        actions=actions,
        observations=observations,
        discount=0.95,
        initial=np.asarray([1.0, 0.0]),
        transition=transition,
        observation=observation,
        reward=reward,
    )


def policy_tree(
    horizon: int, observation_count: int, *, root_action: int = 0,
    child_action: int | None = None, archived_probabilities: list[float] | None = None,
) -> dict:
    child_action = root_action if child_action is None else child_action

    def build(remaining: int, depth: int, action: int) -> dict:
        probabilities = (
            list(archived_probabilities)
            if depth == 0 and archived_probabilities is not None
            else [1.0 / observation_count] * observation_count
        )
        node = {
            "branches": {},
            "horizon": remaining,
            "observation_probabilities": probabilities,
            "optimal_actions": [action],
            "q_values": [0.0, 0.0, 0.0, 0.0],
            "selected_action": action,
            "selected_action_name": fixture_model().actions[action],
            "terminal": False,
            "value": 0.0,
        }
        if remaining > 1:
            for observation in range(observation_count):
                next_action = child_action if depth == 0 else action
                node["branches"][str(observation)] = build(
                    remaining - 1, depth + 1, next_action
                )
        return node

    return build(horizon, 0, root_action)


class V67VerificationTests(unittest.TestCase):
    def test_independent_parser_reads_pinned_source(self) -> None:
        model = independent_parse_pomdp_file(
            PROJECT_ROOT
            / "data/v63-external-unknown-dynamics/source-checkout/pobax/envs/classic/"
            "POMDP/4x3_nonterminating.POMDP"
        )
        self.assertEqual((len(model.states), len(model.actions), len(model.observations)), (11, 4, 6))
        self.assertEqual(model.actions, ("n", "s", "e", "w"))
        self.assertTrue(all(validate_source_model(model).values()))

    def test_quadrature_is_normalized_and_symmetric(self) -> None:
        theta, weights = independent_scaled_beta_2_2_quadrature(17)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)
        self.assertAlmostEqual(float(np.dot(theta, weights)), 0.775, places=13)
        self.assertTrue(np.allclose(weights, weights[::-1], atol=1e-15, rtol=0.0))

    def test_family_rows_normalize(self) -> None:
        family = construct_independent_family(fixture_model(), nodes=9)
        self.assertEqual(family.transitions.shape, (2, 9, 4, 2, 2))
        self.assertLess(float(np.max(np.abs(family.transitions.sum(-1) - 1.0))), 1e-14)

    def test_scalar_and_vector_steps_agree(self) -> None:
        family = construct_independent_family(fixture_model(two_observations=True), nodes=7)
        belief = family.static_prior[:, :, None] * np.asarray([0.6, 0.4])[None, None, :]
        for observation in range(2):
            scalar = scalar_step(family, belief, 2, observation)
            vector = vector_step(family, belief, 2, observation)
            self.assertLess(float(np.max(np.abs(scalar[0] - vector[0]))), 2e-15)
            self.assertAlmostEqual(scalar[1], vector[1], places=14)
            self.assertAlmostEqual(scalar[2], vector[2], places=14)

    def test_hand_computed_reset_filter(self) -> None:
        model = fixture_model(two_observations=True)
        model = IndependentModel(
            states=model.states, actions=model.actions, observations=model.observations,
            discount=model.discount, initial=np.asarray([0.6, 0.4]),
            transition=model.transition,
            observation=np.broadcast_to(
                np.asarray([[0.8, 0.2], [0.1, 0.9]])[None, :, :], (4, 2, 2)
            ).copy(),
            reward=model.reward,
        )
        family = construct_independent_family(model, nodes=5)
        belief, evidence = condition_public_history(family, {
            "record_id": "synthetic", "prefix_length": 0,
            "initial_observation": "zero", "actions": [], "observations": [],
        })
        self.assertAlmostEqual(evidence, 0.52, places=14)
        self.assertAlmostEqual(float(belief.sum(axis=(0, 1))[0]), 0.48 / 0.52, places=14)
        self.assertAlmostEqual(float(belief.sum(axis=(0, 1))[1]), 0.04 / 0.52, places=14)

    def test_deterministic_discounted_return(self) -> None:
        source = fixture_model()
        deterministic_transition = np.broadcast_to(
            np.asarray([[0.0, 1.0], [0.0, 1.0]])[None, :, :], (4, 2, 2)
        ).copy()
        source = IndependentModel(
            states=source.states, actions=source.actions,
            observations=source.observations, discount=source.discount,
            initial=source.initial, transition=deterministic_transition,
            observation=source.observation, reward=source.reward,
        )
        family = construct_independent_family(source, nodes=5)
        belief = family.static_prior[:, :, None] * family.model.initial[None, None, :]
        policy = policy_tree(3, 1)
        result = execute_policy_scalar(family, belief, policy)
        expected = 0.2 + 0.95 * 0.8 + 0.95**2 * 0.8
        self.assertAlmostEqual(result["value"], expected, places=13)
        self.assertEqual(result["reachable_nodes"], 3)

    def test_observation_contingent_action_change(self) -> None:
        family = construct_independent_family(fixture_model(two_observations=True), nodes=5)
        belief = family.static_prior[:, :, None] * np.asarray([0.5, 0.5])[None, None, :]
        policy = policy_tree(2, 2, root_action=2, child_action=3)
        contingent = execute_policy_scalar(family, belief, policy, horizon=2)["value"]
        fixed = execute_policy_scalar(
            family, belief, policy_tree(2, 2, root_action=2, child_action=2), horizon=2
        )["value"]
        self.assertNotAlmostEqual(contingent, fixed, places=9)

    def test_compiler_matches_scalar_executor(self) -> None:
        family = construct_independent_family(fixture_model(two_observations=True), nodes=7)
        belief = family.static_prior[:, :, None] * np.asarray([0.5, 0.5])[None, None, :]
        policy = policy_tree(3, 2, root_action=0, child_action=2)
        scalar = execute_policy_scalar(family, belief, policy)["value"]
        model, checks = compile_policy_dtmc(family, belief, policy)
        statistics = dtmc_statistics(model)
        self.assertAlmostEqual(statistics["expected_return"], scalar, places=13)
        self.assertAlmostEqual(statistics["termination_probability"], 1.0, places=14)
        self.assertEqual(checks["node_invariants"], checks["node_invariant_passes"])
        self.assertEqual(
            checks["transition_normalization_checks"],
            checks["transition_normalization_passes"],
        )

    def test_conditional_rewards_recover_unconditional_reward(self) -> None:
        family = construct_independent_family(fixture_model(two_observations=True), nodes=5)
        belief = family.static_prior[:, :, None] * np.asarray([0.4, 0.6])[None, None, :]
        weighted = 0.0
        for observation in range(2):
            _, probability, reward = scalar_step(family, belief, 2, observation)
            weighted += probability * reward
        direct = float(np.einsum(
            "ins,inst,st->", belief, family.transitions[:, :, 2], family.model.reward[2]
        ))
        self.assertAlmostEqual(weighted, direct, places=14)

    def test_static_persistence_differs_from_mean_transition_mutant(self) -> None:
        family = construct_independent_family(fixture_model(two_observations=True), nodes=5)
        belief = family.static_prior[:, :, None] * np.asarray([0.5, 0.5])[None, None, :]
        policy = policy_tree(3, 2, root_action=0, child_action=2)
        valid = execute_policy_scalar(family, belief, policy)["value"]
        mutant = execute_policy_scalar(
            family, belief, policy,
            mutation="replace_persistent_static_model_with_per_step_mean_transition",
        )["value"]
        self.assertGreater(abs(valid - mutant), 1e-8)

    def test_wrong_policy_horizon_is_rejected(self) -> None:
        family = construct_independent_family(fixture_model(), nodes=5)
        belief = family.static_prior[:, :, None] * family.model.initial[None, None, :]
        policy = policy_tree(3, 1)
        policy["horizon"] = 2
        with self.assertRaises(ValueError):
            execute_policy_scalar(family, belief, policy)

    def test_missing_positive_branch_is_rejected(self) -> None:
        family = construct_independent_family(fixture_model(two_observations=True), nodes=5)
        belief = family.static_prior[:, :, None] * np.asarray([0.5, 0.5])[None, None, :]
        policy = policy_tree(2, 2, root_action=2)
        del policy["branches"]["1"]
        with self.assertRaises(ValueError):
            execute_policy_scalar(family, belief, policy, horizon=2)

    def test_truth_like_public_field_is_rejected(self) -> None:
        family = construct_independent_family(fixture_model(), nodes=5)
        with self.assertRaises(ValueError):
            condition_public_history(family, {
                "record_id": "synthetic", "prefix_length": 0,
                "initial_observation": "only", "actions": [], "observations": [],
                "theta": 0.8,
            })

    def test_storm_round_trip_tiny_dtmc(self) -> None:
        family = construct_independent_family(fixture_model(), nodes=5)
        belief = family.static_prior[:, :, None] * family.model.initial[None, None, :]
        policy = policy_tree(3, 1)
        model, _ = compile_policy_dtmc(family, belief, policy)
        direct = dtmc_statistics(model)
        with tempfile.TemporaryDirectory() as directory:
            write_explicit_dtmc(model, directory)
            checked = run_storm_properties(directory)
        self.assertEqual(storm_version(), "1.13.0")
        self.assertAlmostEqual(checked["termination_probability"], 1.0, places=13)
        self.assertAlmostEqual(checked["expected_return"], direct["expected_return"], places=12)


if __name__ == "__main__":
    unittest.main()
