#!/usr/bin/env python3
"""Synthetic and structural tests for V68 exact multi-environment infrastructure."""
from __future__ import annotations

import unittest

import numpy as np

from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import POMDPModel
from v64_external_eig import load_family as load_v64_family
from v66_bayes_adaptive_reward import (
    StaticKernel,
    plan_bayes_adaptive,
    scalar_plan_bayes_adaptive,
)
from v68_cassandra_pomdp import parse_cassandra_pomdp_file
from v68_multi_environment_exact import (
    best_open_loop_sequence,
    build_command_channel_family,
    cycle_permutations,
    enumerate_public_prefixes,
    evaluate_action_sequence,
    filter_action_observation_history,
    finite_horizon_return_scale,
)


def synthetic_model() -> POMDPModel:
    states = ("s0", "s1")
    actions = ("a", "b", "c")
    observations = ("o0", "o1")
    transition = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.75, 0.25], [0.25, 0.75]],
        ]
    )
    observation = np.broadcast_to(
        np.asarray([[0.8, 0.2], [0.1, 0.9]])[None, :, :], (3, 2, 2)
    ).copy()
    reward = np.zeros((3, 2, 2))
    reward[0, :, 0] = 1.0
    reward[1, :, 1] = 2.0
    reward[2] = 0.25
    return POMDPModel(
        name="synthetic",
        states=states,
        actions=actions,
        observations=observations,
        discount=0.9,
        initial=np.asarray([0.6, 0.4]),
        transition=transition,
        observation=observation,
        reward=reward,
    )


class V68MultiEnvironmentExactTests(unittest.TestCase):
    def test_generic_family_exactly_recovers_v64_on_4x3(self) -> None:
        path = (
            PROJECT_ROOT
            / "data/v63-external-unknown-dynamics/source-checkout/pobax/envs/classic/"
            "POMDP/4x3_nonterminating.POMDP"
        )
        model = parse_cassandra_pomdp_file(path)
        candidate = build_command_channel_family(
            model, ("n", "e", "s", "w"), quadrature_nodes=7
        )
        reference = load_v64_family(quadrature_nodes=7)
        self.assertEqual(candidate.kernel.canonical_actions, reference.canonical_actions)
        self.assertTrue(np.array_equal(candidate.permutations, reference.permutations))
        self.assertTrue(
            np.array_equal(
                candidate.kernel.transitions,
                reference.transitions.reshape(
                    14, len(model.actions), len(model.states), len(model.states)
                ),
            )
        )
        expected = np.concatenate(
            [0.5 * reference.theta_weights, 0.5 * reference.theta_weights]
        )[:, None] * model.initial[None, :]
        self.assertTrue(np.array_equal(candidate.initial_belief, expected))

    def test_cycle_permutations_follow_frozen_cycle_not_storage_order(self) -> None:
        model = synthetic_model()
        permutations, canonical = cycle_permutations(model, ("a", "c", "b"))
        self.assertEqual(canonical, (0, 2, 1))
        self.assertTrue(np.array_equal(permutations[0], (2, 0, 1)))
        self.assertTrue(np.array_equal(permutations[1], (1, 2, 0)))

    def test_family_and_initial_belief_normalize(self) -> None:
        family = build_command_channel_family(
            synthetic_model(), ("a", "b", "c"), quadrature_nodes=9
        )
        self.assertAlmostEqual(float(family.initial_belief.sum()), 1.0, places=14)
        self.assertLess(
            float(np.max(np.abs(family.kernel.transitions.sum(axis=-1) - 1.0))),
            1e-14,
        )

    def test_history_filter_matches_one_step_enumerator(self) -> None:
        family = build_command_channel_family(
            synthetic_model(), ("a", "b", "c"), quadrature_nodes=7
        )
        records = enumerate_public_prefixes(family, maximum_depth=1)
        record = next(row for row in records if row.actions == (0,) and row.observations == (0,))
        filtered, log_evidence = filter_action_observation_history(family, (0,), (0,))
        self.assertTrue(np.array_equal(filtered, record.belief))
        self.assertAlmostEqual(log_evidence, record.log_evidence, places=14)
        self.assertAlmostEqual(np.exp(log_evidence), record.probability, places=14)

    def test_complete_depth_one_census_retains_all_positive_branches(self) -> None:
        family = build_command_channel_family(
            synthetic_model(), ("a", "b", "c"), quadrature_nodes=7
        )
        records = enumerate_public_prefixes(family, maximum_depth=1)
        self.assertEqual(len(records), 1 + 3 * 2)
        self.assertEqual(sum(row.depth == 0 for row in records), 1)
        self.assertEqual(sum(row.depth == 1 for row in records), 6)
        self.assertEqual(len({row.record_id for row in records}), len(records))

    def test_open_loop_enumerates_every_sequence_and_matches_direct_value(self) -> None:
        family = build_command_channel_family(
            synthetic_model(), ("a", "b", "c"), quadrature_nodes=7
        )
        result = best_open_loop_sequence(family.kernel, family.initial_belief, 3)
        self.assertEqual(result["sequence_count"], 27)
        selected = result["selected_actions"]
        self.assertAlmostEqual(
            result["value"],
            evaluate_action_sequence(family.kernel, family.initial_belief, selected),
            places=14,
        )

    def test_vector_and_scalar_bayes_adaptive_planners_agree(self) -> None:
        transitions = np.asarray(
            [
                [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]],
                [[[0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]],
            ]
        )
        observations = np.broadcast_to(
            np.asarray([[1.0, 0.0], [0.0, 1.0]])[None, :, :], (2, 2, 2)
        ).copy()
        rewards = np.zeros((2, 2, 2))
        rewards[0, :, 0] = 1.0
        rewards[1, :, 1] = 1.0
        kernel = StaticKernel(
            action_names=("left", "right"),
            observation_names=("left", "right"),
            state_names=("left", "right"),
            canonical_actions=(0, 1),
            transitions=transitions,
            observations=observations,
            rewards=rewards,
            discount=0.9,
            identities=np.asarray([0, 1]),
            thetas=np.asarray([0.8, 0.8]),
        )
        belief = np.asarray([[0.5, 0.0], [0.5, 0.0]])
        vector = plan_bayes_adaptive(kernel, belief, 3)
        scalar = scalar_plan_bayes_adaptive(kernel, belief, 3)
        self.assertAlmostEqual(vector["value"], scalar["value"], places=14)
        self.assertEqual(vector["selected_action"], scalar["selected_action"])
        self.assertTrue(np.allclose(vector["q_values"], scalar["q_values"], atol=1e-14))

    def test_return_scale_uses_frozen_span_and_discount_sum(self) -> None:
        model = synthetic_model()
        expected = (2.0 - 0.0) * (1.0 + 0.9 + 0.9**2)
        self.assertAlmostEqual(finite_horizon_return_scale(model, 3), expected, places=14)


if __name__ == "__main__":
    unittest.main()
