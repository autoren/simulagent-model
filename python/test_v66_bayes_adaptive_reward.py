from __future__ import annotations

import unittest

import numpy as np

from v62_external_pomdp import ExactPlanner, POMDPModel
from v64_external_eig import filter_public_history, load_family
from v66_bayes_adaptive_reward import (
    StaticKernel,
    assert_synthetic_planner_fixture,
    compact_policy,
    evaluate_policy,
    evaluate_policy_information,
    evaluate_root_action_values,
    exact_eig_crosscheck,
    exact_kernel_and_belief,
    expected_reward,
    map_model_policy,
    persistent_posterior_sampling_mixture,
    plan_bayes_adaptive,
    plan_information_only_policy,
    plan_invalid_mean_transition_policy,
    plan_myopic_reward_policy,
    point_model_kernel_and_belief,
    posterior_weighted_model_oracle,
    restore_compact_policy,
    scalar_plan_bayes_adaptive,
    static_eig,
    step_belief,
    systematic_quantile_indices,
)


class V66BayesAdaptiveRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.family = load_family(quadrature_nodes=3)
        cls.exact_belief, _ = filter_public_history(cls.family, "left", [], [])
        cls.kernel, cls.belief = exact_kernel_and_belief(
            cls.family, cls.exact_belief
        )

    def test_exact_flattened_belief_and_steps_normalize(self) -> None:
        self.assertEqual((6, 11), self.belief.shape)
        self.assertAlmostEqual(1.0, float(self.belief.sum()), places=12)
        for action in self.kernel.canonical_actions:
            step = step_belief(self.kernel, self.belief, action)
            self.assertAlmostEqual(
                1.0, float(step["probabilities"].sum()), places=12
            )
            self.assertAlmostEqual(
                1.0,
                sum(float(step["probabilities"][o]) for o in step["posteriors"]),
                places=12,
            )
            for posterior in step["posteriors"].values():
                self.assertAlmostEqual(1.0, float(posterior.sum()), places=12)

    def test_horizon_one_is_expected_immediate_reward(self) -> None:
        decision = plan_bayes_adaptive(self.kernel, self.belief, 1)
        expected = tuple(
            expected_reward(self.kernel, self.belief, action)
            for action in self.kernel.canonical_actions
        )
        np.testing.assert_allclose(decision["q_values"], expected, atol=1e-12, rtol=0)

    def test_vectorized_planner_matches_independent_scalar_reference(self) -> None:
        primary = plan_bayes_adaptive(
            self.kernel, self.belief, 2, retain_forced_root_actions=True
        )
        reference = scalar_plan_bayes_adaptive(self.kernel, self.belief, 2)
        self.assertAlmostEqual(primary["value"], reference["value"], places=11)
        np.testing.assert_allclose(
            primary["q_values"], reference["q_values"], atol=1e-11, rtol=0
        )
        self.assertEqual(primary["selected_action"], reference["selected_action"])
        self.assertAlmostEqual(
            primary["value"],
            evaluate_policy(self.kernel, self.belief, primary, 2),
            places=11,
        )
        np.testing.assert_allclose(
            primary["q_values"],
            evaluate_root_action_values(self.kernel, self.belief, primary, 2),
            atol=1e-11,
            rtol=0,
        )

    def test_point_mass_static_model_reduces_to_known_model_planner(self) -> None:
        point_kernel, point_belief, _ = point_model_kernel_and_belief(
            self.kernel, self.belief, 0
        )
        primary = plan_bayes_adaptive(point_kernel, point_belief, 2)
        model = POMDPModel(
            name="v66-point-fixture",
            states=self.family.model.states,
            actions=self.family.model.actions,
            observations=self.family.model.observations,
            discount=self.family.model.discount,
            initial=point_belief[0],
            transition=point_kernel.transitions[0],
            observation=self.family.model.observation,
            reward=self.family.model.reward,
        )
        reference = ExactPlanner(model).decision(point_belief[0], 2)
        expected = tuple(reference.q_values[action] for action in point_kernel.canonical_actions)
        np.testing.assert_allclose(primary["q_values"], expected, atol=1e-11, rtol=0)
        self.assertEqual(primary["selected_action"], reference.action)

    def test_oracle_weakly_dominates_bayes_adaptive(self) -> None:
        exact = plan_bayes_adaptive(self.kernel, self.belief, 2)
        oracle = posterior_weighted_model_oracle(self.kernel, self.belief, 2)
        self.assertGreaterEqual(oracle["value"] + 1e-11, exact["value"])

    def test_map_and_persistent_mixture_are_exact_environment_evaluated(self) -> None:
        exact = plan_bayes_adaptive(self.kernel, self.belief, 2)
        map_result = map_model_policy(self.kernel, self.belief, 2)
        mixture = persistent_posterior_sampling_mixture(
            self.kernel, self.belief, 2, points=4, offset=0.125
        )
        self.assertLessEqual(map_result["exact_environment_value"], exact["value"] + 1e-11)
        self.assertLessEqual(mixture["value"], exact["value"] + 1e-11)
        self.assertTrue(mixture["sampled_model_persists_for_full_policy"])
        self.assertEqual(4, len(mixture["models"]))
        self.assertAlmostEqual(
            1.0, sum(mixture["root_action_distribution"]), places=12
        )

    def test_systematic_quantiles_match_explicit_inverse_cdf(self) -> None:
        weights = np.asarray([0.1, 0.2, 0.3, 0.4])
        actual = systematic_quantile_indices(weights, 4, 0.125)
        positions = 0.125 + np.arange(4) / 4
        expected = np.asarray(
            [next(i for i, edge in enumerate(np.cumsum(weights)) if p < edge) for p in positions]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_information_scorer_matches_frozen_v64_reference(self) -> None:
        self.assertLessEqual(exact_eig_crosscheck(self.family, self.exact_belief), 1e-12)
        values = [
            static_eig(self.kernel, self.belief, action)
            for action in self.kernel.canonical_actions
        ]
        self.assertTrue(all(np.isfinite(values)))
        self.assertTrue(all(value >= -1e-12 for value in values))

    def test_registered_control_policies_are_executable(self) -> None:
        exact = plan_bayes_adaptive(self.kernel, self.belief, 2)
        policies = (
            plan_myopic_reward_policy(self.kernel, self.belief, 2),
            plan_information_only_policy(self.kernel, self.belief, 2),
            plan_invalid_mean_transition_policy(self.kernel, self.belief, 2),
        )
        for policy in policies:
            value = evaluate_policy(self.kernel, self.belief, policy, 2)
            information = evaluate_policy_information(
                self.kernel, self.belief, policy, 2
            )
            self.assertTrue(np.isfinite(value))
            self.assertTrue(np.isfinite(information))
            self.assertLessEqual(value, exact["value"] + 1e-11)
        self.assertIn("invalid_static_semantics", policies[-1])

    def test_compact_policy_round_trip_preserves_value(self) -> None:
        policy = plan_bayes_adaptive(self.kernel, self.belief, 2)
        restored = restore_compact_policy(compact_policy(policy))
        self.assertAlmostEqual(
            evaluate_policy(self.kernel, self.belief, policy, 2),
            evaluate_policy(self.kernel, self.belief, restored, 2),
            places=12,
        )

    def test_malformed_belief_and_policy_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            step_belief(self.kernel, self.belief * 0.5, 0)
        policy = plan_bayes_adaptive(self.kernel, self.belief, 2)
        reachable = next(iter(policy["branches"]))
        broken = {**policy, "branches": dict(policy["branches"])}
        del broken["branches"][reachable]
        with self.assertRaises(RuntimeError):
            evaluate_policy(self.kernel, self.belief, broken, 2)

    def test_kernel_rejects_nonpersistent_or_unnormalized_transition(self) -> None:
        transitions = self.kernel.transitions.copy()
        transitions[0, 0, 0] *= 0.5
        with self.assertRaises(ValueError):
            StaticKernel(
                action_names=self.kernel.action_names,
                observation_names=self.kernel.observation_names,
                state_names=self.kernel.state_names,
                canonical_actions=self.kernel.canonical_actions,
                transitions=transitions,
                observations=self.kernel.observations,
                rewards=self.kernel.rewards,
                discount=self.kernel.discount,
                identities=self.kernel.identities,
                thetas=self.kernel.thetas,
            )

    def test_implementation_firewall_rejects_sealed_id_and_history(self) -> None:
        sealed = [{
            "record_id": "sealed",
            "prefix_length": 0,
            "initial_observation": "left",
            "actions": [],
            "observations": [],
        }]
        with self.assertRaises(PermissionError):
            assert_synthetic_planner_fixture(dict(sealed[0]), sealed)
        with self.assertRaises(PermissionError):
            assert_synthetic_planner_fixture({**sealed[0], "record_id": "changed"}, sealed)
        assert_synthetic_planner_fixture({
            "record_id": "synthetic",
            "prefix_length": 1,
            "initial_observation": "left",
            "actions": ["n"],
            "observations": ["neither"],
        }, sealed)


if __name__ == "__main__":
    unittest.main()
