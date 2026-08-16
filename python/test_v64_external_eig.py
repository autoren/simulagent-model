from __future__ import annotations

import math
import unittest

import numpy as np

from v64_scalar_reference import (
    atoms_to_dense as scalar_atoms_to_dense,
    filter_history as scalar_filter_history,
    load_reference,
    score_all_actions as scalar_score_all_actions,
)
from v64_external_eig import (
    assert_public_selection_payload,
    attempted_outcome_leak_selection,
    filter_public_history,
    identity_posterior,
    initial_joint_belief,
    load_family,
    posterior_kl_to_static_prior,
    predict_joint_parameter_observation,
    scaled_beta_2_2_quadrature,
    score_all_actions,
    score_control_policies,
    select_action,
    simulate_step,
    static_posterior,
    theta_posterior,
    true_transition,
    update_joint_belief,
)


class V64ExternalEigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.family = load_family(quadrature_nodes=41)
        cls.reference = load_reference(quadrature_nodes=41)

    def test_frozen_family_shapes_and_probability_invariants(self):
        family = self.family
        self.assertEqual((2, 41, 4, 11, 11), family.transitions.shape)
        self.assertEqual((0, 2, 1, 3), family.canonical_actions)
        self.assertTrue(np.all(family.transitions >= 0.0))
        np.testing.assert_allclose(family.transitions.sum(axis=-1), 1.0, atol=1e-14)
        self.assertAlmostEqual(1.0, float(family.static_prior.sum()), places=14)
        for identity in range(2):
            for theta in (0.6, 0.775, 0.95):
                for action in family.model.actions:
                    np.testing.assert_allclose(
                        true_transition(family, identity, theta, action).sum(axis=1),
                        1.0,
                        atol=1e-14,
                    )

    def test_scaled_beta_quadrature_normalizes_and_has_correct_mean(self):
        theta, weights = scaled_beta_2_2_quadrature(41, 0.6, 0.95)
        self.assertAlmostEqual(1.0, float(weights.sum()), places=14)
        self.assertAlmostEqual(0.775, float(theta @ weights), places=14)
        self.assertGreater(float(theta.min()), 0.6)
        self.assertLess(float(theta.max()), 0.95)

    def test_initial_and_sequential_filter_normalize(self):
        family = self.family
        total = 0.0
        for observation in ("left", "right", "neither", "both"):
            belief, probability = initial_joint_belief(family, observation)
            total += probability
            self.assertAlmostEqual(1.0, float(belief.sum()), places=14)
            np.testing.assert_allclose(static_posterior(belief), family.static_prior, atol=1e-14)
        self.assertAlmostEqual(1.0, total, places=12)
        belief, _ = initial_joint_belief(family, "left")
        _, joint = predict_joint_parameter_observation(family, belief, "n")
        self.assertAlmostEqual(1.0, float(joint.sum()), places=14)
        observation = int(np.argmax(joint.sum(axis=(0, 1))))
        posterior, probability = update_joint_belief(family, belief, "n", observation)
        self.assertGreater(probability, 0.0)
        self.assertAlmostEqual(1.0, float(posterior.sum()), places=14)
        filtered, log_evidence = filter_public_history(
            family, "left", ["n"], [family.model.observations[observation]]
        )
        np.testing.assert_allclose(filtered, posterior, atol=1e-15)
        self.assertTrue(math.isfinite(log_evidence))

    def test_candidate_eig_matches_direct_joint_mutual_information(self):
        family = self.family
        belief, _ = initial_joint_belief(family, "right")
        for row in score_all_actions(family, belief):
            _, joint = predict_joint_parameter_observation(
                family, belief, row["action_index"]
            )
            target = belief.sum(axis=2)
            predictive = joint.sum(axis=(0, 1))
            reference = 0.0
            for identity in range(2):
                for node in range(len(family.theta)):
                    for observation in range(len(family.model.observations)):
                        mass = float(joint[identity, node, observation])
                        denominator = float(target[identity, node] * predictive[observation])
                        if mass > 0.0:
                            reference += mass * math.log(mass / denominator)
            self.assertTrue(row["normalizes"])
            self.assertTrue(row["finite"])
            self.assertGreaterEqual(row["eig"], -1e-12)
            self.assertAlmostEqual(reference, row["eig"], places=13)
            self.assertAlmostEqual(row["eig"], row["entropy_eig"], places=12)

    def test_structurally_separate_scalar_filter_and_scores(self):
        family = self.family
        histories = [
            ("left", [], []),
            ("right", ["s"], ["right"]),
            ("neither", ["e", "n"], ["neither", "left"]),
            ("both", ["w", "s"], ["left", "neither"]),
        ]
        for initial, actions, observations in histories:
            try:
                candidate, candidate_log_evidence = filter_public_history(
                    family, initial, actions, observations
                )
                atoms, reference_log_evidence = scalar_filter_history(
                    self.reference, initial, actions, observations
                )
            except ValueError:
                continue
            np.testing.assert_allclose(
                candidate, scalar_atoms_to_dense(self.reference, atoms), atol=2e-15
            )
            self.assertAlmostEqual(candidate_log_evidence, reference_log_evidence, places=13)
            candidate_scores = score_all_actions(family, candidate)
            reference_scores = scalar_score_all_actions(self.reference, atoms)
            self.assertEqual(
                [row["action"] for row in candidate_scores],
                [row["action"] for row in reference_scores],
            )
            np.testing.assert_allclose(
                [row["eig"] for row in candidate_scores],
                [row["eig"] for row in reference_scores],
                atol=2e-14,
            )

    def test_reachable_histories_change_strict_eig_action(self):
        family = self.family
        selected = set()
        for initial in ("left", "right", "neither", "both"):
            belief, _ = initial_joint_belief(family, initial)
            selected.add(select_action(family, belief)["selected"]["action"])
            for action in family.model.actions:
                _, joint = predict_joint_parameter_observation(family, belief, action)
                for observation, probability in enumerate(joint.sum(axis=(0, 1))):
                    if probability <= 1e-10:
                        continue
                    child, _ = update_joint_belief(
                        family, belief, action, observation
                    )
                    choice = select_action(family, child)
                    scores = sorted(row["eig"] for row in choice["scores"])
                    if scores[-1] - scores[-2] > 1e-4:
                        selected.add(choice["selected"]["action"])
        self.assertEqual(set(family.model.actions), selected)

    def test_posterior_marginals_and_kl(self):
        family = self.family
        belief, _ = initial_joint_belief(family, "left")
        self.assertAlmostEqual(0.0, posterior_kl_to_static_prior(family, belief), places=13)
        posterior, _ = update_joint_belief(family, belief, "n", "both")
        self.assertAlmostEqual(1.0, float(identity_posterior(posterior).sum()), places=14)
        self.assertAlmostEqual(1.0, float(theta_posterior(posterior).sum()), places=14)
        self.assertGreaterEqual(posterior_kl_to_static_prior(family, posterior), -1e-13)

    def test_registered_control_policies_are_complete_and_normalized(self):
        family = self.family
        belief, _ = initial_joint_belief(family, "right")
        controls = score_control_policies(family, belief)
        self.assertEqual(
            {
                "primary",
                "uniform_random_mean_eig",
                "predictive_entropy",
                "state_only_information",
                "map_identity",
                "theta_mean",
                "wrong_permutation",
            },
            set(controls),
        )
        for name, row in controls.items():
            if name == "uniform_random_mean_eig":
                self.assertGreaterEqual(row, 0.0)
            else:
                self.assertIn(row["selected_action"], family.model.actions)
                self.assertEqual(4, len(row["values"]))

    def test_simulator_step_uses_named_source_arrays(self):
        family = self.family
        result = simulate_step(
            family,
            identity=0,
            theta=0.8,
            state=0,
            action="e",
            transition_uniform=0.4,
            observation_uniform=0.2,
        )
        self.assertEqual(3, len(result))
        self.assertIn(result[0], range(11))
        self.assertIn(result[1], range(6))
        self.assertTrue(math.isfinite(result[2]))

    def test_selection_firewalls_reject_truth_and_outcome(self):
        public = {
            "record_id": "fixture",
            "prefix_length": 0,
            "initial_observation": "left",
            "actions": [],
            "observations": [],
        }
        assert_public_selection_payload(public)
        with self.assertRaises(PermissionError):
            assert_public_selection_payload({**public, "theta": 0.8})
        with self.assertRaises(PermissionError):
            attempted_outcome_leak_selection(public, "right")


if __name__ == "__main__":
    unittest.main()
