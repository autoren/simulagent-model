from __future__ import annotations

import json
import unittest

import numpy as np

from evaluate_v64_eig import (
    evaluate_adaptive,
    evaluate_selection,
    normal_lower_95,
    randomized_rank,
    summarize_ranks,
)
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import load_family
from v64_scalar_reference import load_reference


class V64EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.family = load_family(quadrature_nodes=17)
        cls.reference = load_reference(quadrature_nodes=17)
        cls.config = json.loads(
            (PROJECT_ROOT / "configs/v64-design-lock.json").read_text()
        )["config_payload"]

    def test_small_public_selection_fixture_matches_scalar_reference(self):
        rows = [
            {
                "record_id": f"fixture-{index}",
                "prefix_length": 0,
                "initial_observation": observation,
                "actions": [],
                "observations": [],
            }
            for index, observation in enumerate(("left", "right", "neither", "both"))
        ]
        result, controls = evaluate_selection(
            self.family, self.reference, rows, self.config
        )
        self.assertLess(result["maximum_absolute_candidate_eig_error"], 1e-13)
        self.assertEqual(1.0, result["optimal_set_membership_rate"])
        self.assertEqual(1.0, result["candidate_and_predictive_normalization_rate"])
        self.assertEqual(6, len(controls["controls"]))

    def test_small_adaptive_fixture_completes_all_three_policies(self):
        public = []
        audit = []
        states = (0, 1, 2, 4, 5, 7, 8, 9, 10)
        for replication in range(12):
            scenario_id = f"synthetic-{replication}"
            state = states[replication % len(states)]
            observation = self.family.model.observations[
                int(np.argmax(self.family.model.observation[0, state]))
            ]
            public.append(
                {
                    "scenario_id": scenario_id,
                    "replication": replication,
                    "initial_observation": observation,
                }
            )
            policy_streams = {}
            for policy_index, policy in enumerate(("adaptiveEIG", "fixed", "random")):
                rng = np.random.default_rng(1000 + 10 * replication + policy_index)
                policy_streams[policy] = {
                    "transition_uniforms": rng.random(8).tolist(),
                    "observation_uniforms": rng.random(8).tolist(),
                }
            audit.append(
                {
                    "scenario_id": scenario_id,
                    "identity": replication % 2,
                    "theta": 0.7 + 0.1 * (replication % 3),
                    "initial_state": state,
                    "policy_streams": policy_streams,
                    "random_actions": ["n", "e", "s", "w"] * 2,
                }
            )
        result = evaluate_adaptive(self.family, public, audit, self.config)
        self.assertTrue(result["all_trajectories_completed"])
        self.assertEqual(12, result["replications"])
        self.assertEqual(
            {"adaptiveEIG", "fixed", "random"},
            set(result["mean_posterior_KL_from_prior_by_budget"]),
        )
        self.assertEqual(
            {"1", "2", "4", "6", "8"},
            set(result["mean_posterior_KL_from_prior_by_budget"]["adaptiveEIG"]),
        )

    def test_rank_summarizer_accepts_exact_uniform_fixture(self):
        ranks = list(range(128)) * 2
        summary = summarize_ranks(
            {"a": ranks, "b": ranks, "c": ranks}, 16, 128, [0.5, 0.8, 0.95]
        )
        self.assertAlmostEqual(1.0, summary["minimum_rank_chi_square_p_value"])
        self.assertAlmostEqual(0.0, summary["maximum_absolute_rank_bin_z"])
        self.assertLess(summary["maximum_absolute_coverage_z"], 0.2)

    def test_randomized_rank_boundaries_and_ties(self):
        draws = np.asarray([0.0, 1.0, 1.0, 2.0])
        self.assertEqual(1, randomized_rank(draws, 1.0, 0.0))
        self.assertEqual(3, randomized_rank(draws, 1.0, 0.999999))
        self.assertEqual(0, randomized_rank(draws, -1.0, 0.5))
        self.assertEqual(4, randomized_rank(draws, 3.0, 0.5))

    def test_paired_normal_interval_uses_replication_differences(self):
        differences = np.asarray([0.1, 0.2, 0.3, 0.4])
        mean, standard_error, lower = normal_lower_95(differences)
        self.assertAlmostEqual(0.25, mean)
        self.assertAlmostEqual(np.std(differences, ddof=1) / 2.0, standard_error)
        self.assertAlmostEqual(mean - 1.96 * standard_error, lower)


if __name__ == "__main__":
    unittest.main()
