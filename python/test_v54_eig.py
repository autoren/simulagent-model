from __future__ import annotations

import copy
import json
import math
import unittest

from generate_v53_smc2 import build_exact
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, entities
from v53_smc2 import exact_inference, mechanic_registry, quadrature_rule
from v54_eig import (
    assert_selection_payload_is_public,
    attempted_outcome_leak_selection,
    belief_atoms_from_exact,
    candidate_interventions,
    expected_information_gain_from_joint,
    map_program_atoms,
    scalar_reference_eig,
    score_control_policies,
    score_all_interventions,
    select_score,
)


class V54ExactEigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v54 = json.loads(
            (PROJECT_ROOT / "configs/v54-design-lock.json").read_text()
        )["config_payload"]
        cls.v53 = json.loads(
            (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
        )["config_payload"]
        cls.registry = mechanic_registry(cls.v53["population"]["templateSeed"])

    def test_complete_candidate_counts_and_equal_assay_lengths(self):
        for count, expected in ((2, 5), (3, 13)):
            candidates = candidate_interventions(entities(count))
            self.assertEqual(expected, len(candidates))
            self.assertEqual({3}, {len(row["assay"]) for row in candidates})
            self.assertEqual(len(candidates), len({row["key"] for row in candidates}))

    def test_closed_form_binary_bernoulli_mutual_information(self):
        prior = {"low": 0.5, "high": 0.5}
        joint = {
            "failure": {"low": 0.375, "high": 0.125},
            "success": {"low": 0.125, "high": 0.375},
        }
        binary_entropy_quarter = -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
        expected = math.log(2) - binary_entropy_quarter
        result = expected_information_gain_from_joint(prior, joint)
        self.assertAlmostEqual(expected, result["eig"], places=14)
        self.assertAlmostEqual(expected, result["entropy_eig"], places=14)
        self.assertAlmostEqual(expected, scalar_reference_eig(prior, joint), places=14)

    def test_independent_outcome_has_zero_information(self):
        prior = {"a": 0.25, "b": 0.75}
        joint = {
            "x": {"a": 0.1, "b": 0.3},
            "y": {"a": 0.15, "b": 0.45},
        }
        result = expected_information_gain_from_joint(prior, joint)
        self.assertAlmostEqual(0.0, result["eig"], places=14)
        self.assertAlmostEqual(0.0, result["entropy_eig"], places=14)

    def test_exact_history_fixture_scores_normalize_and_match_reference(self):
        fixture = copy.deepcopy(self.v53)
        for key, value in tuple(fixture["population"].items()):
            if key.endswith("Seed") and isinstance(value, int):
                fixture["population"][key] = value + 2_000_000
        fixture["exactBenchmark"].update({
            "recordsPerTemplate": 1,
            "supportEpisodesPerRecord": 1,
            "supportSequenceLengths": [2, 2],
            "querySequenceLengths": [3, 3],
            "queryPrefixLengths": [2, 2],
            "quadratureNodes": 9,
        })
        record = build_exact(self.registry[:1], fixture, set(), set())[0]
        exact = exact_inference(self.registry, record, fixture)
        atoms = belief_atoms_from_exact(exact)
        scores = score_all_interventions(
            atoms, self.registry, record["query"]["entities"],
            record["query"]["prefix_length"],
        )
        self.assertEqual(5, len(scores))
        for row in scores:
            self.assertTrue(row["normalizes"])
            self.assertTrue(row["finite"])
            self.assertGreaterEqual(row["eig"], -1e-12)
            self.assertLessEqual(row["eig"], row["prior_entropy"] + 1e-12)
            self.assertAlmostEqual(row["eig"], row["entropy_eig"], places=12)
            self.assertAlmostEqual(row["eig"], row["reference_eig"], places=12)
        selected = select_score(scores)
        self.assertIn(selected["selected"]["intervention_key"], selected["optimal_keys"])
        collapsed = map_program_atoms(atoms)
        self.assertAlmostEqual(1.0, sum(row["weight"] for row in collapsed), places=14)
        self.assertEqual(1, len({row["program_index"] for row in collapsed}))
        controls = score_control_policies(
            atoms, self.registry, record["query"]["entities"],
            record["query"]["prefix_length"],
        )
        self.assertEqual(
            {
                "primary", "uniform_random_mean_eig", "predictive_entropy",
                "state_only_information", "map_program", "theta_point_mass",
                "likelihood_squared",
            },
            set(controls),
        )

    def test_quadrature_fixture_is_normalized(self):
        rule = quadrature_rule(17, self.v53["parameterModel"])
        self.assertAlmostEqual(1.0, sum(weight for _, weight in rule), places=14)

    def test_outcome_leakage_is_rejected(self):
        with self.assertRaises(PermissionError):
            attempted_outcome_leak_selection({}, {"outcome": True})
        with self.assertRaises(PermissionError):
            assert_selection_payload_is_public({"target_theta": 0.5})
        assert_selection_payload_is_public({"observations": [[{"atom": "x", "value": True}]]})


if __name__ == "__main__":
    unittest.main()
