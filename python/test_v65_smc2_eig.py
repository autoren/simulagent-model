#!/usr/bin/env python3
from __future__ import annotations

import copy
import unittest

import numpy as np

from v64_external_eig import load_family
from v65_scalar_reference import (
    scalar_score_all_actions,
    scalar_state_as_target,
    scalar_summary,
)
from v65_smc2_eig import (
    attempted_outcome_leak,
    canonicalize_atoms,
    collapse_map_identity,
    collapse_theta_mean,
    force_equal_identity_evidence,
    load_config,
    pool_repeats,
    posterior_summary,
    rao_blackwellize_measure,
    score_all_actions,
    score_state_as_target,
    select_action,
    smc2_inference,
    stable_seed,
)


def fixture_atoms(family) -> list[dict]:
    state_a = np.zeros(len(family.model.states))
    state_b = np.zeros(len(family.model.states))
    state_c = np.zeros(len(family.model.states))
    state_a[0] = 0.7
    state_a[1] = 0.3
    state_b[2] = 0.4
    state_b[4] = 0.6
    state_c[6] = 0.25
    state_c[8] = 0.75
    return [
        {"identity": 0, "theta": 0.65, "weight": 0.2, "state": state_a},
        {"identity": 0, "theta": 0.65, "weight": 0.3, "state": state_b},
        {"identity": 1, "theta": 0.90, "weight": 0.5, "state": state_c},
    ]


class V65SMC2EIGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.family = load_family(quadrature_nodes=17)
        cls.config = copy.deepcopy(load_config())
        cls.config["smcSquared"]["innerStateParticleBudget"] = 15
        cls.record = {
            "record_id": "v65-implementation-fixture",
            "prefix_length": 2,
            "initial_observation": "left",
            "actions": ["n", "e"],
            "observations": ["left", "neither"],
        }

    def test_stable_seed_uses_all_parts(self) -> None:
        self.assertEqual(stable_seed(1, "a", 2), stable_seed(1, "a", 2))
        self.assertNotEqual(stable_seed(1, "a", 2), stable_seed(1, "a", 3))

    def test_duplicate_static_atoms_are_merged(self) -> None:
        atoms = canonicalize_atoms(fixture_atoms(self.family))
        self.assertEqual(len(atoms), 2)
        self.assertAlmostEqual(sum(row["weight"] for row in atoms), 1.0, places=13)
        merged = next(row for row in atoms if row["identity"] == 0)
        self.assertAlmostEqual(merged["weight"], 0.5, places=13)

    def test_candidate_scores_match_independent_scalar_reference(self) -> None:
        measure = {"atoms": fixture_atoms(self.family)}
        candidate = score_all_actions(self.family, measure)
        reference = scalar_score_all_actions(self.family, measure)
        self.assertEqual([row["action"] for row in candidate], ["n", "e", "s", "w"])
        for left, right in zip(candidate, reference, strict=True):
            self.assertAlmostEqual(left["eig"], right["eig"], places=13)
            self.assertLess(
                np.max(np.abs(np.asarray(left["predictive"]) - right["predictive"])),
                1e-13,
            )

    def test_state_target_control_matches_scalar_reference(self) -> None:
        measure = {"atoms": fixture_atoms(self.family)}
        for action in self.family.canonical_actions:
            self.assertAlmostEqual(
                score_state_as_target(self.family, measure, action),
                scalar_state_as_target(self.family, measure, action),
                places=13,
            )

    def test_point_mass_static_latent_has_zero_eig(self) -> None:
        atom = fixture_atoms(self.family)[0]
        measure = {"atoms": [{**atom, "weight": 1.0}]}
        for row in score_all_actions(self.family, measure):
            self.assertAlmostEqual(row["eig"], 0.0, places=13)

    def test_pool_before_score_is_not_score_then_average(self) -> None:
        first = {
            "record_id": "pool",
            "outer_budget": 7,
            "repeat": 0,
            "normalizes": True,
            "atoms": [{**fixture_atoms(self.family)[0], "weight": 1.0}],
        }
        second = {
            "record_id": "pool",
            "outer_budget": 7,
            "repeat": 1,
            "normalizes": True,
            "atoms": [{**fixture_atoms(self.family)[2], "weight": 1.0}],
        }
        pooled = pool_repeats([first, second])
        pooled_values = [row["eig"] for row in score_all_actions(self.family, pooled)]
        separate_values = np.mean(
            [
                [row["eig"] for row in score_all_actions(self.family, repeat)]
                for repeat in (first, second)
            ],
            axis=0,
        )
        self.assertGreater(max(np.abs(np.asarray(pooled_values) - separate_values)), 1e-5)

    def test_tiny_smc2_is_deterministic_and_normalized(self) -> None:
        first = smc2_inference(self.family, self.record, self.config, 7, 0)
        second = smc2_inference(self.family, self.record, self.config, 7, 0)
        self.assertTrue(first["normalizes"])
        self.assertTrue(posterior_summary(self.family, first)["normalizes"])
        self.assertEqual(
            [row["theta"] for row in first["atoms"]],
            [row["theta"] for row in second["atoms"]],
        )
        self.assertLess(
            np.max(
                np.abs(
                    np.asarray([row["weight"] for row in first["atoms"]])
                    - np.asarray([row["weight"] for row in second["atoms"]])
                )
            ),
            1e-15,
        )
        self.assertEqual(first["diagnostics"]["random_stream_collision_count"], 0)

    def test_shared_inner_stream_control_is_detected(self) -> None:
        result = smc2_inference(
            self.family,
            self.record,
            self.config,
            7,
            0,
            shared_inner_stream=True,
        )
        self.assertGreater(result["diagnostics"]["random_stream_collision_count"], 0)

    def test_pool_and_controls_remain_normalized(self) -> None:
        repeats = [
            smc2_inference(self.family, self.record, self.config, 7, repeat)
            for repeat in range(3)
        ]
        pooled = pool_repeats(repeats)
        self.assertTrue(pooled["normalizes"])
        for controlled in (
            collapse_map_identity(pooled),
            collapse_theta_mean(pooled),
            force_equal_identity_evidence(pooled),
        ):
            self.assertTrue(posterior_summary(self.family, controlled)["normalizes"])
            self.assertEqual(len(select_action(self.family, controlled)["scores"]), 4)

    def test_rao_blackwellization_preserves_static_weights_after_pooling(self) -> None:
        repeats = [
            smc2_inference(self.family, self.record, self.config, 7, repeat)
            for repeat in range(3)
        ]
        pooled = pool_repeats(repeats)
        repaired = rao_blackwellize_measure(self.family, pooled, self.record)
        before = [row["weight"] for row in pooled["atoms"]]
        after = [row["weight"] for row in repaired["atoms"]]
        self.assertLess(np.max(np.abs(np.asarray(before) - after)), 1e-15)
        self.assertTrue(repaired["rao_blackwellized_known_state"])

    def test_rao_blackwellization_requires_three_repeat_pool(self) -> None:
        single = smc2_inference(self.family, self.record, self.config, 7, 0)
        with self.assertRaises(ValueError):
            rao_blackwellize_measure(self.family, single, self.record)

    def test_posterior_summary_matches_scalar_reference(self) -> None:
        measure = {"atoms": fixture_atoms(self.family)}
        candidate = posterior_summary(self.family, measure)
        reference = scalar_summary(self.family, measure)
        self.assertLess(np.max(np.abs(candidate["identity"] - reference["identity"])), 1e-13)
        self.assertLess(np.max(np.abs(candidate["state"] - reference["state"])), 1e-13)
        self.assertLess(
            np.max(np.abs(candidate["joint_bins"] - reference["joint_bins"])), 1e-13
        )

    def test_outcome_and_truth_leakage_are_rejected(self) -> None:
        attempted_outcome_leak(self.record, None)
        with self.assertRaises(PermissionError):
            attempted_outcome_leak(self.record, "future")
        with self.assertRaises(PermissionError):
            attempted_outcome_leak({**self.record, "theta": 0.8}, None)

    def test_undeclared_public_field_is_rejected(self) -> None:
        with self.assertRaises(PermissionError):
            smc2_inference(
                self.family,
                {**self.record, "audit": {}},
                self.config,
                7,
                0,
            )


if __name__ == "__main__":
    unittest.main()
