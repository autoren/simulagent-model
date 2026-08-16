#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import unittest

import numpy as np

import v65_smc2_eig as v65r1
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import load_family
from v65r2_smc2_eig import (
    ImpossiblePublicHistory,
    ParticleExtinctionWithPositiveSupport,
    boolean_identity_support,
    load_config,
    normalize_identity_log_evidence,
    pool_repeats,
    posterior_summary,
    rao_blackwellize_measure,
    score_all_actions,
    smc2_inference,
)


def sealed_fatal_record() -> dict:
    path = PROJECT_ROOT / "data/v65-smc2-eig-portability/subset-public.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["record_id"] == "c55d371eada6c66063aa84e9":
            return row
    raise AssertionError("sealed V65r2 fatal fixture is missing")


class V65r2SMC2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.family = load_family(quadrature_nodes=17)
        cls.config = copy.deepcopy(load_config())
        cls.config["smcSquared"]["innerStateParticleBudget"] = 15
        cls.fatal = sealed_fatal_record()
        cls.regular = {
            "record_id": "v65r2-regular",
            "prefix_length": 2,
            "initial_observation": "left",
            "actions": ["n", "e"],
            "observations": ["left", "neither"],
        }
        cls.impossible = {
            "record_id": "v65r2-impossible",
            "prefix_length": 0,
            "initial_observation": "good",
            "actions": [],
            "observations": [],
        }

    def test_sealed_fixture_has_one_exact_zero_identity(self) -> None:
        first = boolean_identity_support(self.family, self.fatal, 0)
        second = boolean_identity_support(self.family, self.fatal, 1)
        self.assertTrue(first["supported"])
        self.assertFalse(second["supported"])
        self.assertIsNone(first["extinction_tick_zero_based"])
        self.assertEqual(second["extinction_tick_zero_based"], 4)
        self.assertTrue(first["theta_support_invariant"])
        self.assertTrue(second["theta_support_invariant"])

    def test_exact_zero_identity_returns_no_atoms_and_zero_mass(self) -> None:
        result = smc2_inference(self.family, self.fatal, self.config, 7, 0)
        self.assertTrue(result["normalizes"])
        self.assertEqual(result["identity"][1], 0.0)
        self.assertTrue(all(atom["identity"] == 0 for atom in result["atoms"]))
        self.assertEqual(result["log_evidence_by_identity"][1], -math.inf)
        self.assertEqual(result["diagnostics"]["exact_zero_identity_count"], 1)
        self.assertEqual(
            result["diagnostics"]["identity_status"][1], "exact_zero_identity_support"
        )
        self.assertEqual(
            result["diagnostics"]["work"]["outer_particles_initialized"], 7
        )

    def test_both_identity_impossibility_fails_before_particle_initialization(self) -> None:
        self.assertFalse(boolean_identity_support(self.family, self.impossible, 0)["supported"])
        self.assertFalse(boolean_identity_support(self.family, self.impossible, 1)["supported"])
        with self.assertRaises(ImpossiblePublicHistory):
            smc2_inference(self.family, self.impossible, self.config, 7, 0)

    def test_positive_support_particle_extinction_remains_hard_failure(self) -> None:
        with self.assertRaises(ParticleExtinctionWithPositiveSupport):
            smc2_inference(
                self.family,
                self.regular,
                self.config,
                7,
                0,
                force_positive_support_particle_extinction_identity=0,
            )

    def test_identity_log_evidence_normalization_preserves_exact_zero(self) -> None:
        mass = normalize_identity_log_evidence([math.log(0.2), -math.inf])
        np.testing.assert_array_equal(mass, np.asarray([1.0, 0.0]))
        with self.assertRaises(ImpossiblePublicHistory):
            normalize_identity_log_evidence([-math.inf, -math.inf])

    def test_regular_history_is_bitwise_unchanged_from_v65r1(self) -> None:
        old = v65r1.smc2_inference(self.family, self.regular, self.config, 7, 0)
        new = smc2_inference(self.family, self.regular, self.config, 7, 0)
        self.assertEqual(old["log_evidence_by_identity"], new["log_evidence_by_identity"])
        self.assertEqual(old["diagnostics"]["work"], new["diagnostics"]["work"])
        self.assertEqual(
            old["diagnostics"]["random_stream_count"],
            new["diagnostics"]["random_stream_count"],
        )
        self.assertEqual(len(old["atoms"]), len(new["atoms"]))
        for left, right in zip(old["atoms"], new["atoms"], strict=True):
            self.assertEqual(left["identity"], right["identity"])
            self.assertEqual(left["theta"], right["theta"])
            self.assertEqual(left["weight"], right["weight"])
            np.testing.assert_array_equal(left["state"], right["state"])

    def test_repaired_sealed_fixture_pools_and_scores(self) -> None:
        repeats = [
            smc2_inference(self.family, self.fatal, self.config, 7, repeat)
            for repeat in range(3)
        ]
        pooled = pool_repeats(repeats)
        repaired = rao_blackwellize_measure(self.family, pooled, self.fatal)
        summary = posterior_summary(self.family, pooled)
        self.assertEqual(summary["identity"][1], 0.0)
        scores = score_all_actions(self.family, repaired)
        self.assertEqual([row["action"] for row in scores], ["n", "e", "s", "w"])
        self.assertTrue(all(row["normalizes"] and row["finite"] for row in scores))

    def test_truth_and_undeclared_fields_remain_rejected(self) -> None:
        with self.assertRaises(PermissionError):
            boolean_identity_support(self.family, {**self.regular, "theta": 0.8}, 0)


if __name__ == "__main__":
    unittest.main()
