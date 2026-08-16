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
from v65r3_smc2_eig import (
    ImpossiblePublicHistory,
    ParticleExtinctionWithPositiveSupport,
    assert_synthetic_implementation_fixture,
    boolean_identity_support,
    load_config,
    normalize_identity_log_evidence,
    pool_repeats,
    posterior_summary,
    rao_blackwellize_measure,
    score_all_actions_for_implementation_fixture,
    smc2_inference,
)


def sealed_records() -> list[dict]:
    path = PROJECT_ROOT / "data/v65-smc2-eig-portability/subset-public.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class V65r3SMC2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.family = load_family(quadrature_nodes=17)
        cls.config = copy.deepcopy(load_config())
        cls.sealed = sealed_records()
        cls.fatal = next(
            row for row in cls.sealed if row["record_id"] == "c55d371eada6c66063aa84e9"
        )
        cls.synthetic = {
            "record_id": "v65r3-synthetic-eig-fixture",
            "prefix_length": 2,
            "initial_observation": "left",
            "actions": ["n", "e"],
            "observations": ["left", "neither"],
        }

    def test_sealed_fatal_record_is_support_and_posterior_only(self) -> None:
        self.assertTrue(boolean_identity_support(self.family, self.fatal, 0)["supported"])
        second = boolean_identity_support(self.family, self.fatal, 1)
        self.assertFalse(second["supported"])
        self.assertEqual(second["extinction_tick_zero_based"], 4)
        result = smc2_inference(self.family, self.fatal, self.config, 7, 0)
        self.assertTrue(result["normalizes"])
        self.assertEqual(result["identity"][1], 0.0)
        self.assertTrue(all(atom["identity"] == 0 for atom in result["atoms"]))
        self.assertEqual(result["log_evidence_by_identity"][1], -math.inf)
        self.assertEqual(result["diagnostics"]["exact_zero_identity_count"], 1)

    def test_synthetic_fixture_is_distinct_and_scores_all_actions(self) -> None:
        assert_synthetic_implementation_fixture(self.synthetic, self.sealed)
        repeats = [
            smc2_inference(self.family, self.synthetic, self.config, 7, repeat)
            for repeat in range(3)
        ]
        pooled = pool_repeats(repeats)
        repaired = rao_blackwellize_measure(self.family, pooled, self.synthetic)
        scores = score_all_actions_for_implementation_fixture(
            self.family, repaired, self.synthetic, self.sealed
        )
        self.assertEqual([row["action"] for row in scores], ["n", "e", "s", "w"])
        self.assertTrue(all(row["finite"] and row["normalizes"] for row in scores))

    def test_firewall_rejects_sealed_id_even_with_changed_history(self) -> None:
        mutant = {**self.synthetic, "record_id": self.sealed[0]["record_id"]}
        with self.assertRaises(PermissionError):
            assert_synthetic_implementation_fixture(mutant, self.sealed)

    def test_firewall_rejects_sealed_history_even_with_changed_id(self) -> None:
        mutant = {**self.sealed[0], "record_id": "changed-id"}
        with self.assertRaises(PermissionError):
            assert_synthetic_implementation_fixture(mutant, self.sealed)

    def test_both_impossible_and_positive_support_collapse_fail_closed(self) -> None:
        impossible = {
            "record_id": "v65r3-impossible",
            "prefix_length": 0,
            "initial_observation": "good",
            "actions": [],
            "observations": [],
        }
        with self.assertRaises(ImpossiblePublicHistory):
            smc2_inference(self.family, impossible, self.config, 7, 0)
        with self.assertRaises(ParticleExtinctionWithPositiveSupport):
            smc2_inference(
                self.family,
                self.synthetic,
                self.config,
                7,
                0,
                force_positive_support_particle_extinction_identity=0,
            )

    def test_identity_normalization_retains_exact_zero(self) -> None:
        np.testing.assert_array_equal(
            normalize_identity_log_evidence([math.log(0.2), -math.inf]),
            np.asarray([1.0, 0.0]),
        )
        with self.assertRaises(ImpossiblePublicHistory):
            normalize_identity_log_evidence([-math.inf, -math.inf])

    def test_positive_support_path_remains_bitwise_v65r1(self) -> None:
        old = v65r1.smc2_inference(self.family, self.synthetic, self.config, 7, 0)
        new = smc2_inference(self.family, self.synthetic, self.config, 7, 0)
        self.assertEqual(old["log_evidence_by_identity"], new["log_evidence_by_identity"])
        self.assertEqual(old["diagnostics"]["work"], new["diagnostics"]["work"])
        self.assertEqual(len(old["atoms"]), len(new["atoms"]))
        for left, right in zip(old["atoms"], new["atoms"], strict=True):
            self.assertEqual(left["identity"], right["identity"])
            self.assertEqual(left["theta"], right["theta"])
            self.assertEqual(left["weight"], right["weight"])
            np.testing.assert_array_equal(left["state"], right["state"])

    def test_posterior_summary_of_fatal_record_has_zero_identity_mass(self) -> None:
        result = smc2_inference(self.family, self.fatal, self.config, 7, 1)
        summary = posterior_summary(self.family, result)
        self.assertTrue(summary["normalizes"])
        self.assertEqual(summary["identity"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
