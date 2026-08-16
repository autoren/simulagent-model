from __future__ import annotations

import copy
import json
import math
import unittest
from fractions import Fraction
from pathlib import Path

from generate_v53_smc2 import build_exact
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, effect, entities, unary
from v46_stochastic import _rule, canonical_program, stochastic
from v53_smc2 import (
    continuous_particle_filter_episode,
    continuous_unit_transition,
    exact_inference,
    instantiate_program,
    mechanic_registry,
    parameterize_program,
    pool_smc2_repeats,
    quadrature_rule,
    smc2_inference,
)


class V53SmcSquaredTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
        )["config_payload"]
        cls.registry = mechanic_registry(cls.config["population"]["templateSeed"])

    def test_registry_is_balanced_and_unique(self):
        self.assertEqual(8, len(self.registry))
        self.assertEqual(8, len({row["key"] for row in self.registry}))
        self.assertEqual(
            {2},
            {
                sum(row["family"] == family for row in self.registry)
                for family in self.config["population"]["families"]
            },
        )

    def test_continuous_transition_has_exact_rational_mass(self):
        entity_rows = entities(2)
        world = deterministic_world(entity_rows, "v53-unit-transition")
        world["u:active:unit_1"] = False
        finite = canonical_program({"rules": [
            _rule("pulse", stochastic_immediate=[
                stochastic("1/2", effect("set_true", unary("active", "target")))
            ]),
            _rule("route"),
        ]})
        program = instantiate_program(parameterize_program(finite), 0.321)
        action = {
            "id": "pulse",
            "binding": {"actor": "unit_0", "target": "unit_1"},
        }
        branches = continuous_unit_transition(
            program, entity_rows, world, [], action, 0
        )
        self.assertEqual(Fraction(1), sum(
            (row["mass"] for row in branches.values()), Fraction(0)
        ))
        true_mass = sum(
            (row["mass"] for row in branches.values()
             if row["world"]["u:active:unit_1"]),
            Fraction(0),
        )
        self.assertEqual(Fraction("0.32100000000000001"), true_mass)
        with self.assertRaises(ValueError):
            canonical_program(program)

    def test_continuous_particle_local_likelihood_is_exact(self):
        entity_rows = entities(2)
        world = deterministic_world(entity_rows, "v53-unit-particle")
        world["u:active:unit_1"] = False
        finite = canonical_program({"rules": [
            _rule("pulse", stochastic_immediate=[
                stochastic("1/2", effect("set_true", unary("active", "target")))
            ]),
            _rule("route"),
        ]})
        program = instantiate_program(parameterize_program(finite), 0.321)
        action = {
            "id": "pulse",
            "binding": {"actor": "unit_0", "target": "unit_1"},
        }
        observed = [[{"atom": "u:active:unit_1", "value": True}]]
        value, groups, diagnostic = continuous_particle_filter_episode(
            program, entity_rows, world, [action], observed, 31, 17,
            ("unit", "continuous"),
        )
        self.assertAlmostEqual(math.log(0.32100000000000001), float(value), places=14)
        self.assertTrue(groups)
        self.assertFalse(diagnostic["extinct"])

    def test_quadrature_and_joint_atoms_normalize(self):
        rule = quadrature_rule(65, self.config["parameterModel"])
        self.assertAlmostEqual(1.0, sum(weight for _, weight in rule), places=14)
        mean = sum(theta * weight for theta, weight in rule)
        self.assertAlmostEqual(0.5, mean, places=13)

    def test_altered_seed_exact_smc_and_repeat_pool(self):
        fixture = copy.deepcopy(self.config)
        for key, value in tuple(fixture["population"].items()):
            if key.endswith("Seed") and isinstance(value, int):
                fixture["population"][key] = value + 1_000_000
        fixture["exactBenchmark"].update({
            "recordsPerTemplate": 1,
            "supportEpisodesPerRecord": 2,
            "supportSequenceLengths": [3, 4],
            "querySequenceLengths": [5, 6],
            "queryPrefixLengths": [3, 4],
            "quadratureNodes": 17,
        })
        fixture["smcSquared"].update({
            "innerStateParticleBudget": 7,
            "rejuvenationStepsPerOuterResampling": 1,
        })
        altered_registry = mechanic_registry(fixture["population"]["templateSeed"])
        record = build_exact(altered_registry[:1], fixture, set(), set())[0]
        exact = exact_inference(altered_registry, record, fixture)
        repeats = [
            smc2_inference(altered_registry, record, fixture, 7, repeat, "unit")
            for repeat in range(2)
        ]
        pooled = pool_smc2_repeats(repeats)
        for result in (exact, *repeats, pooled):
            self.assertAlmostEqual(1.0, sum(result["program"]), places=12)
            self.assertAlmostEqual(1.0, sum(result["theta_weights"]), places=12)
            self.assertAlmostEqual(1.0, sum(result["configuration"].values()), places=12)
            self.assertAlmostEqual(1.0, sum(row["weight"] for row in result["atoms"]), places=12)
        self.assertEqual(
            len(repeats[0]["atoms"]) + len(repeats[1]["atoms"]),
            len(pooled["atoms"]),
        )

    def test_fixed_ambiguity_probe_is_analytic(self):
        fixture = copy.deepcopy(self.config)
        fixture["exactBenchmark"]["quadratureNodes"] = 33
        records = build_exact(self.registry[:1], fixture, set(), set())
        probe = records[0]
        self.assertTrue(probe["ambiguity_probe"])
        self.assertTrue(all(
            action["id"] == "wait"
            for episode in [*probe["supports"], probe["query"]]
            for action in episode["actions"]
        ))
        exact = exact_inference(self.registry, probe, fixture)
        for probability in exact["program"]:
            self.assertAlmostEqual(1 / 8, probability, places=13)
        self.assertAlmostEqual(
            0.5,
            sum(value * weight for value, weight in zip(
                exact["theta_values"], exact["theta_weights"], strict=True
            )),
            places=13,
        )


if __name__ == "__main__":
    unittest.main()
