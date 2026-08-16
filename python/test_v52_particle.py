from __future__ import annotations

import copy
import json
import unittest
from decimal import Decimal, localcontext

from generate_v52_particle import build_exact
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, effect, entities, unary
from v46_stochastic import _rule, canonical_program, stochastic
from v52_particle import (
    configuration_distribution,
    mechanic_registry,
    particle_filter_episode,
    particle_inference,
    stream_id,
)


class V52RaoBlackwellizedParticleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lock = json.loads(
            (PROJECT_ROOT / "configs/v52r1-design-lock.json").read_text()
        )
        cls.config = copy.deepcopy(lock["config_payload"])
        for key, value in tuple(cls.config["population"].items()):
            if key.endswith("Seed") and isinstance(value, int):
                cls.config["population"][key] = value + 1_000_019
        cls.config["exactBenchmark"]["recordsPerMechanic"] = 1
        cls.registry = mechanic_registry()

    def simple_fixture(self):
        entity_rows = entities(2)
        world = deterministic_world(entity_rows, "v52-unit-filter")
        world["u:active:unit_1"] = False
        program = canonical_program({"rules": [
            _rule(
                "pulse",
                stochastic_immediate=[
                    stochastic(
                        "1/4", effect("set_true", unary("active", "target"))
                    )
                ],
            ),
            _rule("route"),
        ]})
        action = {
            "id": "pulse",
            "binding": {"actor": "unit_0", "target": "unit_1"},
        }
        return entity_rows, world, program, action

    def test_registry_is_fresh_unique_and_balanced(self):
        self.assertEqual(len(self.registry), 48)
        self.assertEqual(len({row["key"] for row in self.registry}), 48)
        cells = {}
        for row in self.registry:
            cell = (row["family"], row["probability"])
            cells[cell] = cells.get(cell, 0) + 1
        self.assertEqual(set(cells.values()), {4})

    def test_exact_local_branching_matches_quarter_probability(self):
        entity_rows, world, program, action = self.simple_fixture()
        log_likelihood, groups, diagnostics = particle_filter_episode(
            program, entity_rows, world, [action], [[]], 32, 17,
            ("unit", "exact-quarter"),
        )
        distribution = configuration_distribution(groups)
        true_mass = sum(
            mass for key, mass in distribution.items()
            if dict(json.loads(key)["world"])["u:active:unit_1"]
        )
        self.assertEqual(log_likelihood, Decimal(0))
        self.assertEqual(true_mass, Decimal("0.25"))
        self.assertEqual(sum(distribution.values(), Decimal(0)), Decimal(1))
        self.assertEqual(len(diagnostics["resampling_stream_ids"]), 1)

    def test_coprime_budget_exposes_bounded_monte_carlo_error(self):
        entity_rows, world, program, action = self.simple_fixture()
        _, groups, _ = particle_filter_episode(
            program, entity_rows, world, [action], [[]], 31, 23,
            ("unit", "coprime-quarter"),
        )
        distribution = configuration_distribution(groups)
        true_mass = sum(
            mass for key, mass in distribution.items()
            if dict(json.loads(key)["world"])["u:active:unit_1"]
        )
        self.assertIn(true_mass, {Decimal(7) / 31, Decimal(8) / 31})
        self.assertLessEqual(abs(true_mass - Decimal("0.25")), Decimal(1) / 31)

    def test_observation_conditioning_has_exact_log_likelihood(self):
        entity_rows, world, program, action = self.simple_fixture()
        evidence = [[{"atom": "u:active:unit_1", "value": True}]]
        log_likelihood, groups, diagnostics = particle_filter_episode(
            program, entity_rows, world, [action], evidence, 31, 29,
            ("unit", "observed-quarter"),
        )
        with localcontext() as context:
            context.prec = 100
            self.assertLess(
                abs(log_likelihood - Decimal("0.25").ln()), Decimal("1e-90")
            )
        self.assertEqual(len(configuration_distribution(groups)), 1)
        self.assertFalse(diagnostics["extinct"])

    def test_particle_inference_normalizes_all_marginals(self):
        records = build_exact(self.registry[:1], self.config, set(), set())
        record = records[0]
        inference = particle_inference(
            self.registry,
            record["supports"],
            record["query"],
            31,
            self.config["population"]["particleSeed"],
            "unit",
            record["id"],
            0,
            self.config["algorithm"]["resamplingEssThresholdFraction"],
        )
        for values in (
            inference["support_program"], inference["query_program"],
            inference["probability"].values(), inference["joint"].values(),
            inference["configuration"].values(), inference["suffix"].values(),
        ):
            self.assertLess(abs(sum(values, Decimal(0)) - 1), Decimal("1e-80"))

    def test_stream_identity_partitions_every_required_axis(self):
        base = ("exact", "record", 3, 31, 0, "support-0", 2)
        identifiers = {stream_id(101, *base)}
        for index in range(len(base)):
            changed = list(base)
            changed[index] = f"changed-{index}"
            identifiers.add(stream_id(101, *changed))
        identifiers.add(stream_id(102, *base))
        self.assertEqual(len(identifiers), len(base) + 2)


if __name__ == "__main__":
    unittest.main()
