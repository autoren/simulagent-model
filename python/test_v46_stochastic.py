import json
import unittest
from fractions import Fraction

from generate_v46_stochastic import build_population
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, effect, entities, unary
from v46_stochastic import (
    _rule, canonical_program, delayed, execute_distribution, mechanic_registry, stochastic,
)


class V46StochasticTests(unittest.TestCase):
    def test_immediate_branch_uses_exact_rational_mass(self):
        entity_rows = entities(2)
        world = deterministic_world(entity_rows, "v46-immediate")
        world["u:active:unit_1"] = False
        program = canonical_program({"rules": [
            _rule("pulse", stochastic_immediate=[stochastic("1/4", effect("set_true", unary("active", "target")))]),
            _rule("route"),
        ]})
        result = execute_distribution(program, entity_rows, world, [{"id": "pulse", "binding": {"actor": "unit_0", "target": "unit_1"}}])
        masses = sorted(Fraction(row["mass"]["numerator"], row["mass"]["denominator"]) for row in result[0])
        self.assertEqual([Fraction(1, 4), Fraction(3, 4)], masses)

    def test_delayed_choice_is_hidden_until_delivery(self):
        entity_rows = entities(2)
        world = deterministic_world(entity_rows, "v46-delayed")
        world["u:ready:unit_1"] = False
        program = canonical_program({"rules": [
            _rule("pulse", stochastic_delayed=[delayed(1, stochastic("3/4", effect("set_true", unary("ready", "target"))))]),
            _rule("route"),
        ]})
        actions = [{"id": "pulse", "binding": {"actor": "unit_0", "target": "unit_1"}}, {"id": "wait", "binding": {}}]
        result = execute_distribution(program, entity_rows, world, actions)
        self.assertEqual(1, len(result[0]))
        self.assertEqual(2, len(result[1]))

    def test_multiple_delayed_rules_are_rejected(self):
        branch = delayed(1, stochastic("1/2", effect("toggle", unary("active", "target"))))
        with self.assertRaises(ValueError):
            canonical_program({"rules": [_rule("pulse", stochastic_delayed=[branch]), _rule("route", stochastic_delayed=[branch])]})

    def test_registry_and_population_quotas(self):
        registry = mechanic_registry()
        self.assertEqual(40, len(registry))
        self.assertEqual(40, len({row["key"] for row in registry}))
        config = json.loads((PROJECT_ROOT / "configs/v46-oracle-stochastic-transitions.json").read_text())
        rows = build_population(config)
        self.assertEqual(40, len(rows))
        self.assertEqual(960, sum(len(row["agent_input"]["queries"]) for row in rows))
        self.assertTrue(all(row["oracle_metadata"]["probability_sensitive_queries"] for row in rows))


if __name__ == "__main__":
    unittest.main()
