import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from generate_v42_sequential import build_population
from v42_stateful import (
    apply_action,
    canonical_program,
    effect,
    entities,
    epistemic_rows,
    execute_partial,
    execute_sequence,
    mechanic_registry,
    relation,
    unary,
)


class V42StatefulTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((PROJECT_ROOT / "configs/v42-sequential-state-foundation.json").read_text())

    def test_registry_has_registered_unique_families(self):
        registry = mechanic_registry()
        self.assertEqual(len(registry), 40)
        self.assertEqual(len({row["key"] for row in registry}), 40)
        self.assertEqual(
            {family: sum(row["family"] == family for row in registry) for family in self.config["population"]["families"]},
            {family: 10 for family in self.config["population"]["families"]},
        )

    def test_effects_are_simultaneous_and_state_persists(self):
        entity_rows = entities(2)
        atoms = [
            f"u:{predicate}:unit_{index}" for predicate in ("active", "marked", "ready") for index in range(2)
        ] + ["r:linked:unit_0:unit_1", "r:linked:unit_1:unit_0"]
        world = {atom: False for atom in atoms}
        world["u:active:unit_0"] = True
        program = canonical_program({"rules": [
            {"action": "pulse", "effects": [
                effect("toggle", unary("active", "actor")),
                effect("copy", unary("marked", "target"), unary("active", "actor")),
            ]},
            {"action": "route", "effects": [effect("set_true", unary("ready", "target"))]},
        ]})
        binding = {"actor": "unit_0", "target": "unit_1"}
        after_pulse = apply_action(program, entity_rows, world, {"id": "pulse", "binding": binding})
        self.assertFalse(after_pulse["u:active:unit_0"])
        self.assertTrue(after_pulse["u:marked:unit_1"])
        trajectory = execute_sequence(program, entity_rows, world, [
            {"id": "pulse", "binding": binding}, {"id": "route", "binding": binding},
        ])
        self.assertTrue(trajectory[-1]["u:marked:unit_1"])
        self.assertTrue(trajectory[-1]["u:ready:unit_1"])

    def test_partial_execution_unions_compatible_worlds(self):
        mechanic = mechanic_registry()[0]
        entity_rows = entities(2)
        world = {atom: False for atom in (
            "u:active:unit_0", "u:active:unit_1", "u:marked:unit_0", "u:marked:unit_1",
            "u:ready:unit_0", "u:ready:unit_1", "r:linked:unit_0:unit_1", "r:linked:unit_1:unit_0",
        )}
        result = execute_partial(
            [mechanic["program"]], entity_rows,
            epistemic_rows(world, ["u:ready:unit_0"]),
            [{"id": "pulse", "binding": {"actor": "unit_0", "target": "unit_1"}}],
        )
        self.assertEqual(len(result["possible_final_observations"]), 2)

    def test_population_quotas_and_no_query_targets(self):
        rows = build_population(self.config)
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(row["split"] == "development_fit" for row in rows), 24)
        self.assertEqual(sum(row["split"] == "development_evaluation" for row in rows), 16)
        self.assertTrue(all(len(row["agent_input"]["queries"]) == 24 for row in rows))
        self.assertFalse(any("target" in query for row in rows for query in row["agent_input"]["queries"]))


if __name__ == "__main__":
    unittest.main()
