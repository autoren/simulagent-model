import json
import unittest
from pathlib import Path

from v22_relational import (
    action_bindings,
    atom_universe,
    canonical_expression,
    canonical_state_hash,
    entities_for_layout,
    epistemic_from_world,
    evaluate_program,
    execute_partial,
    expression_catalog,
    find_expression_counterexample,
    greedy_identifying_support,
    hashed_world,
    rename_state,
    target_hypotheses,
    enumerate_program_hypotheses,
    validate_epistemic_state,
)


class V22RelationalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/dataset.v22.json").read_text())

    def test_complete_false_and_unknown_are_distinct(self):
        entities = entities_for_layout({"units": 2, "hubs": 0})
        world = {atom: False for atom in atom_universe(self.config, entities)}
        known = epistemic_from_world(self.config, entities, world)
        atom = next(iter(known))
        unknown = epistemic_from_world(self.config, entities, world, [atom])
        self.assertEqual(known[atom], (False,))
        self.assertEqual(unknown[atom], (False, True))
        missing = dict(unknown)
        missing.pop(atom)
        with self.assertRaises(ValueError):
            validate_epistemic_state(self.config, entities, missing)

    def test_alpha_renaming_and_commutative_order_canonicalize(self):
        left = {
            "op": "exists", "var": "middle", "entity_type": "unit",
            "distinct_from": ["target", "actor"],
            "where": {"op": "and", "args": [
                {"op": "relation", "predicate": "linked", "source": "actor", "target": "middle"},
                {"op": "unary", "predicate": "stable", "var": "middle"},
            ]},
        }
        right = {
            "op": "exists", "var": "witness", "entity_type": "unit",
            "distinct_from": ["actor", "target"],
            "where": {"op": "and", "args": [
                {"op": "unary", "predicate": "stable", "var": "witness"},
                {"op": "relation", "predicate": "linked", "source": "actor", "target": "witness"},
            ]},
        }
        self.assertEqual(canonical_expression(left), canonical_expression(right))

    def test_relation_direction_changes_a_direct_program(self):
        entities = entities_for_layout({"units": 2, "hubs": 0})
        binding = action_bindings(self.config, entities)[0]
        target = next(
            value for value in target_hypotheses("relation_conditioned", 1)
            if value.component_names == ("actor_links_target",)
        )
        world = {atom: False for atom in atom_universe(self.config, entities)}
        world[f"r:linked:{binding['actor']}:{binding['target']}"] = True
        self.assertEqual(evaluate_program(target.program, self.config, entities, world, binding), "transition_1")
        world[f"r:linked:{binding['actor']}:{binding['target']}"] = False
        world[f"r:linked:{binding['target']}:{binding['actor']}"] = True
        self.assertEqual(evaluate_program(target.program, self.config, entities, world, binding), "transition_0")

    def test_partial_execution_returns_set_valued_answer(self):
        entities = entities_for_layout({"units": 2, "hubs": 0})
        binding = action_bindings(self.config, entities)[0]
        world = {atom: False for atom in atom_universe(self.config, entities)}
        atom = f"u:stable:{binding['actor']}"
        # Select the known actor-stable target so the unknown fact is genuinely sensitive.
        target = next(
            value for value in target_hypotheses("unary_selection", 1)
            if value.component_names == ("actor_stable",)
        )
        state = epistemic_from_world(self.config, entities, world, [atom])
        answer = execute_partial([target.program], self.config, entities, state, binding, 1)
        self.assertEqual(answer["possible_transition_codes"], ["transition_0", "transition_1"])
        self.assertFalse(answer["identifiable"])

    def test_permutation_preserves_hash_and_program_output(self):
        entities = entities_for_layout({"units": 3, "hubs": 1})
        binding = action_bindings(self.config, entities)[1]
        world = hashed_world(self.config, entities, "permutation-test")
        target = target_hypotheses("two_hop_composition", 2)[0]
        mapping = {
            "unit_0": "unit_2", "unit_1": "unit_1", "unit_2": "unit_0", "hub_0": "hub_0",
        }
        renamed_entities, renamed_world, renamed_binding = rename_state(
            entities, world, binding, mapping
        )
        self.assertEqual(
            canonical_state_hash(self.config, entities, world, binding),
            canonical_state_hash(self.config, renamed_entities, renamed_world, renamed_binding),
        )
        self.assertEqual(
            evaluate_program(target.program, self.config, entities, world, binding),
            evaluate_program(target.program, self.config, renamed_entities, renamed_world, renamed_binding),
        )

    def test_catalog_has_no_bounded_equivalent_pair(self):
        catalog = expression_catalog()
        for index, left in enumerate(catalog):
            for right in catalog[index + 1:]:
                self.assertIsNotNone(find_expression_counterexample(
                    left.expression, right.expression, self.config,
                    self.config["entityLayouts"],
                    self.config["limits"]["maximumTruthTableAtomsPerEquivalenceCheck"],
                ), msg=f"{left.name} == {right.name}")

    def test_counterexample_support_identifies_two_bit_target(self):
        target = target_hypotheses("existential_aggregation", 2)[0]
        hypotheses = enumerate_program_hypotheses(2)
        layouts = [
            value for value in self.config["entityLayouts"]
            if value["units"] + value["hubs"] in self.config["supportEntityCounts"]
        ]
        support, diagnostics = greedy_identifying_support(
            target, hypotheses, self.config, layouts,
            self.config["limits"]["maximumSupportTraces"],
        )
        self.assertLessEqual(len(support), self.config["limits"]["maximumSupportTraces"])
        self.assertEqual(diagnostics["remaining_hypotheses"], 1)


if __name__ == "__main__":
    unittest.main()
