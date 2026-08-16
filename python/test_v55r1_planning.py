from __future__ import annotations

import copy
import json
import unittest

from v22_relational import canonical_json, unary_atom
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, entities
from v53_smc2 import quadrature_rule
from v54_eig import prior_belief_atoms
from v55_planning import (
    assert_planning_payload_is_public,
    attempted_future_outcome_leak,
    candidate_actions,
    evaluate_policy,
    plan_exact,
    scalar_plan,
    step_belief,
    terminal_value,
)
from v55r1_planning import (
    delay_suppressed_registry,
    planning_registry,
    registry_audit,
    trigger_action,
)


class V55r1PlanningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads(
            (PROJECT_ROOT / "configs/v55r1-design-lock.json").read_text()
        )["config_payload"]
        cls.v53 = json.loads(
            (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
        )["config_payload"]
        cls.registry = planning_registry(cls.design)
        cls.suppressed = delay_suppressed_registry(cls.registry, 3)
        cls.entities = entities(2)

    def known_atom(self, program_index: int, theta: float, goal: dict) -> dict:
        world = deterministic_world(
            self.entities, f"v55r1-known-{program_index}-{goal['value']}"
        )
        world[goal["atom"]] = not goal["value"]
        return {
            "program_index": program_index,
            "node_index": 0,
            "theta": theta,
            "configuration_key": canonical_json({
                "world": sorted(world.items()), "queue": [],
            }),
            "world": world,
            "queue": [],
            "weight": 1.0,
        }

    def test_registry_matches_frozen_blueprints(self):
        audit = registry_audit(self.registry)
        self.assertEqual(8, audit["templates"])
        self.assertEqual(8, audit["unique_template_keys"])
        self.assertEqual(
            {"delay_two": 4, "delay_one": 2, "immediate": 2},
            audit["delay_class_counts"],
        )
        self.assertEqual(8, audit["theta_branches"])
        self.assertEqual(8, audit["active_stochastic_targets"])
        self.assertEqual(0, audit["active_deterministic_targets"])

    def test_complete_action_set_is_unchanged(self):
        actions = candidate_actions(self.entities)
        self.assertEqual(5, len(actions))
        self.assertEqual(5, len({row["key"] for row in actions}))

    def test_all_delay_two_blueprints_are_decision_relevant(self):
        for program_index, row in enumerate(self.registry[:4]):
            target_id = (
                self.entities[1]["id"]
                if row["blueprint"]["targetVar"] == "target"
                else self.entities[0]["id"]
            )
            goal = {
                "atom": unary_atom("active", target_id),
                "value": row["blueprint"]["operation"] == "set_true",
            }
            atom = self.known_atom(program_index, 0.8, goal)
            primary = plan_exact(
                [atom], self.registry, self.entities, goal, 3, 0, self.design
            )
            counterfactual = plan_exact(
                [atom], self.suppressed, self.entities, goal, 3, 0, self.design
            )
            self.assertGreater(primary["value"] - counterfactual["value"], 0.001)
            self.assertEqual(
                row["blueprint"]["trigger"], primary["selected_action"]["id"]
            )

    def test_delay_two_is_delivered_before_third_action(self):
        row = self.registry[0]
        actor, target = self.entities[0]["id"], self.entities[1]["id"]
        goal = {"atom": unary_atom("active", target), "value": False}
        atom = self.known_atom(0, 0.8, goal)
        belief = [atom]
        for tick, action in enumerate((
            trigger_action(row, actor, target),
            {"id": "wait", "binding": {}},
        )):
            branches = step_belief(
                belief, self.registry, self.entities, action, tick
            )
            belief = [
                {**item, "weight": branch["probability"] * item["weight"]}
                for branch in branches.values() for item in branch["atoms"]
            ]
        self.assertAlmostEqual(0.0, terminal_value(belief, goal), places=13)
        branches = step_belief(
            belief, self.registry, self.entities,
            {"id": "wait", "binding": {}}, 2,
        )
        belief = [
            {**item, "weight": branch["probability"] * item["weight"]}
            for branch in branches.values() for item in branch["atoms"]
        ]
        self.assertAlmostEqual(0.8, terminal_value(belief, goal), places=13)

    def test_primary_scalar_and_policy_agree(self):
        world = deterministic_world(self.entities, "v55r1-reference")
        goal = {
            "atom": unary_atom("active", self.entities[1]["id"]),
            "value": not world[unary_atom("active", self.entities[1]["id"])],
        }
        quadrature = quadrature_rule(3, self.v53["parameterModel"])
        atoms = prior_belief_atoms(self.registry[:2], quadrature, world)
        primary = plan_exact(
            atoms, self.registry[:2], self.entities, goal, 3, 0, self.design
        )
        reference = scalar_plan(
            atoms, self.registry[:2], self.entities, goal, 3, 0, self.design
        )
        evaluated = evaluate_policy(
            atoms, primary, self.registry[:2], self.entities, goal,
            3, 0, self.design,
        )
        self.assertAlmostEqual(primary["value"], reference["value"], places=13)
        self.assertAlmostEqual(primary["value"], evaluated, places=13)
        self.assertIn(primary["selected_action_key"], reference["optimal_action_keys"])

    def test_branches_normalize(self):
        world = deterministic_world(self.entities, "v55r1-normalize")
        quadrature = quadrature_rule(3, self.v53["parameterModel"])
        atoms = prior_belief_atoms(self.registry[:2], quadrature, world)
        for candidate in candidate_actions(self.entities):
            branches = step_belief(
                atoms, self.registry[:2], self.entities,
                candidate["action"], 0,
            )
            self.assertAlmostEqual(
                1.0, sum(branch["probability"] for branch in branches.values()),
                places=13,
            )
            for branch in branches.values():
                self.assertAlmostEqual(
                    1.0, sum(atom["weight"] for atom in branch["atoms"]),
                    places=13,
                )

    def test_firewalls_remain_active(self):
        with self.assertRaises(PermissionError):
            attempted_future_outcome_leak({}, {"future": True})
        with self.assertRaises(PermissionError):
            assert_planning_payload_is_public({"truth": {"secret": True}})


if __name__ == "__main__":
    unittest.main()
