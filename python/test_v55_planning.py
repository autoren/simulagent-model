from __future__ import annotations

import copy
import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v22_relational import unary_atom
from v42_stateful import deterministic_world, entities
from v53_smc2 import mechanic_registry, quadrature_rule
from v54_eig import prior_belief_atoms
from v55_planning import (
    attempted_future_outcome_leak,
    best_open_loop,
    candidate_actions,
    clairvoyant_value,
    disabled_static_step,
    eig_policy_value,
    evaluate_policy,
    evaluate_static_update_disabled_policy,
    greedy_policy_value,
    map_program_policy_value,
    plan_exact,
    posterior_mean_theta_policy_value,
    scalar_plan,
    static_marginal,
    step_belief,
    terminal_value,
)


class V55PlanningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (PROJECT_ROOT / "configs/v55-design-lock.json").read_text()
        )["config_payload"]
        cls.v53 = json.loads(
            (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
        )["config_payload"]
        cls.registry = mechanic_registry(5303)
        cls.entities = entities(2)
        cls.world = deterministic_world(cls.entities, "v55-unit")
        rule = quadrature_rule(5, cls.v53["parameterModel"])
        cls.atoms = prior_belief_atoms(cls.registry[:2], rule, cls.world)
        cls.goal = {"atom": sorted(cls.world)[0], "value": not cls.world[sorted(cls.world)[0]]}

    def test_complete_action_set(self):
        self.assertEqual(5, len(candidate_actions(self.entities)))

    def test_horizon_zero_is_terminal_goal_probability(self):
        result = plan_exact(
            self.atoms, self.registry[:2], self.entities, self.goal,
            0, 0, self.config,
        )
        self.assertEqual(terminal_value(self.atoms, self.goal), result["value"])

    def test_one_step_matches_scalar_reference_and_policy_evaluation(self):
        primary = plan_exact(
            self.atoms, self.registry[:2], self.entities, self.goal,
            1, 0, self.config,
        )
        reference = scalar_plan(
            self.atoms, self.registry[:2], self.entities, self.goal,
            1, 0, self.config,
        )
        self.assertAlmostEqual(primary["value"], reference["value"], places=13)
        self.assertIn(primary["selected_action_key"], reference["optimal_action_keys"])
        evaluated = evaluate_policy(
            self.atoms, primary, self.registry[:2], self.entities,
            self.goal, 1, 0, self.config,
        )
        self.assertAlmostEqual(primary["value"], evaluated, places=13)

    def test_two_step_adaptive_value_dominates_open_loop(self):
        primary = plan_exact(
            self.atoms, self.registry[:2], self.entities, self.goal,
            2, 0, self.config,
        )
        open_loop = best_open_loop(
            self.atoms, self.registry[:2], self.entities, self.goal,
            2, 0, self.config,
        )
        self.assertGreaterEqual(primary["value"] + 1e-12, open_loop["value"])

    def test_registered_baselines_are_executable_and_bounded(self):
        primary = plan_exact(
            self.atoms, self.registry[:2], self.entities, self.goal,
            1, 0, self.config,
        )["value"]
        values = (
            greedy_policy_value(
                self.atoms, self.registry[:2], self.entities, self.goal,
                1, 0, self.config,
            ),
            eig_policy_value(
                self.atoms, self.registry[:2], self.entities, self.goal,
                1, 0, self.config,
            ),
            map_program_policy_value(
                self.atoms, self.registry[:2], self.entities, self.goal,
                1, 0, self.config,
            ),
            posterior_mean_theta_policy_value(
                self.atoms, self.registry[:2], self.entities, self.goal,
                1, 0, self.config,
            ),
            evaluate_static_update_disabled_policy(
                self.atoms, self.atoms, self.registry[:2], self.entities,
                self.goal, 1, 0, self.config,
            ),
        )
        self.assertTrue(all(primary + 1e-12 >= value for value in values))
        self.assertGreaterEqual(
            clairvoyant_value(
                self.atoms, self.registry[:2], self.entities, self.goal,
                1, 0, self.config,
            ) + 1e-12,
            primary,
        )

    def test_disabled_update_holds_static_marginal_fixed(self):
        target = static_marginal(self.atoms)
        action = candidate_actions(self.entities)[0]["action"]
        branches = disabled_static_step(
            self.atoms, self.registry[:2], self.entities, action, 0, target
        )
        for branch in branches.values():
            actual = static_marginal(branch["atoms"])
            self.assertEqual(set(target), set(actual))
            for key in target:
                self.assertAlmostEqual(target[key], actual[key], places=13)

    def test_delay_two_effect_is_visible_only_after_third_action(self):
        registry = [self.registry[5]]
        atom = copy.deepcopy(self.atoms[0])
        atom.update({
            "program_index": 0,
            "node_index": 0,
            "theta": 0.8,
            "weight": 1.0,
        })
        target = self.entities[1]["id"]
        goal_atom = unary_atom("active", target)
        atom["world"][goal_atom] = True
        atom["configuration_key"] = json.dumps({
            "queue": [],
            "world": sorted(atom["world"].items()),
        }, sort_keys=True, separators=(",", ":"))
        pulse = {
            "id": "pulse",
            "binding": {
                "actor": self.entities[0]["id"],
                "target": target,
            },
        }
        wait = {"id": "wait", "binding": {}}
        belief = [atom]
        for tick, action in enumerate((pulse, wait)):
            branches = step_belief(belief, registry, self.entities, action, tick)
            belief = [
                {**row, "weight": branch["probability"] * row["weight"]}
                for branch in branches.values() for row in branch["atoms"]
            ]
        before = terminal_value(belief, {"atom": goal_atom, "value": False})
        branches = step_belief(belief, registry, self.entities, wait, 2)
        belief = [
            {**row, "weight": branch["probability"] * row["weight"]}
            for branch in branches.values() for row in branch["atoms"]
        ]
        after = terminal_value(belief, {"atom": goal_atom, "value": False})
        self.assertAlmostEqual(0.0, before, places=13)
        self.assertAlmostEqual(0.8, after, places=13)

    def test_step_beliefs_normalize(self):
        for candidate in candidate_actions(self.entities):
            branches = step_belief(
                self.atoms, self.registry[:2], self.entities,
                candidate["action"], 0,
            )
            self.assertAlmostEqual(1.0, sum(
                branch["probability"] for branch in branches.values()
            ), places=13)
            for branch in branches.values():
                self.assertAlmostEqual(1.0, sum(
                    atom["weight"] for atom in branch["atoms"]
                ), places=13)

    def test_future_outcome_leak_is_rejected(self):
        with self.assertRaises(PermissionError):
            attempted_future_outcome_leak({}, {"future": True})


if __name__ == "__main__":
    unittest.main()
