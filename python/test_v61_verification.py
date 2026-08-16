#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v42_stateful import atom_universe, effect, entities, unary, world_signature
from v46_stochastic import _rule, canonical_program, delayed, stochastic
from v53_smc2 import parameterize_program
from v55_planning import candidate_actions
from v56_verification import (
    model_statistics, run_storm_properties, tool_versions,
    transition_rows_normalize, write_explicit_model,
)
from v59_planning import ActionStats, HistoryNode, SearchResult
from v61_verification import (
    compile_search_policy_dtmc,
    formal_transition_distribution,
    independent_deployment_action,
    independent_policy_statistics,
    independent_transition_distribution,
    verify_compiled_model_symbolically,
)


def blank_world():
    return {atom: False for atom in atom_universe(entities(2))}


def search_with_root(action_key: str, action_rows: list[dict]) -> SearchResult:
    root = HistoryNode(visits=10)
    root.actions = {row["key"]: ActionStats() for row in action_rows}
    root.actions[action_key].visits = 10
    root.actions[action_key].total_return = 5.0
    row = next(row for row in action_rows if row["key"] == action_key)
    return SearchResult(
        root=root, budget=10, simulations_run=10,
        selected_action=row["action"], selected_action_key=action_key,
        root_action_rows=[], root_sample_counts={}, tree_nodes=1,
        branching_action_nodes=0, visited_action_nodes=1,
        tree_sha256="fixture", merge_observations=False, seed=1,
    )


class V61VerificationTests(unittest.TestCase):
    def setUp(self):
        self.entities = entities(2)
        self.actions = candidate_actions(self.entities)
        self.pulse = next(
            row for row in self.actions
            if row["action"] == {
                "id": "pulse",
                "binding": {"actor": "unit_0", "target": "unit_1"},
            }
        )
        self.program = canonical_program({"rules": [
            _rule("pulse", stochastic_immediate=[stochastic(
                "1/4", effect("set_true", unary("active", "target"))
            )]),
            _rule("route", deterministic_immediate=[effect(
                "set_true", unary("marked", "actor")
            )]),
        ]})

    def test_tool_versions(self):
        self.assertEqual(tool_versions(), {"storm": "1.13.0", "z3": "4.16.0"})

    def test_independent_distribution_matches_formal_probability(self):
        left = independent_transition_distribution(
            self.program, self.entities, blank_world(), [],
            self.pulse["action"], 0,
        )
        right = formal_transition_distribution(
            self.program, self.entities, blank_world(), [],
            self.pulse["action"], 0,
        )
        self.assertEqual(
            [(row["mass"], row["world"], row["queue"]) for row in left],
            [(row["mass"], row["world"], row["queue"]) for row in right],
        )

    def test_due_queue_delivery_and_probability_mutants_differ(self):
        payload = effect("set_true", unary("active", "target"))
        queue = [{
            "due": 2, "effect": payload,
            "binding": {"actor": "unit_0", "target": "unit_1"},
        }]
        wait = {"id": "wait", "binding": {}}
        normal = independent_transition_distribution(
            self.program, self.entities, blank_world(), queue, wait, 2
        )
        omitted = independent_transition_distribution(
            self.program, self.entities, blank_world(), queue, wait, 2,
            "omit_due_queue_delivery",
        )
        self.assertNotEqual(normal[0]["world"], omitted[0]["world"])
        complemented = independent_transition_distribution(
            self.program, self.entities, blank_world(), [],
            self.pulse["action"], 0, "complement_stochastic_probability",
        )
        self.assertNotEqual(
            [row["mass"] for row in normal],
            [row["mass"] for row in complemented],
        )

    def test_independent_deployment_visit_and_fallback_rules(self):
        node = HistoryNode(visits=15)
        node.actions = {row["key"]: ActionStats() for row in self.actions}
        first, second = self.actions[:2]
        node.actions[first["key"]] = ActionStats(visits=10, total_return=1.0)
        node.actions[second["key"]] = ActionStats(visits=5, total_return=4.0)
        normal = independent_deployment_action(node, self.actions, [], 5953)
        mutant = independent_deployment_action(
            node, self.actions, [], 5953,
            "choose_maximum_mean_instead_of_maximum_visits",
        )
        self.assertEqual(normal["key"], first["key"])
        self.assertEqual(mutant["key"], second["key"])
        differences = [
            independent_deployment_action(None, self.actions, [{"n": n}], 5953)["key"]
            != independent_deployment_action(
                None, self.actions, [{"n": n}], 5953,
                "change_public_history_fallback_seed",
            )["key"]
            for n in range(32)
        ]
        self.assertTrue(any(differences))

    def test_compiler_independent_executor_and_storm_round_trip(self):
        registry = [{"template": parameterize_program(self.program)}]
        atoms = [{
            "program_index": 0, "node_index": 0, "theta": 0.25,
            "configuration_key": "fixture", "world": blank_world(),
            "queue": [], "weight": 1.0,
        }]
        search = search_with_root(self.pulse["key"], self.actions)
        goal = {"atom": "u:active:unit_1", "value": True}
        config = {
            "planningModel": {"actionCost": {
                "pulse": 0.01, "route": 0.01, "wait": 0.0,
            }},
            "evaluation": {"evaluationSeed": 5953},
        }
        model = compile_search_policy_dtmc(
            atoms, search, registry, self.entities, goal, 1, 0, config
        )
        direct = independent_policy_statistics(
            atoms, search, registry, self.entities, goal, 1, 0, config
        )
        graph = model_statistics(model)
        symbolic = verify_compiled_model_symbolically(model)
        self.assertAlmostEqual(direct["success_probability"], 0.25)
        self.assertAlmostEqual(direct["expected_return"], 0.24)
        self.assertAlmostEqual(graph["expected_return"], 0.24)
        self.assertTrue(transition_rows_normalize(model))
        self.assertEqual(symbolic["support_checks"], symbolic["support_passes"])
        self.assertEqual(symbolic["probability_passes"], symbolic["support_checks"])
        self.assertEqual(symbolic["deployment_checks"], symbolic["deployment_passes"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            write_explicit_model(model, path)
            storm = run_storm_properties(path)
        self.assertEqual(storm, graph)

    def test_delayed_two_tick_fixture(self):
        payload = effect("set_true", unary("active", "target"))
        program = canonical_program({"rules": [
            _rule("pulse", stochastic_delayed=[delayed(
                2, stochastic("1/2", payload)
            )]),
            _rule("route"),
        ]})
        first = next(row for row in independent_transition_distribution(
            program, self.entities, blank_world(), [], self.pulse["action"], 0
        ) if row["queue"])
        wait = {"id": "wait", "binding": {}}
        second = independent_transition_distribution(
            program, self.entities, first["world"], first["queue"], wait, 1
        )[0]
        third = independent_transition_distribution(
            program, self.entities, second["world"], second["queue"], wait, 2
        )[0]
        self.assertFalse(second["world"]["u:active:unit_1"])
        self.assertTrue(third["world"]["u:active:unit_1"])


if __name__ == "__main__":
    unittest.main()
