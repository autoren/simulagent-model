#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from v42_stateful import atom_universe, effect, entities, relation, unary
from v46_stochastic import _rule, canonical_program, delayed, stochastic
from v53_smc2 import parameterize_program
from v56_verification import (
    compile_policy_dtmc,
    direct_policy_statistics,
    formal_transition_support,
    independent_transition_support,
    model_statistics,
    prove_support_equivalence,
    run_storm_properties,
    tool_versions,
    transition_rows_normalize,
    validate_world_queue_action,
    verify_compiled_model_symbolically,
    write_explicit_model,
)


def world(value=False):
    return {atom: value for atom in atom_universe(entities(2))}


def tiny_model(probability=0.25, success_reward=1.0, action_cost=0.0):
    # root -> success/failure staging -> done
    return {
        "states": [
            {"id": 0, "kind": "root"},
            {"id": 1, "kind": "terminal", "success": True},
            {"id": 2, "kind": "terminal", "success": False},
            {"id": 3, "kind": "done"},
        ],
        "transitions": [
            {"source": 0, "target": 1, "probability": probability, "reward": -action_cost, "annotations": []},
            {"source": 0, "target": 2, "probability": 1 - probability, "reward": -action_cost, "annotations": []},
            {"source": 1, "target": 3, "probability": 1.0, "reward": success_reward, "annotations": []},
            {"source": 2, "target": 3, "probability": 1.0, "reward": 0.0, "annotations": []},
            {"source": 3, "target": 3, "probability": 1.0, "reward": 0.0, "annotations": []},
        ],
        "root_state": 0,
        "done_state": 3,
    }


class V56VerificationTests(unittest.TestCase):
    def setUp(self):
        self.entities = entities(2)
        self.action = {
            "id": "pulse",
            "binding": {"actor": "unit_0", "target": "unit_1"},
        }

    def test_tool_versions(self):
        self.assertEqual(tool_versions(), {"storm": "1.13.0", "z3": "4.16.0"})

    def test_independent_support_matches_immediate_formal_executor(self):
        program = canonical_program({"rules": [
            _rule("pulse", stochastic_immediate=[stochastic(
                "1/2", effect("toggle", unary("active", "target"))
            )]),
            _rule("route", deterministic_immediate=[effect(
                "toggle", relation("actor", "target")
            )]),
        ]})
        left = independent_transition_support(
            program, self.entities, world(), [], self.action, 0
        )
        right = formal_transition_support(
            program, self.entities, world(), [], self.action, 0
        )
        self.assertTrue(prove_support_equivalence(left, right)["equivalent"])

    def test_independent_support_matches_delayed_delivery(self):
        payload = effect("set_true", unary("active", "target"))
        program = canonical_program({"rules": [
            _rule("pulse", stochastic_delayed=[delayed(
                2, stochastic("1/2", payload)
            )]),
            _rule("route", deterministic_immediate=[effect(
                "toggle", unary("marked", "actor")
            )]),
        ]})
        queue = [{
            "due": 1, "effect": payload,
            "binding": {"actor": "unit_0", "target": "unit_1"},
        }]
        wait = {"id": "wait", "binding": {}}
        left = independent_transition_support(
            program, self.entities, world(), queue, wait, 1
        )
        right = formal_transition_support(
            program, self.entities, world(), queue, wait, 1
        )
        self.assertTrue(prove_support_equivalence(left, right)["equivalent"])
        self.assertTrue(left[0]["world"]["u:active:unit_1"])

    def test_support_mismatch_is_satisfiable(self):
        first = [{"world": world(False), "queue": []}]
        second_world = world(False)
        second_world["u:active:unit_0"] = True
        proof = prove_support_equivalence(
            first, [{"world": second_world, "queue": []}]
        )
        self.assertEqual(proof["status"], "sat")
        self.assertFalse(proof["equivalent"])

    def test_action_and_queue_validation(self):
        self.assertTrue(validate_world_queue_action(
            self.entities, world(), [], self.action, 0
        ))
        self.assertFalse(validate_world_queue_action(
            self.entities, world(), [],
            {"id": "pulse", "binding": {"actor": "unit_0", "target": "unit_0"}},
            0,
        ))

    def test_storm_known_bernoulli_and_reward(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            write_explicit_model(tiny_model(0.25, 1.0, 0.01), path)
            result = run_storm_properties(path)
        self.assertAlmostEqual(result["termination_probability"], 1.0, places=12)
        self.assertAlmostEqual(result["success_probability"], 0.25, places=12)
        self.assertAlmostEqual(result["expected_return"], 0.24, places=12)

    def test_policy_compiler_round_trip(self):
        program = canonical_program({"rules": [
            _rule("pulse", stochastic_immediate=[stochastic(
                "1/2", effect("toggle", unary("active", "target"))
            )]),
            _rule("route", deterministic_immediate=[effect(
                "toggle", relation("actor", "target")
            )]),
        ]})
        registry = [{"template": parameterize_program(program)}]
        observation = __import__("v42_stateful").world_signature(world())

        def wait_policy(depth):
            if depth == 0:
                return {"terminal": True, "value": 1.0}
            child = wait_policy(depth - 1)
            return {
                "terminal": False,
                "value": 1.0,
                "selected_action": {"id": "wait", "binding": {}},
                "selected_action_key": '{"binding":{},"id":"wait"}',
                "branches": {observation: child},
                "observation_probabilities": {observation: 1.0},
                "action_values": {'{"binding":{},"id":"wait"}': 1.0},
                "optimal_action_keys": ['{"binding":{},"id":"wait"}'],
            }

        atoms = [{
            "program_index": 0,
            "node_index": 0,
            "theta": 0.5,
            "configuration_key": "fixture",
            "world": world(),
            "queue": [],
            "weight": 1.0,
        }]
        policy = wait_policy(3)
        goal = {"atom": "u:active:unit_0", "value": False}
        config = {"formalExecutor": {"actionCosts": {
            "pulse": 0.01, "route": 0.01, "wait": 0.0,
        }}}
        model = compile_policy_dtmc(
            atoms, policy, registry, self.entities, goal, 3, 0, config
        )
        direct = direct_policy_statistics(
            atoms, policy, registry, self.entities, goal, 3, 0, config
        )
        graph = model_statistics(model)
        symbolic = verify_compiled_model_symbolically(model)
        self.assertTrue(transition_rows_normalize(model))
        self.assertEqual(direct, {"success_probability": 1.0, "expected_return": 1.0})
        self.assertEqual(graph, {
            "success_probability": 1.0,
            "expected_return": 1.0,
            "termination_probability": 1.0,
        })
        self.assertEqual(symbolic["support_checks"], symbolic["support_passes"])
        self.assertEqual(symbolic["invariant_checks"], symbolic["invariant_passes"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            write_explicit_model(model, path)
            storm = run_storm_properties(path)
        self.assertEqual(storm, graph)

    def test_corrupt_transition_mass_is_rejected_by_storm(self):
        model = tiny_model(0.25)
        model["transitions"][0]["probability"] = 0.20
        self.assertFalse(transition_rows_normalize(model))


if __name__ == "__main__":
    unittest.main()
