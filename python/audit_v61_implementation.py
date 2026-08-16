#!/usr/bin/env python3
"""Audit V61 verifier controls before source policies become accessible."""
from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
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


ENTITY_ROWS = entities(2)
ACTIONS = candidate_actions(ENTITY_ROWS)
CONFIG = {
    "planningModel": {"actionCost": {
        "pulse": 0.01, "route": 0.01, "wait": 0.0,
    }},
    "evaluation": {"evaluationSeed": 5953},
}


def blank_world():
    return {atom: False for atom in atom_universe(ENTITY_ROWS)}


def action_row(action_id="pulse", actor="unit_0", target="unit_1"):
    binding = {} if action_id == "wait" else {"actor": actor, "target": target}
    return next(
        row for row in ACTIONS
        if row["action"] == {"id": action_id, "binding": binding}
    )


def selected_node(row: dict, visits=10, total=5.0) -> HistoryNode:
    node = HistoryNode(visits=visits)
    node.actions = {candidate["key"]: ActionStats() for candidate in ACTIONS}
    node.actions[row["key"]] = ActionStats(visits=visits, total_return=total)
    return node


def search(root: HistoryNode) -> SearchResult:
    chosen = independent_deployment_action(root, ACTIONS, [], 5953)
    return SearchResult(
        root=root, budget=10, simulations_run=10,
        selected_action=chosen["action"], selected_action_key=chosen["key"],
        root_action_rows=[], root_sample_counts={}, tree_nodes=1,
        branching_action_nodes=0, visited_action_nodes=1,
        tree_sha256="altered-seed-fixture", merge_observations=False, seed=6103,
    )


def atoms(theta=0.5):
    return [{
        "program_index": 0, "node_index": 0, "theta": theta,
        "configuration_key": "fixture", "world": blank_world(),
        "queue": [], "weight": 1.0,
    }]


def registry(program):
    return [{"template": parameterize_program(program)}]


def immediate_program(probability="1/2"):
    return canonical_program({"rules": [
        _rule("pulse", stochastic_immediate=[stochastic(
            probability, effect("set_true", unary("active", "target"))
        )]),
        _rule("route"),
    ]})


def evaluate_fixture(name, program, root, goal, horizon, theta, expected):
    model = compile_search_policy_dtmc(
        atoms(theta), search(root), registry(program), ENTITY_ROWS,
        goal, horizon, 0, CONFIG,
    )
    direct = independent_policy_statistics(
        atoms(theta), search(root), registry(program), ENTITY_ROWS,
        goal, horizon, 0, CONFIG,
    )
    symbolic = verify_compiled_model_symbolically(model)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        write_explicit_model(model, directory)
        storm = run_storm_properties(directory)
    observed = (
        storm["termination_probability"], storm["success_probability"],
        storm["expected_return"],
    )
    error = max(abs(left - right) for left, right in zip(observed, expected, strict=True))
    reference_error = max(
        abs(storm[key] - direct[key])
        for key in ("success_probability", "expected_return")
    )
    passed = (
        error <= 1e-12
        and reference_error <= 1e-12
        and transition_rows_normalize(model)
        and symbolic["support_checks"] == symbolic["support_passes"]
        and symbolic["support_checks"] == symbolic["probability_passes"]
        and symbolic["deployment_checks"] == symbolic["deployment_passes"]
    )
    return {
        "fixture": name, "expected": expected, "observed": observed,
        "maximum_expected_error": error,
        "maximum_independent_reference_error": reference_error,
        "passed": passed,
    }


def analytic_fixtures():
    pulse, wait = action_row(), action_row("wait")
    goal_irrelevant_stochastic = immediate_program("1/2")
    fallback = independent_deployment_action(None, ACTIONS, [], 5953)
    fixtures = [
        evaluate_fixture(
            "deterministic_success_with_public_fallback", goal_irrelevant_stochastic,
            HistoryNode(), {"atom": "u:ready:unit_0", "value": False},
            1, 0.5, (
                1.0, 1.0,
                1.0 - CONFIG["planningModel"]["actionCost"][fallback["action"]["id"]],
            ),
        )
    ]
    fixtures.append(evaluate_fixture(
        "bernoulli_one_quarter_immediate_success", immediate_program("1/4"),
        selected_node(pulse), {"atom": "u:active:unit_1", "value": True},
        1, 0.25, (1.0, 0.25, 0.24),
    ))

    delayed_program = canonical_program({"rules": [
        _rule("pulse", stochastic_delayed=[delayed(
            2, stochastic("1/2", effect("set_true", unary("active", "target")))
        )]),
        _rule("route"),
    ]})
    delayed_root = selected_node(pulse)
    observation = world_signature(blank_world())
    delayed_wait_1 = selected_node(wait)
    delayed_wait_2 = selected_node(wait)
    delayed_root.actions[pulse["key"]].children[observation] = delayed_wait_1
    delayed_wait_1.actions[wait["key"]].children[observation] = delayed_wait_2
    fixtures.append(evaluate_fixture(
        "two_tick_delayed_success", delayed_program, delayed_root,
        {"atom": "u:active:unit_1", "value": True}, 3, 0.5,
        (1.0, 0.5, 0.49),
    ))

    contingent_root = selected_node(pulse)
    false_observation = world_signature(blank_world())
    true_world = blank_world()
    true_world["u:active:unit_1"] = True
    true_observation = world_signature(true_world)
    contingent_root.actions[pulse["key"]].children[false_observation] = selected_node(pulse)
    contingent_root.actions[pulse["key"]].children[true_observation] = selected_node(wait)
    fixtures.append(evaluate_fixture(
        "observation_contingent_action_change", immediate_program("1/2"),
        contingent_root, {"atom": "u:active:unit_1", "value": True},
        2, 0.5, (1.0, 0.75, 0.735),
    ))
    fixtures.append(evaluate_fixture(
        "negative_action_cost_plus_terminal_reward", immediate_program("3/4"),
        selected_node(pulse), {"atom": "u:active:unit_1", "value": True},
        1, 0.75, (1.0, 0.75, 0.74),
    ))
    missing_history = [{
        "action_key": wait["key"], "observation": world_signature(blank_world())
    }]
    missing_fallback = independent_deployment_action(
        None, ACTIONS, missing_history, 5953
    )
    fixtures.append(evaluate_fixture(
        "tree_child_missing_uses_public_history_fallback", goal_irrelevant_stochastic,
        selected_node(wait), {"atom": "u:ready:unit_0", "value": False},
        2, 0.5, (
            1.0, 1.0,
            1.0 - CONFIG["planningModel"]["actionCost"][missing_fallback["action"]["id"]],
        ),
    ))
    return fixtures


def mutation_controls():
    rows = []
    pulse = action_row()
    program = immediate_program("1/4")
    normal = independent_transition_distribution(
        program, ENTITY_ROWS, blank_world(), [], pulse["action"], 0
    )
    formal = formal_transition_distribution(
        program, ENTITY_ROWS, blank_world(), [], pulse["action"], 0
    )
    for mutant in ("swap_actor_and_target", "complement_stochastic_probability"):
        changed = independent_transition_distribution(
            program, ENTITY_ROWS, blank_world(), [], pulse["action"], 0, mutant
        )
        rows.append({"mutant": mutant, "killed": changed != formal})

    due = [{
        "due": 0,
        "effect": effect("set_true", unary("active", "target")),
        "binding": {"actor": "unit_0", "target": "unit_1"},
    }]
    wait = action_row("wait")
    delivered = independent_transition_distribution(
        program, ENTITY_ROWS, blank_world(), due, wait["action"], 0
    )
    omitted = independent_transition_distribution(
        program, ENTITY_ROWS, blank_world(), due, wait["action"], 0,
        "omit_due_queue_delivery",
    )
    rows.append({"mutant": "omit_due_queue_delivery", "killed": delivered != omitted})

    node = HistoryNode(visits=15)
    node.actions = {row["key"]: ActionStats() for row in ACTIONS}
    node.actions[ACTIONS[0]["key"]] = ActionStats(visits=10, total_return=1.0)
    node.actions[ACTIONS[1]["key"]] = ActionStats(visits=5, total_return=4.0)
    normal_action = independent_deployment_action(node, ACTIONS, [], 5953)
    mean_action = independent_deployment_action(
        node, ACTIONS, [], 5953, "choose_maximum_mean_instead_of_maximum_visits"
    )
    rows.append({
        "mutant": "choose_maximum_mean_instead_of_maximum_visits",
        "killed": normal_action["key"] != mean_action["key"],
    })
    fallback_changed = any(
        independent_deployment_action(None, ACTIONS, [{"n": n}], 5953)["key"]
        != independent_deployment_action(
            None, ACTIONS, [{"n": n}], 5953,
            "change_public_history_fallback_seed",
        )["key"]
        for n in range(64)
    )
    rows.append({
        "mutant": "change_public_history_fallback_seed", "killed": fallback_changed
    })

    goal = {"atom": "u:active:unit_1", "value": True}
    base_search = search(selected_node(pulse))
    direct = independent_policy_statistics(
        atoms(0.25), base_search, registry(program), ENTITY_ROWS, goal, 1, 0, CONFIG
    )
    for mutant in (
        "drop_one_stochastic_successor", "flip_terminal_success_label",
        "omit_action_cost_reward", "corrupt_initial_distribution_mass",
    ):
        model = compile_search_policy_dtmc(
            atoms(0.25), base_search, registry(program), ENTITY_ROWS,
            goal, 1, 0, CONFIG, mutant,
        )
        graph = model_statistics(model)
        if mutant in {"drop_one_stochastic_successor", "corrupt_initial_distribution_mass"}:
            killed = not transition_rows_normalize(model)
        elif mutant == "flip_terminal_success_label":
            killed = abs(graph["success_probability"] - direct["success_probability"]) > 0.1
        else:
            killed = abs(graph["expected_return"] - direct["expected_return"]) > 0.005
        rows.append({"mutant": mutant, "killed": killed})

    contingent_root = selected_node(pulse)
    false_observation = world_signature(blank_world())
    true_world = blank_world(); true_world["u:active:unit_1"] = True
    contingent_root.actions[pulse["key"]].children[false_observation] = selected_node(pulse)
    contingent_root.actions[pulse["key"]].children[world_signature(true_world)] = selected_node(wait)
    contingent_search = search(contingent_root)
    standard = compile_search_policy_dtmc(
        atoms(0.5), contingent_search, registry(immediate_program("1/2")),
        ENTITY_ROWS, goal, 2, 0, CONFIG,
    )
    merged = compile_search_policy_dtmc(
        atoms(0.5), contingent_search, registry(immediate_program("1/2")),
        ENTITY_ROWS, goal, 2, 0, CONFIG, "merge_observation_routes",
    )
    rows.append({
        "mutant": "merge_observation_routes",
        "killed": abs(
            model_statistics(standard)["expected_return"]
            - model_statistics(merged)["expected_return"]
        ) > 1e-6,
    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v61-design-lock.json")
    parser.add_argument("--implementation", default="python/v61_verification.py")
    parser.add_argument("--tests", default="python/test_v61_verification.py")
    parser.add_argument(
        "--output", default="outputs/v61-long-horizon-policy-verification/implementation-audit.json"
    )
    args = parser.parse_args()
    design_path, implementation_path, tests_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.design_lock, args.implementation, args.tests, args.output)
    )
    design = json.loads(design_path.read_text())
    if not design["authorization"]["write_and_audit_v61_implementation"]:
        raise RuntimeError("V61 design does not authorize implementation")
    test_run = subprocess.run(
        [sys.executable, "-m", "unittest", str(tests_path.relative_to(PROJECT_ROOT)), "-v"],
        cwd=PROJECT_ROOT, env={**__import__("os").environ, "PYTHONPATH": "python"},
        capture_output=True, text=True,
    )
    fixtures = analytic_fixtures()
    mutants = mutation_controls()
    fixture_rate = sum(row["passed"] for row in fixtures) / len(fixtures)
    kill_rate = sum(row["killed"] for row in mutants) / len(mutants)
    transition_source = inspect.getsource(independent_transition_distribution)
    evaluator_source = inspect.getsource(independent_policy_statistics)
    independence = {
        "independent_transition_does_not_call_formal_executor": (
            "continuous_unit_transition" not in transition_source
        ),
        "independent_evaluator_does_not_call_v59_policy_episode": (
            "_policy_episode" not in evaluator_source
        ),
        "independent_evaluator_uses_separate_deployment_interpreter": (
            "independent_deployment_action" in evaluator_source
        ),
    }
    versions = tool_versions()
    expected_versions = {
        "storm": design["config_payload"]["probabilisticVerification"]["version"],
        "z3": design["config_payload"]["independentVerification"]["z3SolverVersion"],
    }
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v61-implementation-lock.json",
            "configs/v61-verification-bundle-seal.json",
            "outputs/v61-long-horizon-policy-verification/verification-bundle",
            "outputs/v61-long-horizon-policy-verification/verification-attempt.json",
        )
    )
    passed = (
        test_run.returncode == 0
        and fixture_rate == 1.0
        and kill_rate == 1.0
        and all(independence.values())
        and versions == expected_versions
        and len(mutants) == 10
        and downstream_absent
    )
    audit = {
        "schema_version": 61,
        "experiment": "v61_implementation_audit",
        "passed": passed,
        "decision": "freeze_v61_implementation" if passed else "repair_v61_implementation",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_sha256": file_sha256(implementation_path),
        "unit_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "unit_tests_sha256": file_sha256(tests_path),
        "unit_test_run": {
            "returncode": test_run.returncode,
            "stdout": test_run.stdout,
            "stderr": test_run.stderr,
        },
        "analytic_fixtures": fixtures,
        "analytic_fixture_pass_rate": fixture_rate,
        "mutation_controls": mutants,
        "mutation_kill_rate": kill_rate,
        "independence_checks": independence,
        "tool_versions": versions,
        "expected_tool_versions": expected_versions,
        "downstream_absence": downstream_absent,
        "data_access": {
            "v59_candidate_public_records_accessed": 0,
            "v59_audit_truth_records_accessed": 0,
            "v60_source_policy_cells_accessed": 0,
            "v61_verification_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed, "fixture_rate": fixture_rate,
        "mutation_kill_rate": kill_rate, "mutants": mutants,
        "independence": independence, "tool_versions": versions,
        "unit_tests": test_run.returncode,
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
