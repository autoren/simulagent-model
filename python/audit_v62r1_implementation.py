#!/usr/bin/env python3
"""Audit and mutation-test the V62r1 measurement-repair implementation."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys

import numpy as np

from test_v62r1_terminal_residual import (
    FixedPlanner,
    one_step_fixture,
    terminal_reward_fixture,
    two_step_observation_fixture,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import Decision, ExactPlanner, bellman_residual
from v62r1_terminal_residual import terminal_aware_bellman_residual


def mutant_any_action_terminal(model, planner, belief, horizon: int) -> float:
    support = np.flatnonzero(np.asarray(belief) > 1e-14)
    any_absorbing = False
    for state in support:
        for action in range(len(model.actions)):
            target = np.zeros(len(model.states))
            target[state] = 1.0
            any_absorbing |= bool(np.array_equal(model.transition[action, state], target))
    if horizon <= 0 or any_absorbing:
        decision = planner.decision(belief, horizon)
        return max(abs(decision.value), *(abs(value) for value in decision.q_values))
    return terminal_aware_bellman_residual(model, planner, belief, horizon)


def mutant_nonterminal_recomposition(
    model, planner, belief, horizon: int, *, omit_discount: bool, pretransition_obs: bool
) -> float:
    decision = planner.decision(belief, horizon)
    recomposed = []
    for action in range(len(model.actions)):
        value = float(
            sum(
                belief[state]
                * model.transition[action, state, successor]
                * model.reward[action, state, successor]
                for state in range(len(model.states))
                for successor in range(len(model.states))
            )
        )
        continuation = 0.0
        if horizon > 1:
            predicted = np.asarray(belief) @ model.transition[action]
            for observation in range(len(model.observations)):
                if pretransition_obs:
                    joint = np.asarray(belief) * model.observation[action, :, observation]
                else:
                    joint = predicted * model.observation[action, :, observation]
                probability = float(joint.sum())
                if probability > 1e-15:
                    continuation += probability * planner.decision(
                        joint / probability, horizon - 1
                    ).value
        value += (1.0 if omit_discount else model.discount) * continuation
        recomposed.append(value)
    return max(
        abs(recomposed[action] - decision.q_values[action])
        for action in range(len(model.actions))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v62r1-design-lock.json")
    parser.add_argument(
        "--implementation", default="python/v62r1_terminal_residual.py"
    )
    parser.add_argument("--tests", default="python/test_v62r1_terminal_residual.py")
    parser.add_argument(
        "--output",
        default="outputs/v62r1-terminal-residual-repair/implementation-audit.json",
    )
    args = parser.parse_args()
    design_path, implementation_path, tests_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.design_lock, args.implementation, args.tests, args.output)
    )
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []

    source_bindings_ok = (
        design["authorization"]["write_and_audit_repair_implementation"]
        and file_sha256(PROJECT_ROOT / design["source_v62_outcome_lock"])
        == design["source_v62_outcome_lock_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_v62_evaluation_implementation_lock"])
        == design["source_v62_evaluation_implementation_lock_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_v62_external_bundle_seal"])
        == design["source_v62_external_bundle_seal_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_post_hoc_diagnostic"])
        == design["source_post_hoc_diagnostic_sha256"]
    )
    if not source_bindings_ok:
        errors.append("V62r1 design or immutable source bindings changed")

    tree = ast.parse(implementation_path.read_text())
    forbidden_calls = {
        "bellman_residual",
        "expected_reward",
        "observation_distribution",
        "update_belief",
        "terminal_mask",
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    independent_checker = not (forbidden_calls & calls) and not (
        forbidden_calls & imports
    )
    if not independent_checker:
        errors.append("repair checker reuses a forbidden V62 calculation helper")

    test_run = subprocess.run(
        [sys.executable, str(tests_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    fixture_count = test_run.stdout.count(".") + test_run.stderr.splitlines()[0].count(".") if test_run.stderr else 0
    fixture_pass = test_run.returncode == 0
    if not fixture_pass:
        errors.append("V62r1 analytic fixture tests failed")

    terminal_model, terminal_belief = terminal_reward_fixture()
    terminal_planner = ExactPlanner(terminal_model)
    one_step_model, one_step_belief = one_step_fixture()
    one_step_planner = ExactPlanner(one_step_model)
    branch_model, branch_belief = two_step_observation_fixture()
    branch_planner = ExactPlanner(branch_model)
    q_bad = FixedPlanner(Decision(0, 0.0, (0.0, 0.5), (0,)))
    value_bad = FixedPlanner(Decision(0, 0.5, (0.0, 0.0), (0, 1)))
    mutant_residuals = {
        "omit_terminal_support_base_case": bellman_residual(
            terminal_model, terminal_planner, terminal_belief, 3
        ),
        "declare_terminal_if_any_action_absorbs": mutant_any_action_terminal(
            one_step_model, one_step_planner, one_step_belief, 1
        ),
        "ignore_terminal_q_values": abs(q_bad.decision(terminal_belief, 2).value),
        "ignore_terminal_decision_value": max(
            abs(value)
            for value in value_bad.decision(terminal_belief, 2).q_values
        ),
        "omit_discount_from_nonterminal_backup": mutant_nonterminal_recomposition(
            branch_model,
            branch_planner,
            branch_belief,
            2,
            omit_discount=True,
            pretransition_obs=False,
        ),
        "use_pretransition_observation_likelihood": mutant_nonterminal_recomposition(
            branch_model,
            branch_planner,
            branch_belief,
            2,
            omit_discount=False,
            pretransition_obs=True,
        ),
    }
    # The value/Q mutants return zero because they skip the corrupted field; the
    # production checker must detect the same corruption.
    production_corruption_residuals = {
        "ignore_terminal_q_values": terminal_aware_bellman_residual(
            terminal_model, q_bad, terminal_belief, 2
        ),
        "ignore_terminal_decision_value": terminal_aware_bellman_residual(
            terminal_model, value_bad, terminal_belief, 2
        ),
    }
    killed = {
        name: (
            residual > 1e-8
            if name not in production_corruption_residuals
            else residual <= 1e-12 and production_corruption_residuals[name] > 1e-8
        )
        for name, residual in mutant_residuals.items()
    }
    mutant_kill_rate = sum(killed.values()) / len(killed)
    if mutant_kill_rate != config["implementationAudit"]["requiredMutantKillRate"]:
        errors.append("not every registered V62r1 mutant was killed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v62r1-implementation-lock.json",
            "configs/v62r1-outcome-lock.json",
            "outputs/v62r1-terminal-residual-repair/rescore-attempt.json",
            "outputs/v62r1-terminal-residual-repair/rescore/result.json",
            "docs/v62r1-results.md",
        )
    )
    if not downstream_absent:
        errors.append("V62r1 rescore artifacts exist before implementation lock")

    result = {
        "schema_version": "62r1",
        "experiment": "v62r1_implementation_audit",
        "passed": not errors,
        "decision": (
            "freeze_v62r1_implementation" if not errors else "repair_v62r1_implementation"
        ),
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_sha256": file_sha256(implementation_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "checks": {
            "immutable_source_bindings": source_bindings_ok,
            "independent_scalar_checker": independent_checker,
            "analytic_fixtures": fixture_pass,
            "all_registered_mutants_killed": mutant_kill_rate == 1.0,
            "downstream_absence": downstream_absent,
        },
        "analytic_fixture_pass_rate": 1.0 if fixture_pass else 0.0,
        "test_process_output": (test_run.stdout + test_run.stderr).strip(),
        "mutant_residuals": mutant_residuals,
        "production_corruption_residuals": production_corruption_residuals,
        "mutants_killed": killed,
        "mutant_kill_rate": mutant_kill_rate,
        "data_access": {
            "external_source_models": 0,
            "repair_rescores": 0,
            "new_candidate_evaluations": 0,
            "new_external_rollouts": 0,
            "human_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
