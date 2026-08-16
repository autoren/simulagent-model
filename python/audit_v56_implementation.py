#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import itertools
import json
import tempfile
from pathlib import Path

from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import atom_universe, effect, entities, relation, unary
from v46_stochastic import _rule, canonical_program, delayed, stochastic
from v53_smc2 import instantiate_program, mechanic_registry, parameterize_program
from v55r1_planning import planning_registry
from v56_verification import (
    compile_policy_dtmc,
    formal_transition_support,
    independent_transition_support,
    prove_support_equivalence,
    run_storm_properties,
    tool_versions,
    transition_rows_normalize,
    validate_world_queue_action,
    write_explicit_model,
)


def _worlds(entity_rows):
    atoms = atom_universe(entity_rows)
    for values in itertools.product((False, True), repeat=len(atoms)):
        yield dict(zip(atoms, values, strict=True))


def _actions():
    bindings = (
        {"actor": "unit_0", "target": "unit_1"},
        {"actor": "unit_1", "target": "unit_0"},
    )
    return [
        {"id": action, "binding": dict(binding)}
        for action in ("pulse", "route") for binding in bindings
    ] + [{"id": "wait", "binding": {}}]


def _queue_fixtures(tick):
    event = {
        "effect": effect("toggle", unary("ready", "target")),
        "binding": {"actor": "unit_0", "target": "unit_1"},
    }
    return [
        [],
        [{**event, "due": tick}],
        [{**event, "due": tick + 1}],
    ]


def _tiny_model(probability=0.25, success_reward=1.0, action_cost=0.0):
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


def _delayed_fixture():
    return {
        "states": [
            {"id": 0, "kind": "root"},
            {"id": 1, "kind": "execution"},
            {"id": 2, "kind": "execution"},
            {"id": 3, "kind": "terminal", "success": True},
            {"id": 4, "kind": "done"},
        ],
        "transitions": [
            {"source": 0, "target": 1, "probability": 1.0, "reward": 0.0, "annotations": []},
            {"source": 1, "target": 2, "probability": 1.0, "reward": 0.0, "annotations": []},
            {"source": 2, "target": 3, "probability": 1.0, "reward": 0.0, "annotations": []},
            {"source": 3, "target": 4, "probability": 1.0, "reward": 1.0, "annotations": []},
            {"source": 4, "target": 4, "probability": 1.0, "reward": 0.0, "annotations": []},
        ],
        "root_state": 0,
        "done_state": 4,
    }


def _contingent_fixture():
    return {
        "states": [
            {"id": 0, "kind": "root"},
            {"id": 1, "kind": "execution"},
            {"id": 2, "kind": "execution"},
            {"id": 3, "kind": "terminal", "success": True},
            {"id": 4, "kind": "terminal", "success": False},
            {"id": 5, "kind": "done"},
        ],
        "transitions": [
            {"source": 0, "target": 1, "probability": 0.5, "reward": 0.0, "annotations": []},
            {"source": 0, "target": 2, "probability": 0.5, "reward": 0.0, "annotations": []},
            {"source": 1, "target": 3, "probability": 1.0, "reward": 0.0, "annotations": []},
            {"source": 2, "target": 3, "probability": 0.5, "reward": 0.0, "annotations": []},
            {"source": 2, "target": 4, "probability": 0.5, "reward": 0.0, "annotations": []},
            {"source": 3, "target": 5, "probability": 1.0, "reward": 1.0, "annotations": []},
            {"source": 4, "target": 5, "probability": 1.0, "reward": 0.0, "annotations": []},
            {"source": 5, "target": 5, "probability": 1.0, "reward": 0.0, "annotations": []},
        ],
        "root_state": 0,
        "done_state": 5,
    }


def _storm_fixture_results():
    fixtures = {
        "deterministic_success": (_tiny_model(1.0), (1.0, 1.0, 1.0)),
        "bernoulli_one_quarter_success": (_tiny_model(0.25), (1.0, 0.25, 0.25)),
        "two_tick_delayed_success": (_delayed_fixture(), (1.0, 1.0, 1.0)),
        "observation_contingent_second_action": (_contingent_fixture(), (1.0, 0.75, 0.75)),
        "negative_action_cost_plus_terminal_reward": (_tiny_model(0.75, 1.0, 0.02), (1.0, 0.75, 0.73)),
    }
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for name, (model, expected) in fixtures.items():
            directory = root / name
            write_explicit_model(model, directory)
            observed = run_storm_properties(directory)
            vector = (
                observed["termination_probability"],
                observed["success_probability"],
                observed["expected_return"],
            )
            error = max(abs(left - right) for left, right in zip(vector, expected, strict=True))
            rows.append({
                "fixture": name,
                "observed": observed,
                "expected": {
                    "termination_probability": expected[0],
                    "success_probability": expected[1],
                    "expected_return": expected[2],
                },
                "maximum_error": error,
                "passed": error <= 1e-12,
            })
    return rows


def _symbolic_mutation_controls(entity_rows):
    blank = {atom: False for atom in atom_universe(entity_rows)}
    binding = {"actor": "unit_0", "target": "unit_1"}
    action = {"id": "pulse", "binding": binding}
    immediate = canonical_program({"rules": [
        _rule("pulse", stochastic_immediate=[stochastic(
            "1/2", effect("toggle", unary("active", "target"))
        )]),
        _rule("route", deterministic_immediate=[effect(
            "toggle", relation("actor", "target")
        )]),
    ]})
    delayed_program = canonical_program({"rules": [
        _rule("pulse", stochastic_delayed=[delayed(
            2, stochastic("1/2", effect("toggle", unary("active", "target")))
        )]),
        _rule("route", deterministic_immediate=[effect(
            "toggle", unary("marked", "actor")
        )]),
    ]})
    condition_program = canonical_program({"rules": [
        _rule(
            "pulse",
            deterministic_immediate=[effect("toggle", relation("actor", "target"))],
            stochastic_immediate=[stochastic(
                "1/2", effect("toggle", unary("ready", "target")),
                relation("actor", "target"),
            )],
        ),
        _rule("route", deterministic_immediate=[effect(
            "toggle", unary("marked", "actor")
        )]),
    ]})
    due_queue = [{
        "due": 2,
        "effect": effect("toggle", unary("ready", "target")),
        "binding": binding,
    }]

    rows = []
    cases = [
        ("swap_actor_and_target", immediate, blank, [], action, 0),
        ("omit_due_queue_delivery", immediate, blank, due_queue, {"id": "wait", "binding": {}}, 2),
        ("evaluate_condition_after_deterministic_effect", condition_program, blank, [], action, 0),
        ("shift_delayed_due_tick_by_one", delayed_program, blank, [], action, 0),
    ]
    for mutation, program, source, queue, selected, tick in cases:
        reference = formal_transition_support(
            program, entity_rows, source, queue, selected, tick
        )
        mutated = independent_transition_support(
            program, entity_rows, source, queue, selected, tick, mutation
        )
        proof = prove_support_equivalence(mutated, reference)
        rows.append({
            "mutant": mutation,
            "killed": not proof["equivalent"],
            "proof_status": proof["status"],
        })
    self_action = {
        "id": "pulse",
        "binding": {"actor": "unit_0", "target": "unit_0"},
    }
    independent_accepted = True
    formal_rejected = False
    try:
        independent_transition_support(
            immediate, entity_rows, blank, [], self_action, 0,
            "permit_self_binding",
        )
    except Exception:
        independent_accepted = False
    try:
        formal_transition_support(
            immediate, entity_rows, blank, [], self_action, 0
        )
    except ValueError:
        formal_rejected = True
    rows.append({
        "mutant": "permit_self_binding",
        "killed": independent_accepted and formal_rejected,
        "proof_status": "schema_rejection",
    })
    return rows


def _missing_observation_mutant_killed(entity_rows):
    blank = {atom: False for atom in atom_universe(entity_rows)}
    program = canonical_program({"rules": [
        _rule("pulse", stochastic_immediate=[stochastic(
            "1/2", effect("toggle", unary("active", "target"))
        )]),
        _rule("route", deterministic_immediate=[effect(
            "toggle", unary("marked", "actor")
        )]),
    ]})
    registry = [{"template": parameterize_program(program)}]
    action = {"id": "pulse", "binding": {"actor": "unit_0", "target": "unit_1"}}
    observation = __import__("v42_stateful").world_signature(blank)
    terminal = {"terminal": True, "value": 0.0}
    incomplete = {
        "terminal": False,
        "value": 0.0,
        "selected_action": action,
        "selected_action_key": "fixture",
        "branches": {observation: terminal},
        "observation_probabilities": {observation: 0.5},
        "action_values": {"fixture": 0.0},
        "optimal_action_keys": ["fixture"],
    }
    atoms = [{
        "program_index": 0, "node_index": 0, "theta": 0.5,
        "configuration_key": "fixture", "world": blank, "queue": [],
        "weight": 1.0,
    }]
    config = {"formalExecutor": {"actionCosts": {
        "pulse": 0.01, "route": 0.01, "wait": 0.0,
    }}}
    try:
        compile_policy_dtmc(
            atoms, incomplete, registry, entity_rows,
            {"atom": "u:active:unit_0", "value": True}, 3, 0, config,
        )
    except RuntimeError as error:
        return "omits a reachable observation" in str(error)
    return False


def _probabilistic_mutation_controls():
    rows = []
    dropped = _tiny_model(0.25)
    dropped["transitions"] = [
        row for row in dropped["transitions"]
        if not (row["source"] == 0 and row["target"] == 1)
    ]
    rows.append({
        "mutant": "drop_one_stochastic_successor",
        "killed": not transition_rows_normalize(dropped),
    })
    rows.append({
        "mutant": "omit_one_reachable_policy_observation_branch",
        "killed": _missing_observation_mutant_killed(entities(2)),
    })
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        flipped = _tiny_model(0.25)
        flipped["states"][1]["success"] = False
        write_explicit_model(flipped, root / "flipped")
        flipped_value = run_storm_properties(root / "flipped")["success_probability"]
        rows.append({
            "mutant": "flip_terminal_success_label",
            "killed": abs(flipped_value - 0.25) > 1e-12,
        })
        no_cost = _tiny_model(0.25, 1.0, 0.0)
        write_explicit_model(no_cost, root / "no_cost")
        no_cost_value = run_storm_properties(root / "no_cost")["expected_return"]
        rows.append({
            "mutant": "omit_action_cost_reward",
            "killed": abs(no_cost_value - 0.24) > 1e-12,
        })
    corrupted = _tiny_model(0.25)
    corrupted["transitions"][0]["probability"] = 0.20
    rows.append({
        "mutant": "corrupt_initial_distribution_mass",
        "killed": not transition_rows_normalize(corrupted),
    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v56-design-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v56-symbolic-probabilistic-policy-verification/implementation-audit.json",
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors = []

    authorization_ok = (
        design["authorization"]["install_pinned_verification_dependencies"]
        and design["authorization"]["write_and_audit_independent_verifiers"]
        and not design["authorization"]["construct_v56_verification_bundle"]
        and not design["authorization"]["run_v56_candidate_formal_verification"]
        and file_sha256(PROJECT_ROOT / design["config"]) == design["config_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_outcome_lock"])
        == design["source_outcome_lock_sha256"]
    )
    if not authorization_ok:
        errors.append("V56 design lock is not intact or does not authorize implementation")

    versions = tool_versions()
    version_ok = versions == {
        "storm": config["probabilisticVerification"]["version"],
        "z3": config["symbolicVerification"]["solverVersion"],
    }
    if not version_ok:
        errors.append(f"pinned verifier version mismatch: {versions}")

    implementation_source = inspect.getsource(independent_transition_support)
    banned = (
        "step_belief", "plan_exact", "evaluate_policy",
        "continuous_unit_transition", "formal_transition_support",
    )
    independence_ok = not any(token in implementation_source for token in banned)
    if not independence_ok:
        errors.append("independent SMT transition encoder calls a forbidden implementation")

    entity_rows = entities(2)
    v55r1_config = json.loads((
        PROJECT_ROOT / "configs/v55r1-delayed-consequence-adequacy-confirmation.json"
    ).read_text())
    template_rows = [
        *mechanic_registry(5303), *planning_registry(v55r1_config)
    ]
    unique = {canonical_json(row["template"]): row for row in template_rows}
    programs = [
        instantiate_program(unique[key]["template"], 0.37)
        for key in sorted(unique)
    ]
    exhaustive_cases = proof_passes = invariant_passes = unknown_count = 0
    first_counterexample = None
    tick = 2
    for program in programs:
        for source_world in _worlds(entity_rows):
            for action in _actions():
                for queue in _queue_fixtures(tick):
                    exhaustive_cases += 1
                    invariant_passes += int(validate_world_queue_action(
                        entity_rows, source_world, queue, action, tick
                    ))
                    independent = independent_transition_support(
                        program, entity_rows, source_world, queue, action, tick
                    )
                    formal = formal_transition_support(
                        program, entity_rows, source_world, queue, action, tick
                    )
                    proof = prove_support_equivalence(independent, formal)
                    proof_passes += int(proof["equivalent"])
                    unknown_count += int(proof["status"] == "unknown")
                    if not proof["equivalent"] and first_counterexample is None:
                        first_counterexample = proof
    expected_cases = (
        len(programs)
        * config["symbolicVerification"]["exhaustiveSyntheticDomain"]["worldsPerTemplate"]
        * config["symbolicVerification"]["exhaustiveSyntheticDomain"]["actionsPerWorld"]
        * len(config["symbolicVerification"]["exhaustiveSyntheticDomain"]["queueFixtures"])
    )
    exhaustive_ok = (
        len(programs) == len(unique) == 16
        and exhaustive_cases == expected_cases == 61440
        and proof_passes == exhaustive_cases
        and invariant_passes == exhaustive_cases
        and unknown_count == 0
    )
    if not exhaustive_ok:
        errors.append("exhaustive synthetic symbolic audit failed")

    analytic_rows = _storm_fixture_results()
    analytic_passes = sum(row["passed"] for row in analytic_rows)
    analytic_ok = analytic_passes == len(analytic_rows) == 5
    if not analytic_ok:
        errors.append("Storm analytic fixture suite failed")

    symbolic_mutants = _symbolic_mutation_controls(entity_rows)
    probabilistic_mutants = _probabilistic_mutation_controls()
    mutant_rows = [*symbolic_mutants, *probabilistic_mutants]
    mutant_kills = sum(row["killed"] for row in mutant_rows)
    mutants_ok = mutant_kills == len(mutant_rows) == 10
    if not mutants_ok:
        errors.append("V56 mutation suite did not kill all ten registered mutants")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v56-implementation-lock.json",
            "configs/v56-verification-bundle-seal.json",
            "configs/v56-evaluation-implementation-lock.json",
            "configs/v56-outcome-lock.json",
            "data/v56-symbolic-probabilistic-policy-verification",
            "outputs/v56-symbolic-probabilistic-policy-verification/evaluation-attempt.json",
            "outputs/v56-symbolic-probabilistic-policy-verification/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V56 candidate or downstream artifact exists during implementation audit")

    checks = {
        "design_lock_authorization_and_binding": authorization_ok,
        "pinned_tool_versions": version_ok,
        "independent_symbolic_encoder_static_boundary": independence_ok,
        "exhaustive_synthetic_transition_support": exhaustive_ok,
        "analytic_storm_fixtures": analytic_ok,
        "all_registered_mutants_killed": mutants_ok,
        "candidate_bundle_and_results_absent": downstream_absent,
    }
    audit = {
        "schema_version": 56,
        "experiment": "v56_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v56_implementation_lock" if not errors
            else "repair_v56_verifier_implementation"
        ),
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation": "python/v56_verification.py",
        "implementation_sha256": file_sha256(PROJECT_ROOT / "python/v56_verification.py"),
        "unit_tests": "python/test_v56_verification.py",
        "unit_tests_sha256": file_sha256(PROJECT_ROOT / "python/test_v56_verification.py"),
        "tool_versions": versions,
        "checks": checks,
        "exhaustive_symbolic_audit": {
            "unique_templates": len(programs),
            "cases": exhaustive_cases,
            "expected_cases": expected_cases,
            "invariant_passes": invariant_passes,
            "support_proof_passes": proof_passes,
            "z3_unknown_count": unknown_count,
            "first_counterexample": first_counterexample,
        },
        "analytic_storm_fixtures": analytic_rows,
        "mutation_controls": {
            "symbolic": symbolic_mutants,
            "probabilistic": probabilistic_mutants,
            "kills": mutant_kills,
            "registered": len(mutant_rows),
            "kill_rate": mutant_kills / len(mutant_rows),
        },
        "data_access": {
            "v55_candidate_policy_records_accessed": 0,
            "v55r1_candidate_policy_records_accessed": 0,
            "candidate_formal_verification_runs": 0,
            "analytic_storm_runs": 5,
            "mutation_storm_runs": 2,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
