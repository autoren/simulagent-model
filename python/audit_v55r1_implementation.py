#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
from collections import Counter

from generate_v55r1_planning import (
    build_record,
    goal_assignments,
    history_class_for_record,
    prior_observation_design_keys,
    target_assignments,
)
from v10_protocol import file_sha256
from v22_relational import canonical_json, unary_atom
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, entities
from v53_smc2 import exact_inference
from v54_eig import belief_atoms_from_exact
from v55_planning import (
    assert_planning_payload_is_public,
    attempted_future_outcome_leak,
    candidate_actions,
    evaluate_policy,
    plan_exact,
    scalar_plan,
    step_belief,
)
from v55r1_planning import (
    delay_suppressed_registry,
    planning_registry,
    registry_audit,
)


IMPLEMENTATION_FILES = (
    "python/v55r1_planning.py",
    "python/generate_v55r1_planning.py",
    "python/test_v55r1_planning.py",
    "python/audit_v55r1_implementation.py",
)

BASE_DEPENDENCIES = (
    "python/v46_stochastic.py",
    "python/v49_belief.py",
    "python/v53_smc2.py",
    "python/v54_eig.py",
    "python/v55_planning.py",
    "configs/v53r2-design-lock.json",
    "configs/v53r2-outcome-lock.json",
    "configs/v55-implementation-lock.json",
    "configs/v55-outcome-lock.json",
)


def known_atom(entity_rows, program_index: int, goal: dict) -> dict:
    world = deterministic_world(entity_rows, f"v55r1-audit-{program_index}")
    world[goal["atom"]] = not goal["value"]
    return {
        "program_index": program_index,
        "node_index": 0,
        "theta": 0.8,
        "configuration_key": canonical_json({
            "world": sorted(world.items()), "queue": [],
        }),
        "world": world,
        "queue": [],
        "weight": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v55r1-design-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/implementation-audit.json",
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    errors: list[str] = []

    design_bound = (
        design["authorization"]["write_and_audit_v55r1_implementation"]
        and not design["authorization"]["construct_v55r1_population"]
        and not design["authorization"]["run_v55r1_evaluation"]
        and file_sha256(PROJECT_ROOT / design["config"]) == design["config_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_outcome_lock"])
        == design["source_outcome_lock_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_localization"])
        == design["source_localization_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_implementation_lock"])
        == design["source_implementation_lock_sha256"]
    )
    if not design_bound:
        errors.append("V55r1 design lock is not intact or does not authorize implementation")

    registry = planning_registry(config)
    registry_metrics = registry_audit(registry)
    registry_ok = (
        registry_metrics["templates"] == 8
        and registry_metrics["unique_template_keys"] == 8
        and registry_metrics["delay_class_counts"]
        == config["planningSpecificRegistry"]["delayClassCounts"]
        and registry_metrics["theta_branches"] == 8
        and registry_metrics["active_stochastic_targets"] == 8
        and registry_metrics["active_deterministic_targets"] == 0
        and all(row["blueprint"] == blueprint for row, blueprint in zip(
            registry,
            config["planningSpecificRegistry"]["templateBlueprints"],
            strict=True,
        ))
    )
    if not registry_ok:
        errors.append("V55r1 registry does not implement the frozen blueprints")

    entity_rows = entities(2)
    actions = candidate_actions(entity_rows)
    candidates_ok = len(actions) == 5 and len({row["key"] for row in actions}) == 5
    if not candidates_ok:
        errors.append("V55r1 changed or pruned the complete action set")

    assignments = target_assignments(config)
    goals = goal_assignments(config)
    allocation_ok = (
        Counter(assignments) == Counter({index: 2 for index in range(8)})
        and Counter((goal["atom"], goal["value"]) for goal in goals)
        == Counter({
            (unary_atom("active", "unit_0"), False): 4,
            (unary_atom("active", "unit_0"), True): 4,
            (unary_atom("active", "unit_1"), False): 4,
            (unary_atom("active", "unit_1"), True): 4,
        })
        and Counter(history_class_for_record(index) for index in range(16))
        == Counter(config["population"]["historyClasses"])
    )
    if not allocation_ok:
        errors.append("V55r1 truth, goal, or history allocation is imbalanced")

    fixture_config = copy.deepcopy(config)
    for key, value in tuple(fixture_config["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            fixture_config["population"][key] = value + 5_000_000
    fixture_registry = planning_registry(fixture_config)
    used, prior = set(), prior_observation_design_keys()
    generated = [
        build_record(
            index,
            index,
            {"atom": unary_atom("active", f"unit_{index}"), "value": bool(index)},
            fixture_registry,
            fixture_config,
            used,
            prior,
        )
        for index in range(2)
    ]
    generator_ok = True
    for row in generated:
        query, goal = row["public"]["query"], row["public"]["goal"]
        initial = {
            item["atom"]: item["allowed_values"][0]
            for item in query["initial_state"]
        }
        generator_ok &= all(
            initial[unary_atom("active", entity["id"])] is not goal["value"]
            for entity in query["entities"]
        )
        generator_ok &= all(
            action["id"] == "wait"
            for episode in [*row["public"]["supports"], query]
            for action in episode["actions"]
        ) if row["history_class"] == "prior_like_all_wait" else (
            [action["id"] for action in query["actions"][:2]]
            == ["pulse", "route"]
        )
        try:
            assert_planning_payload_is_public(row["public"])
        except PermissionError:
            generator_ok = False
    if not generator_ok:
        errors.append("V55r1 altered-seed generator fixture failed")

    suppressed = delay_suppressed_registry(registry, 3)
    decision_differences = []
    for program_index, row in enumerate(registry[:4]):
        target_id = (
            entity_rows[1]["id"]
            if row["blueprint"]["targetVar"] == "target"
            else entity_rows[0]["id"]
        )
        goal = {
            "atom": unary_atom("active", target_id),
            "value": row["blueprint"]["operation"] == "set_true",
        }
        atom = known_atom(entity_rows, program_index, goal)
        primary = plan_exact(
            [atom], registry, entity_rows, goal, 3, 0, config
        )
        counterfactual = plan_exact(
            [atom], suppressed, entity_rows, goal, 3, 0, config
        )
        difference = primary["value"] - counterfactual["value"]
        decision_differences.append(difference)
    decision_relevance_ok = all(value > 0.001 for value in decision_differences)
    if not decision_relevance_ok:
        errors.append("A delay-two blueprint lacks an exhaustive decision-relevance fixture")

    exact_config = copy.deepcopy(v53)
    exact_config["exactBenchmark"]["quadratureNodes"] = 5
    record = generated[1]
    exact = exact_inference(
        registry,
        {"supports": record["public"]["supports"], "query": record["public"]["query"]},
        exact_config,
    )
    atoms = belief_atoms_from_exact(exact)
    query, goal = record["public"]["query"], record["public"]["goal"]
    tick = query["prefix_length"]
    primary = plan_exact(atoms, registry, entity_rows, goal, 3, tick, config)
    reference = scalar_plan(atoms, registry, entity_rows, goal, 3, tick, config)
    evaluated = evaluate_policy(
        atoms, primary, registry, entity_rows, goal, 3, tick, config
    )
    root_error = abs(primary["value"] - reference["value"])
    policy_error = abs(primary["value"] - evaluated)
    exact_fixture_ok = (
        math.isfinite(primary["value"])
        and root_error <= 1e-12
        and policy_error <= 1e-12
        and primary["selected_action_key"] in reference["optimal_action_keys"]
    )
    if not exact_fixture_ok:
        errors.append("V55r1 altered-seed exact planning fixture failed")

    normalization_ok = True
    for candidate in actions:
        branches = step_belief(
            atoms, registry, entity_rows, candidate["action"], tick
        )
        normalization_ok &= abs(sum(
            branch["probability"] for branch in branches.values()
        ) - 1.0) <= 1e-12
        normalization_ok &= all(abs(sum(
            atom["weight"] for atom in branch["atoms"]
        ) - 1.0) <= 1e-12 for branch in branches.values())
    if not normalization_ok:
        errors.append("V55r1 predictive or posterior branches do not normalize")

    leakage_rejected = payload_rejected = False
    try:
        attempted_future_outcome_leak({}, {"future": True})
    except PermissionError:
        leakage_rejected = True
    try:
        assert_planning_payload_is_public({"truth": {"secret": True}})
    except PermissionError:
        payload_rejected = True
    selector_signature_ok = not (
        {"truth", "future_observation", "realized_outcome"}
        & set(inspect.signature(plan_exact).parameters)
    )
    firewall_ok = leakage_rejected and payload_rejected and selector_signature_ok
    if not firewall_ok:
        errors.append("V55r1 truth or future-observation firewall is incomplete")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v55r1-implementation-lock.json",
            "configs/v55r1-evaluation-implementation-lock.json",
            "configs/v55r1-population-seal.json",
            "configs/v55r1-outcome-lock.json",
            "data/v55r1-delayed-consequence-adequacy-confirmation",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation-attempt.json",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V55r1 downstream population or evaluation artifact exists")

    audit = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v55r1_implementation_lock" if not errors
            else "repair_v55r1_implementation"
        ),
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION_FILES
        },
        "base_dependencies_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in BASE_DEPENDENCIES
        },
        "checks": {
            "design_bound_and_prepopulation": design_bound,
            "registry_matches_frozen_blueprints": registry_ok,
            "complete_candidate_actions": candidates_ok,
            "truth_independent_goal_and_history_allocation": allocation_ok,
            "altered_seed_generator_fixture": generator_ok,
            "all_delay_two_blueprints_are_decision_relevant": decision_relevance_ok,
            "horizon_three_primary_scalar_and_policy_agreement": exact_fixture_ok,
            "belief_and_observation_normalization": normalization_ok,
            "truth_and_future_observation_firewalls": firewall_ok,
            "downstream_absent": downstream_absent,
        },
        "fixture_metrics": {
            "registry": registry_metrics,
            "delay_two_primary_minus_suppressed_values": decision_differences,
            "minimum_delay_two_decision_value_change": min(decision_differences),
            "altered_seed_root_value": primary["value"],
            "altered_seed_root_reference_value": reference["value"],
            "altered_seed_root_value_error": root_error,
            "altered_seed_independent_policy_error": policy_error,
            "altered_seed_belief_atoms": len(atoms),
        },
        "data_access": {
            "v55r1_candidate_population_records_accessed": 0,
            "v55r1_population_generator_executions": 0,
            "v55r1_planning_evaluation_runs": 0,
            "additional_v55_planning_evaluation_runs": 0,
            "altered_seed_generator_fixture_records": 2,
            "altered_seed_planning_fixture_records": 1,
            "known_model_decision_relevance_fixtures": 4,
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
