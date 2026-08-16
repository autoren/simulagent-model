#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
from collections import Counter

from generate_v55_planning import (
    build_record,
    goal_values,
    history_class_for_record,
    observation_design_key,
    prior_observation_design_keys,
    target_assignments,
)
from v10_protocol import file_sha256
from v22_relational import canonical_json, unary_atom
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, entities
from v53_smc2 import exact_inference, mechanic_registry, quadrature_rule
from v54_eig import belief_atoms_from_exact, prior_belief_atoms
from v55_planning import (
    assert_planning_payload_is_public,
    attempted_future_outcome_leak,
    best_open_loop,
    candidate_actions,
    clairvoyant_value,
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


IMPLEMENTATION_FILES = (
    "python/v55_planning.py",
    "python/generate_v55_planning.py",
    "python/test_v55_planning.py",
    "python/audit_v55_implementation.py",
)

BASE_DEPENDENCIES = (
    "python/v46_stochastic.py",
    "python/v49_belief.py",
    "python/v53_smc2.py",
    "python/v54_eig.py",
    "configs/v53r2-design-lock.json",
    "configs/v53r2-outcome-lock.json",
    "configs/v54-outcome-lock.json",
)


def flatten_predictive(branches):
    return [
        {**atom, "weight": branch["probability"] * atom["weight"]}
        for branch in branches.values() for atom in branch["atoms"]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v55-design-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/implementation-audit.json",
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    errors = []

    design_bound = (
        design["authorization"]["write_and_audit_exact_bayes_adaptive_planner"]
        and not design["authorization"]["construct_v55_planning_population"]
        and not design["authorization"]["run_v55_planning_evaluation"]
        and file_sha256(PROJECT_ROOT / design["config"]) == design["config_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
    )
    if not design_bound:
        errors.append("V55 design lock is not intact or does not authorize implementation")

    entity_rows = entities(2)
    candidates_ok = (
        len(candidate_actions(entity_rows)) == config["planningModel"]["candidateCount"]
        and len({row["key"] for row in candidate_actions(entity_rows)}) == 5
    )
    if not candidates_ok:
        errors.append("V55 candidate action set is incomplete")

    assignments = target_assignments(config)
    values = goal_values(config)
    allocation_ok = (
        Counter(assignments) == Counter({index: 4 for index in range(8)})
        and Counter(values) == Counter({False: 16, True: 16})
        and Counter(history_class_for_record(index) for index in range(32))
        == Counter(config["population"]["historyClasses"])
    )
    if not allocation_ok:
        errors.append("V55 truth, goal, or history-class allocation is imbalanced")

    fixture_config = copy.deepcopy(config)
    for key, value in tuple(fixture_config["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            fixture_config["population"][key] = value + 5_000_000
    registry = mechanic_registry(5303)
    used, prior = set(), prior_observation_design_keys()
    generated = [
        build_record(index, index, bool(index), registry, fixture_config, used, prior)
        for index in range(2)
    ]
    generator_ok = (
        [row["history_class"] for row in generated]
        == ["prior_like_all_wait", "mixed_informative"]
        and all(
            action["id"] == "wait"
            for episode in [
                *generated[0]["public"]["supports"],
                generated[0]["public"]["query"],
            ]
            for action in episode["actions"]
        )
        and [
            action["id"] for action in generated[1]["public"]["query"]["actions"][:2]
        ] == ["pulse", "route"]
        and all(
            episode["observation_design_key"] == observation_design_key(episode)
            for row in generated
            for episode in [*row["public"]["supports"], row["public"]["query"]]
        )
        and all(row["public"]["goal"]["atom"] in {
            item["atom"] for item in row["public"]["query"]["initial_state"]
        } for row in generated)
    )
    try:
        for row in generated:
            assert_planning_payload_is_public(row["public"])
    except PermissionError:
        generator_ok = False
    if not generator_ok:
        errors.append("V55 altered-seed population generator fixture failed")

    exact_config = copy.deepcopy(v53)
    exact_config["exactBenchmark"]["quadratureNodes"] = 5
    record = generated[1]
    exact = exact_inference(
        registry,
        {
            "supports": record["public"]["supports"],
            "query": record["public"]["query"],
        },
        exact_config,
    )
    atoms = belief_atoms_from_exact(exact)
    goal = record["public"]["goal"]
    query = record["public"]["query"]
    horizon = config["planningModel"]["horizonActions"]
    primary = plan_exact(
        atoms, registry, entity_rows, goal, horizon,
        query["prefix_length"], config,
    )
    reference = scalar_plan(
        atoms, registry, entity_rows, goal, horizon,
        query["prefix_length"], config,
    )
    evaluated = evaluate_policy(
        atoms, primary, registry, entity_rows, goal, horizon,
        query["prefix_length"], config,
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
        errors.append("V55 altered-seed horizon-three exact planner fixture failed")

    open_loop = best_open_loop(
        atoms, registry, entity_rows, goal, horizon,
        query["prefix_length"], config,
    )
    baselines = {
        "open_loop": open_loop["value"],
        "greedy": greedy_policy_value(
            atoms, registry, entity_rows, goal, horizon,
            query["prefix_length"], config,
        ),
        "map_program": map_program_policy_value(
            atoms, registry, entity_rows, goal, horizon,
            query["prefix_length"], config,
        ),
        "posterior_mean_theta": posterior_mean_theta_policy_value(
            atoms, registry, entity_rows, goal, horizon,
            query["prefix_length"], config,
        ),
        "eig_only": eig_policy_value(
            atoms, registry, entity_rows, goal, horizon,
            query["prefix_length"], config,
        ),
        "belief_update_disabled": evaluate_static_update_disabled_policy(
            atoms, atoms, registry, entity_rows, goal, horizon,
            query["prefix_length"], config,
        ),
    }
    clairvoyant = clairvoyant_value(
        atoms, registry, entity_rows, goal, horizon,
        query["prefix_length"], config,
    )
    controls_ok = (
        all(primary["value"] + 1e-10 >= value for value in baselines.values())
        and clairvoyant + 1e-10 >= primary["value"]
        and all(math.isfinite(value) for value in [*baselines.values(), clairvoyant])
    )
    if not controls_ok:
        errors.append("V55 baseline or clairvoyant implementation violates exact bounds")

    world = deterministic_world(entity_rows, "v55-uninformative-fixture")
    quadrature = quadrature_rule(5, v53["parameterModel"])
    uninformative_atoms = prior_belief_atoms(registry[:1], quadrature, world)
    unaffected = unary_atom("ready", entity_rows[0]["id"])
    uninformative_goal = {"atom": unaffected, "value": not world[unaffected]}
    uninformative_plan = plan_exact(
        uninformative_atoms, registry[:1], entity_rows, uninformative_goal,
        3, 0, config,
    )
    uninformative_open = best_open_loop(
        uninformative_atoms, registry[:1], entity_rows, uninformative_goal,
        3, 0, config,
    )
    uninformative_ok = (
        abs(uninformative_plan["value"] - uninformative_open["value"]) <= 1e-12
        and uninformative_plan["selected_action"]["id"] == "wait"
    )
    if not uninformative_ok:
        errors.append("V55 uninformative observation fixture did not reduce to open loop")

    delay_atom = dict(uninformative_atoms[0])
    delay_atom.update({
        "program_index": 0,
        "node_index": 0,
        "theta": 0.8,
        "weight": 1.0,
    })
    target = entity_rows[1]["id"]
    delayed_goal_atom = unary_atom("active", target)
    delay_atom["world"][delayed_goal_atom] = True
    delay_atom["configuration_key"] = canonical_json({
        "world": sorted(delay_atom["world"].items()), "queue": [],
    })
    pulse = {
        "id": "pulse",
        "binding": {"actor": entity_rows[0]["id"], "target": target},
    }
    wait = {"id": "wait", "binding": {}}
    delay_belief = [delay_atom]
    for tick, action in enumerate((pulse, wait)):
        delay_belief = flatten_predictive(step_belief(
            delay_belief, [registry[5]], entity_rows, action, tick
        ))
    before = terminal_value(
        delay_belief, {"atom": delayed_goal_atom, "value": False}
    )
    delay_belief = flatten_predictive(step_belief(
        delay_belief, [registry[5]], entity_rows, wait, 2
    ))
    after = terminal_value(
        delay_belief, {"atom": delayed_goal_atom, "value": False}
    )
    delayed_ok = abs(before) <= 1e-12 and abs(after - 0.8) <= 1e-12
    if not delayed_ok:
        errors.append("V55 two-tick delayed-effect fixture failed")

    normalization_ok = True
    for candidate in candidate_actions(entity_rows):
        branches = step_belief(
            atoms, registry, entity_rows, candidate["action"],
            query["prefix_length"],
        )
        normalization_ok &= abs(sum(
            branch["probability"] for branch in branches.values()
        ) - 1.0) <= 1e-12
        normalization_ok &= all(abs(sum(
            atom["weight"] for atom in branch["atoms"]
        ) - 1.0) <= 1e-12 for branch in branches.values())
    if not normalization_ok:
        errors.append("V55 predictive or posterior branches do not normalize")

    leakage_rejected = False
    payload_rejected = False
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
        errors.append("V55 truth or future-observation firewall is incomplete")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v55-implementation-lock.json",
            "configs/v55-evaluation-implementation-lock.json",
            "configs/v55-population-seal.json",
            "configs/v55-outcome-lock.json",
            "data/v55-short-horizon-bayes-adaptive-planning",
            "outputs/v55-short-horizon-bayes-adaptive-planning/evaluation-attempt.json",
            "outputs/v55-short-horizon-bayes-adaptive-planning/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V55 downstream population or evaluation artifact exists")

    audit = {
        "schema_version": 55,
        "experiment": "v55_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v55_implementation_lock" if not errors
            else "repair_v55_implementation"
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
            "complete_candidate_actions": candidates_ok,
            "truth_goal_and_history_allocation": allocation_ok,
            "altered_seed_generator_fixture": generator_ok,
            "horizon_three_primary_scalar_and_policy_agreement": exact_fixture_ok,
            "registered_baseline_and_clairvoyant_bounds": controls_ok,
            "uninformative_reduces_to_open_loop": uninformative_ok,
            "two_tick_delay_visible_at_horizon_three": delayed_ok,
            "belief_and_observation_normalization": normalization_ok,
            "truth_and_future_observation_firewalls": firewall_ok,
            "downstream_absent": downstream_absent,
        },
        "fixture_metrics": {
            "root_value": primary["value"],
            "root_reference_value": reference["value"],
            "root_value_error": root_error,
            "independent_policy_evaluation_error": policy_error,
            "open_loop_value": open_loop["value"],
            "baseline_values": baselines,
            "clairvoyant_value": clairvoyant,
            "uninformative_adaptive_minus_open_loop": (
                uninformative_plan["value"] - uninformative_open["value"]
            ),
            "delayed_goal_probability_before_delivery": before,
            "delayed_goal_probability_after_delivery": after,
            "static_components": len(static_marginal(atoms)),
        },
        "data_access": {
            "v55_candidate_population_records_accessed": 0,
            "v55_population_generator_executions": 0,
            "v55_planning_evaluation_runs": 0,
            "altered_seed_generator_fixture_records": 2,
            "altered_seed_planning_fixture_records": 1,
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
