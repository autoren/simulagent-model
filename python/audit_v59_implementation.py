#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
from collections import Counter

from generate_v59_planning import (
    build_record,
    goal_assignments,
    history_class_for_record,
    horizon_assignments,
    target_assignments,
)
from v10_protocol import file_sha256
from v22_relational import canonical_json, unary_atom
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, entities
from v59_planning import (
    _rollout_action,
    assert_search_payload_is_public,
    domain_transition_factory,
    evaluate_policy_pair,
    forbidden_latent_conditioned_rollout,
    run_root_sampled_uct,
    sample_root_counts,
    static_atom_label,
)
from v55_planning import candidate_actions
from v55r1_planning import planning_registry, registry_audit


IMPLEMENTATION_FILES = (
    "python/v59_planning.py",
    "python/generate_v59_planning.py",
    "python/test_v59_planning.py",
    "python/audit_v59_implementation.py",
)

BASE_DEPENDENCIES = (
    "python/v46_stochastic.py",
    "python/v49_belief.py",
    "python/v53_smc2.py",
    "python/v55_planning.py",
    "python/v55r1_planning.py",
    "configs/v53r2-design-lock.json",
    "configs/v53r2-outcome-lock.json",
    "configs/v55r1-design-lock.json",
    "configs/v55r1-outcome-lock.json",
    "configs/v56-outcome-lock.json",
)


TOY_ACTIONS = [
    {"action": {"id": name}, "key": name}
    for name in ("left", "probe", "right")
]


def toy_rows():
    return [
        {"mode": "left", "success": False, "weight": 0.5},
        {"mode": "right", "success": False, "weight": 0.5},
    ]


def toy_transition(state, action, tick, draw):
    del draw
    result = dict(state)
    if action["id"] == "probe":
        observation = state["mode"]
    else:
        if tick >= 1:
            result["success"] = action["id"] == state["mode"]
        observation = "uninformative"
    return result, observation


def nonpersistent_transition(state, action, tick, draw):
    result, observation = toy_transition(state, action, tick, draw)
    if action["id"] == "probe":
        result["mode"] = "left" if draw < 0.5 else "right"
    return result, observation


def terminal(state):
    return float(state["success"])


def action_cost(action):
    return 0.01 if action["id"] == "probe" else 0.0


def toy_label(state):
    return state["mode"]


def toy_search(seed, *, blind=False, transition=toy_transition, omit_cost=False, budget=3000, override=None):
    return run_root_sampled_uct(
        toy_rows(), TOY_ACTIONS, 2, 0, budget, seed, transition,
        terminal, action_cost, toy_label,
        merge_observations=blind,
        omit_action_cost=omit_cost,
        simulation_limit_override=override,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v59-design-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v59-budgeted-root-sampled-planning/implementation-audit.json",
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []

    design_bound = (
        design["authorization"]["write_and_audit_candidate_search"]
        and design["authorization"]["write_and_audit_independent_generator"]
        and not design["authorization"]["construct_v59_population"]
        and not design["authorization"]["run_v59_evaluation"]
        and not design["authorization"]["simulate_human_v58_records"]
        and file_sha256(PROJECT_ROOT / design["config"]) == design["config_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["v58_deferral"])
        == design["v58_deferral_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for path, digest in design["source_outcome_locks_sha256"].items()
        )
    )
    if not design_bound:
        errors.append("V59 design lock is not intact or does not authorize implementation")

    v55r1_config = json.loads(
        (PROJECT_ROOT / "configs/v55r1-design-lock.json").read_text()
    )["config_payload"]
    registry = planning_registry(v55r1_config)
    registry_metrics = registry_audit(registry)
    registry_ok = (
        registry_metrics["templates"] == 8
        and registry_metrics["unique_template_keys"] == 8
        and registry_metrics["theta_branches"] == 8
        and registry_metrics["active_stochastic_targets"] == 8
        and registry_metrics["active_deterministic_targets"] == 0
    )
    if not registry_ok:
        errors.append("V59 does not retain the frozen V55r1 planning registry")

    entity_rows = entities(2)
    actions = candidate_actions(entity_rows)
    complete_actions = len(actions) == 5 and len({row["key"] for row in actions}) == 5
    if not complete_actions:
        errors.append("V59 action set is incomplete or pruned")

    assignments = target_assignments(config)
    goals = goal_assignments(config)
    horizons = horizon_assignments(config)
    allocation_ok = (
        Counter(assignments) == Counter({index: 3 for index in range(8)})
        and Counter(horizons) == Counter({3: 8, 5: 8, 7: 8})
        and Counter((goal["atom"], goal["value"]) for goal in goals)
        == Counter({
            (unary_atom("active", "unit_0"), False): 6,
            (unary_atom("active", "unit_0"), True): 6,
            (unary_atom("active", "unit_1"), False): 6,
            (unary_atom("active", "unit_1"), True): 6,
        })
        and Counter(history_class_for_record(index) for index in range(24))
        == Counter(config["population"]["historyClasses"])
    )
    if not allocation_ok:
        errors.append("V59 template, goal, horizon, or history allocation is imbalanced")

    fixture_config = copy.deepcopy(config)
    for key, value in tuple(fixture_config["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            fixture_config["population"][key] = value + 5_900_000
    generated = [
        build_record(
            index, index,
            {"atom": unary_atom("active", f"unit_{index}"), "value": bool(index)},
            (3, 5)[index], registry, fixture_config, set(), set(),
        )
        for index in range(2)
    ]
    generator_ok = True
    for public_row, audit_row in generated:
        generator_ok &= "truth" not in public_row
        generator_ok &= set(audit_row) == {"id", "schema_version", "record", "truth"}
        generator_ok &= public_row["id"] == audit_row["id"]
        query, goal = public_row["public"]["query"], public_row["public"]["goal"]
        initial = {
            item["atom"]: item["allowed_values"][0]
            for item in query["initial_state"]
        }
        generator_ok &= all(
            initial[unary_atom("active", entity["id"])] is not goal["value"]
            for entity in query["entities"]
        )
        try:
            assert_search_payload_is_public(public_row)
        except PermissionError:
            generator_ok = False
    if not generator_ok:
        errors.append("V59 altered-seed public/audit split generator fixture failed")

    counts = sample_root_counts(
        [
            {"label": "a", "weight": 0.2},
            {"label": "b", "weight": 0.3},
            {"label": "c", "weight": 0.5},
        ],
        50000, config["population"]["populationAuditSeed"],
        lambda state: state["label"],
    )
    expected = {"a": 0.2, "b": 0.3, "c": 0.5}
    total_variation = 0.5 * sum(
        abs(counts[key] / 50000 - probability)
        for key, probability in expected.items()
    )
    root_sampling_ok = (
        total_variation <= config["gates"]["maximumAnalyticRootSampleStaticTotalVariation"]
    )
    if not root_sampling_ok:
        errors.append("V59 analytic root sampling distribution is inaccurate")

    candidate = toy_search(59031)
    replay = toy_search(59031)
    blind = toy_search(59031, blind=True)
    comparison = evaluate_policy_pair(
        candidate, blind, toy_rows(), TOY_ACTIONS, 2, 0, 5000, 59033,
        toy_transition, terminal, action_cost, 59037,
    )
    analytic_ok = (
        candidate.selected_action_key == "probe"
        and candidate.tree_sha256 == replay.tree_sha256
        and candidate.simulations_run == candidate.budget == 3000
        and blind.branching_action_nodes == 0
        and candidate.branching_action_nodes > 0
        and comparison["candidate_mean_return"] > 0.9
        and comparison["paired_mean_difference"] > 0.35
    )
    if not analytic_ok:
        errors.append("V59 analytic observation-contingent search fixture failed")

    mutant_results = {}
    nonpersistent = toy_search(59031, transition=nonpersistent_transition)
    nonpersistent_comparison = evaluate_policy_pair(
        candidate, nonpersistent, toy_rows(), TOY_ACTIONS, 2, 0, 5000, 59039,
        toy_transition, terminal, action_cost, 59041,
    )
    mutant_results["nonpersistent_static_latent"] = (
        nonpersistent_comparison["paired_mean_difference"] > 0.30
    )

    latent_rejected = False
    try:
        forbidden_latent_conditioned_rollout({"program_index": 0})
    except PermissionError:
        latent_rejected = True
    mutant_results["latent_conditioned_rollout"] = latent_rejected

    permuted = copy.deepcopy(candidate)
    probe_children = permuted.root.actions["probe"].children
    if set(probe_children) == {"left", "right"}:
        probe_children["left"], probe_children["right"] = (
            probe_children["right"], probe_children["left"]
        )
    permutation_comparison = evaluate_policy_pair(
        candidate, permuted, toy_rows(), TOY_ACTIONS, 2, 0, 5000, 59043,
        toy_transition, terminal, action_cost, 59047,
    )
    mutant_results["observation_permutation"] = (
        permutation_comparison["paired_mean_difference"] > 0.70
    )

    cost_mutant = toy_search(59031, omit_cost=True, budget=512)
    normal_cost = toy_search(59031, budget=512)
    normal_probe = next(
        row for row in normal_cost.root_action_rows if row["action_key"] == "probe"
    )
    mutant_probe = next(
        row for row in cost_mutant.root_action_rows if row["action_key"] == "probe"
    )
    mutant_results["action_cost_omission"] = (
        mutant_probe["mean_return"] > normal_probe["mean_return"]
    )

    budget_mutant = toy_search(59031, budget=64, override=63)
    mutant_results["budget_off_by_one"] = (
        budget_mutant.simulations_run != budget_mutant.budget
    )
    mutant_kill_rate = sum(mutant_results.values()) / len(mutant_results)
    mutants_ok = mutant_kill_rate == 1.0
    if not mutants_ok:
        errors.append("V59 implementation mutation controls were not all detected")

    world = deterministic_world(entity_rows, "v59-domain-transition-audit")
    atoms = [
        {
            "program_index": index,
            "node_index": 0,
            "theta": theta,
            "configuration_key": canonical_json({
                "world": sorted(world.items()), "queue": [],
            }),
            "world": dict(world),
            "queue": [],
            "weight": 0.5,
        }
        for index, theta in ((0, 0.3), (1, 0.7))
    ]
    transition = domain_transition_factory(registry, entity_rows)
    next_state, _ = transition(
        atoms[0], actions[0]["action"], 0, 0.25
    )
    persistence_ok = (
        next_state["program_index"] == atoms[0]["program_index"]
        and next_state["node_index"] == atoms[0]["node_index"]
        and next_state["theta"] == atoms[0]["theta"]
        and abs(sum(atom["weight"] for atom in atoms) - 1.0) <= 1e-12
    )
    if not persistence_ok:
        errors.append("V59 domain transition does not preserve static root latents")

    payload_rejected = False
    try:
        assert_search_payload_is_public({"truth": {"target_program_index": 0}})
    except PermissionError:
        payload_rejected = True
    rollout_signature_ok = set(inspect.signature(_rollout_action).parameters) == {
        "action_rows", "history", "rng"
    }
    candidate_signature_ok = not (
        {"truth", "future_observation", "realized_outcome"}
        & set(inspect.signature(run_root_sampled_uct).parameters)
    )
    firewall_ok = payload_rejected and latent_rejected and rollout_signature_ok and candidate_signature_ok
    if not firewall_ok:
        errors.append("V59 truth, future-observation, or latent-rollout firewall is incomplete")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v59-implementation-lock.json",
            "configs/v59-population-seal.json",
            "configs/v59-evaluation-implementation-lock.json",
            "configs/v59-outcome-lock.json",
            "data/v59-budgeted-root-sampled-planning",
            "outputs/v59-budgeted-root-sampled-planning/evaluation-attempt.json",
            "outputs/v59-budgeted-root-sampled-planning/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V59 downstream population or evaluation artifact exists")

    audit = {
        "schema_version": 59,
        "experiment": "v59_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v59_implementation_lock" if not errors
            else "repair_v59_implementation"
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
            "frozen_v55r1_registry": registry_ok,
            "complete_candidate_actions": complete_actions,
            "balanced_template_goal_horizon_history_allocation": allocation_ok,
            "altered_seed_public_audit_split_generator": generator_ok,
            "analytic_root_sampling_distribution": root_sampling_ok,
            "analytic_observation_contingent_search": analytic_ok,
            "all_implementation_mutants_detected": mutants_ok,
            "domain_static_latent_persistence": persistence_ok,
            "truth_future_and_latent_rollout_firewalls": firewall_ok,
            "downstream_absent": downstream_absent,
        },
        "fixture_metrics": {
            "registry": registry_metrics,
            "analytic_root_sample_counts": counts,
            "analytic_root_sample_total_variation": total_variation,
            "toy_candidate_return": comparison["candidate_mean_return"],
            "toy_observation_blind_return": comparison["control_mean_return"],
            "toy_candidate_minus_blind": comparison["paired_mean_difference"],
            "nonpersistent_candidate_minus_mutant": nonpersistent_comparison["paired_mean_difference"],
            "correct_minus_permuted_observation_policy": permutation_comparison["paired_mean_difference"],
            "mutant_results": mutant_results,
            "implementation_mutant_kill_rate": mutant_kill_rate,
        },
        "data_access": {
            "v59_candidate_population_records_accessed": 0,
            "v59_population_generator_executions": 0,
            "v59_evaluation_runs": 0,
            "altered_seed_generator_fixture_records": 2,
            "analytic_search_fixture_runs": 1,
            "human_authored_records_collected": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

