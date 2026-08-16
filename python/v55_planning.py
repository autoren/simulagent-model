"""Exact short-horizon Bayes-adaptive planning utilities for V55."""
from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

from v22_relational import canonical_json
from v42_stateful import world_signature
from v49_belief import _configuration_key_with_history
from v53_smc2 import continuous_advance_configurations, instantiate_program
from v54_eig import (
    candidate_interventions,
    expected_information_gain_from_joint,
    map_program_atoms,
    target_key,
    theta_point_mass_atoms,
)


FORBIDDEN_PLANNING_KEYS = frozenset({
    "truth", "target_program_index", "target_program_key",
    "target_program_ordinal", "target_theta", "query_configuration_key",
    "query_world", "query_queue", "future_observation",
    "realized_outcome",
})


def assert_planning_payload_is_public(value):
    if isinstance(value, dict):
        overlap = set(value) & FORBIDDEN_PLANNING_KEYS
        if overlap:
            raise PermissionError(
                "V55 planning payload contains forbidden fields: "
                + ", ".join(sorted(overlap))
            )
        for child in value.values():
            assert_planning_payload_is_public(child)
    elif isinstance(value, list):
        for child in value:
            assert_planning_payload_is_public(child)


def candidate_actions(entity_rows):
    rows = [{
        "action": candidate["action"],
        "key": canonical_json(candidate["action"]),
    } for candidate in candidate_interventions(entity_rows)]
    rows.sort(key=lambda row: row["key"])
    expected = 1 + 2 * len(entity_rows) * (len(entity_rows) - 1)
    if len(rows) != expected or len({row["key"] for row in rows}) != expected:
        raise RuntimeError("V55 action set is incomplete or duplicated")
    return rows


def action_cost(action, config):
    return float(config["planningModel"]["actionCost"][action["id"]])


def belief_key(atoms):
    return tuple(sorted(
        (
            atom["program_index"], atom["node_index"],
            float(atom["theta"]).hex(), atom["configuration_key"],
            float(atom["weight"]).hex(),
        )
        for atom in atoms if atom["weight"] > 0
    ))


def _program(cache, registry, atom):
    key = (atom["program_index"], atom["node_index"], atom["theta"])
    if key not in cache:
        cache[key] = instantiate_program(
            registry[atom["program_index"]]["template"], atom["theta"]
        )
    return cache[key]


def step_belief(atoms, registry, entity_rows, action, tick, program_cache=None):
    """Return exact p(observation) and posterior beliefs after one action."""
    program_cache = {} if program_cache is None else program_cache
    raw = defaultdict(dict)
    outcome_mass = defaultdict(float)
    for atom in atoms:
        seed = {
            _configuration_key_with_history(atom["world"], atom["queue"], []): {
                "world": dict(atom["world"]),
                "queue": list(atom["queue"]),
                "history": [],
                "mass": Fraction(1),
            }
        }
        branches = continuous_advance_configurations(
            _program(program_cache, registry, atom), entity_rows,
            seed, [action], tick,
        )
        for branch in branches.values():
            outcome = world_signature(branch["world"])
            mass = atom["weight"] * float(branch["mass"])
            outcome_mass[outcome] += mass
            configuration_key = canonical_json({
                "world": sorted(branch["world"].items()),
                "queue": sorted(branch["queue"], key=canonical_json),
            })
            key = (
                atom["program_index"], atom["node_index"],
                float(atom["theta"]).hex(), configuration_key,
            )
            if key not in raw[outcome]:
                raw[outcome][key] = {
                    "program_index": atom["program_index"],
                    "node_index": atom["node_index"],
                    "theta": atom["theta"],
                    "configuration_key": configuration_key,
                    "world": dict(branch["world"]),
                    "queue": list(branch["queue"]),
                    "weight": 0.0,
                }
            raw[outcome][key]["weight"] += mass
    total = sum(outcome_mass.values())
    if abs(total - 1.0) > 1e-12:
        raise RuntimeError("V55 predictive observations do not normalize")
    result = {}
    for outcome in sorted(raw):
        probability = outcome_mass[outcome]
        posterior = list(raw[outcome].values())
        for atom in posterior:
            atom["weight"] /= probability
        if abs(sum(atom["weight"] for atom in posterior) - 1.0) > 1e-12:
            raise RuntimeError("V55 posterior belief does not normalize")
        result[outcome] = {"probability": probability, "atoms": posterior}
    return result


def terminal_value(atoms, goal):
    return sum(
        atom["weight"]
        for atom in atoms
        if atom["world"][goal["atom"]] is goal["value"]
    )


def _select_action(action_rows, tolerance):
    maximum = max(row["value"] for row in action_rows)
    optimal = sorted(
        (row for row in action_rows if row["value"] >= maximum - tolerance),
        key=lambda row: row["action_key"],
    )
    return maximum, optimal[0], [row["action_key"] for row in optimal]


def plan_exact(
    atoms, registry, entity_rows, goal, horizon, tick, config,
    memo=None, program_cache=None,
):
    memo = {} if memo is None else memo
    program_cache = {} if program_cache is None else program_cache
    if horizon == 0:
        value = terminal_value(atoms, goal)
        return {"value": value, "terminal": True, "goal": goal}
    key = (horizon, tick, goal["atom"], goal["value"], belief_key(atoms))
    if key in memo:
        return memo[key]
    action_rows = []
    for candidate in candidate_actions(entity_rows):
        branches = step_belief(
            atoms, registry, entity_rows, candidate["action"], tick, program_cache
        )
        children = {}
        future = 0.0
        for outcome, branch in branches.items():
            child = plan_exact(
                branch["atoms"], registry, entity_rows, goal,
                horizon - 1, tick + 1, config, memo, program_cache,
            )
            children[outcome] = child
            future += branch["probability"] * child["value"]
        action_rows.append({
            "action": candidate["action"],
            "action_key": candidate["key"],
            "value": -action_cost(candidate["action"], config) + future,
            "branches": children,
            "observation_probabilities": {
                outcome: branch["probability"] for outcome, branch in branches.items()
            },
        })
    maximum, selected, optimal_keys = _select_action(
        action_rows, config["planningModel"]["tieTolerance"]
    )
    result = {
        "value": maximum,
        "terminal": False,
        "selected_action": selected["action"],
        "selected_action_key": selected["action_key"],
        "optimal_action_keys": optimal_keys,
        "action_values": {
            row["action_key"]: row["value"] for row in action_rows
        },
        "branches": selected["branches"],
        "observation_probabilities": selected["observation_probabilities"],
    }
    memo[key] = result
    return result


def scalar_plan(atoms, registry, entity_rows, goal, horizon, tick, config):
    """Explicit scalar reference with no memoization or shared program cache."""
    if horizon == 0:
        return {"value": terminal_value(atoms, goal), "terminal": True}
    rows = []
    for candidate in candidate_actions(entity_rows):
        branches = step_belief(
            atoms, registry, entity_rows, candidate["action"], tick, {}
        )
        value = -action_cost(candidate["action"], config)
        for branch in branches.values():
            value += branch["probability"] * scalar_plan(
                branch["atoms"], registry, entity_rows, goal,
                horizon - 1, tick + 1, config,
            )["value"]
        rows.append({"action_key": candidate["key"], "value": value})
    maximum, selected, optimal = _select_action(
        rows, config["planningModel"]["tieTolerance"]
    )
    return {
        "value": maximum,
        "terminal": False,
        "selected_action_key": selected["action_key"],
        "optimal_action_keys": optimal,
        "action_values": {row["action_key"]: row["value"] for row in rows},
    }


def evaluate_policy(atoms, policy, registry, entity_rows, goal, horizon, tick, config):
    if horizon == 0:
        return terminal_value(atoms, goal)
    branches = step_belief(
        atoms, registry, entity_rows, policy["selected_action"], tick, {}
    )
    value = -action_cost(policy["selected_action"], config)
    for outcome, branch in branches.items():
        if outcome not in policy["branches"]:
            raise RuntimeError("V55 frozen policy omits a reachable observation")
        value += branch["probability"] * evaluate_policy(
            branch["atoms"], policy["branches"][outcome], registry,
            entity_rows, goal, horizon - 1, tick + 1, config,
        )
    return value


def marginal_step(atoms, registry, entity_rows, action, tick):
    branches = step_belief(atoms, registry, entity_rows, action, tick, {})
    merged = {}
    for branch in branches.values():
        for atom in branch["atoms"]:
            mass = branch["probability"] * atom["weight"]
            key = (
                atom["program_index"], atom["node_index"],
                float(atom["theta"]).hex(), atom["configuration_key"],
            )
            if key not in merged:
                merged[key] = {**atom, "weight": 0.0}
            merged[key]["weight"] += mass
    result = list(merged.values())
    total = sum(atom["weight"] for atom in result)
    for atom in result:
        atom["weight"] /= total
    return result


def best_open_loop(atoms, registry, entity_rows, goal, horizon, tick, config):
    candidates = candidate_actions(entity_rows)
    rows = []

    def visit(current, depth, current_tick, actions, costs):
        if depth == 0:
            rows.append({
                "action_keys": tuple(action["key"] for action in actions),
                "actions": tuple(action["action"] for action in actions),
                "value": terminal_value(current, goal) - costs,
            })
            return
        for candidate in candidates:
            visit(
                marginal_step(
                    current, registry, entity_rows, candidate["action"], current_tick
                ),
                depth - 1, current_tick + 1, [*actions, candidate],
                costs + action_cost(candidate["action"], config),
            )

    visit(atoms, horizon, tick, [], 0.0)
    maximum = max(row["value"] for row in rows)
    tolerance = config["planningModel"]["tieTolerance"]
    optimal = sorted(
        (row for row in rows if row["value"] >= maximum - tolerance),
        key=lambda row: row["action_keys"],
    )
    return {"value": maximum, "selected": optimal[0], "sequence_count": len(rows)}


def greedy_action(atoms, registry, entity_rows, goal, tick, config):
    rows = []
    for candidate in candidate_actions(entity_rows):
        branches = step_belief(atoms, registry, entity_rows, candidate["action"], tick, {})
        one_step = -action_cost(candidate["action"], config) + sum(
            branch["probability"] * terminal_value(branch["atoms"], goal)
            for branch in branches.values()
        )
        rows.append({
            "action": candidate["action"],
            "action_key": candidate["key"],
            "value": one_step,
        })
    return _select_action(rows, config["planningModel"]["tieTolerance"])[1]


def eig_action(atoms, registry, entity_rows, tick, config):
    rows = []
    prior = defaultdict(float)
    for atom in atoms:
        prior[target_key(atom)] += atom["weight"]
    for candidate in candidate_actions(entity_rows):
        branches = step_belief(atoms, registry, entity_rows, candidate["action"], tick, {})
        joint = {}
        for outcome, branch in branches.items():
            masses = defaultdict(float)
            for atom in branch["atoms"]:
                masses[target_key(atom)] += branch["probability"] * atom["weight"]
            joint[outcome] = dict(masses)
        value = expected_information_gain_from_joint(dict(prior), joint)["eig"]
        rows.append({
            "action": candidate["action"],
            "action_key": candidate["key"],
            "value": value,
        })
    return _select_action(rows, config["planningModel"]["tieTolerance"])[1]


def evaluate_replanning_policy(
    atoms, registry, entity_rows, goal, horizon, tick, config, selector
):
    if horizon == 0:
        return terminal_value(atoms, goal)
    action = selector(atoms, horizon, tick)
    branches = step_belief(atoms, registry, entity_rows, action, tick, {})
    return -action_cost(action, config) + sum(
        branch["probability"] * evaluate_replanning_policy(
            branch["atoms"], registry, entity_rows, goal,
            horizon - 1, tick + 1, config, selector,
        )
        for branch in branches.values()
    )


def greedy_policy_value(atoms, registry, entity_rows, goal, horizon, tick, config):
    return evaluate_replanning_policy(
        atoms, registry, entity_rows, goal, horizon, tick, config,
        lambda belief, _horizon, current_tick: greedy_action(
            belief, registry, entity_rows, goal, current_tick, config
        )["action"],
    )


def eig_policy_value(atoms, registry, entity_rows, goal, horizon, tick, config):
    return evaluate_replanning_policy(
        atoms, registry, entity_rows, goal, horizon, tick, config,
        lambda belief, _horizon, current_tick: eig_action(
            belief, registry, entity_rows, current_tick, config
        )["action"],
    )


def transformed_planner_value(
    atoms, registry, entity_rows, goal, horizon, tick, config, transform
):
    def selector(belief, remaining, current_tick):
        planned = plan_exact(
            transform(belief), registry, entity_rows, goal,
            remaining, current_tick, config,
        )
        return planned["selected_action"]

    return evaluate_replanning_policy(
        atoms, registry, entity_rows, goal, horizon, tick, config, selector
    )


def map_program_policy_value(atoms, registry, entity_rows, goal, horizon, tick, config):
    return transformed_planner_value(
        atoms, registry, entity_rows, goal, horizon, tick, config,
        map_program_atoms,
    )


def posterior_mean_theta_policy_value(
    atoms, registry, entity_rows, goal, horizon, tick, config
):
    return transformed_planner_value(
        atoms, registry, entity_rows, goal, horizon, tick, config,
        theta_point_mass_atoms,
    )


def static_key(atom):
    return (
        atom["program_index"], atom["node_index"], float(atom["theta"]).hex()
    )


def static_marginal(atoms):
    result = defaultdict(float)
    for atom in atoms:
        result[static_key(atom)] += atom["weight"]
    total = sum(result.values())
    if total <= 0:
        raise RuntimeError("V55 static marginal has zero mass")
    return {key: value / total for key, value in result.items()}


def _configuration_key(world, queue):
    return canonical_json({
        "world": sorted(world.items()),
        "queue": sorted(queue, key=canonical_json),
    })


def _world_from_signature(signature):
    import json

    rows = json.loads(signature)
    return {row["atom"]: row["value"] for row in rows}


def _merge_normalized_atoms(rows):
    merged = {}
    for atom in rows:
        key = (
            atom["program_index"], atom["node_index"],
            float(atom["theta"]).hex(), atom["configuration_key"],
        )
        if key not in merged:
            merged[key] = {**atom, "weight": 0.0}
        merged[key]["weight"] += atom["weight"]
    result = list(merged.values())
    total = sum(atom["weight"] for atom in result)
    if total <= 0:
        raise RuntimeError("V55 atom merge has zero mass")
    for atom in result:
        atom["weight"] /= total
    return result


def restore_static_marginal(
    posterior_atoms, predictive_atoms, observed_world_signature, target_marginal
):
    """Disable static learning while retaining within-static dynamic filtering.

    Each program/theta component keeps its preregistered mass.  When an
    observation has zero likelihood under one component, the control is kept
    total by synchronizing that component's predicted world to the observed
    world while retaining its exact predictive queue distribution.  This is a
    deliberately misspecified control, not a Bayesian posterior.
    """
    observed_world = _world_from_signature(observed_world_signature)
    conditioned = defaultdict(list)
    predictive = defaultdict(list)
    for atom in posterior_atoms:
        conditioned[static_key(atom)].append(atom)
    for atom in predictive_atoms:
        predictive[static_key(atom)].append(atom)

    restored = []
    for key, target_mass in sorted(target_marginal.items()):
        source = conditioned.get(key)
        if source:
            within_total = sum(atom["weight"] for atom in source)
            rows = source
        else:
            rows = []
            for atom in predictive.get(key, []):
                row = {
                    **atom,
                    "world": dict(observed_world),
                    "configuration_key": _configuration_key(
                        observed_world, atom["queue"]
                    ),
                }
                rows.append(row)
            within_total = sum(atom["weight"] for atom in rows)
        if within_total <= 0:
            raise RuntimeError(
                "V55 disabled-update control lost a static component"
            )
        for atom in rows:
            restored.append({
                **atom,
                "weight": target_mass * atom["weight"] / within_total,
            })
    result = _merge_normalized_atoms(restored)
    actual = static_marginal(result)
    if any(abs(actual[key] - value) > 1e-12 for key, value in target_marginal.items()):
        raise RuntimeError("V55 disabled-update static marginal drifted")
    return result


def disabled_static_step(
    atoms, registry, entity_rows, action, tick, target_marginal, program_cache=None
):
    """Exact observation branches for the frozen-static-marginal control."""
    program_cache = {} if program_cache is None else program_cache
    bayes_branches = step_belief(
        atoms, registry, entity_rows, action, tick, program_cache
    )
    predictive_atoms = marginal_step(
        atoms, registry, entity_rows, action, tick
    )
    return {
        outcome: {
            "probability": branch["probability"],
            "atoms": restore_static_marginal(
                branch["atoms"], predictive_atoms, outcome, target_marginal
            ),
        }
        for outcome, branch in bayes_branches.items()
    }


def plan_static_update_disabled(
    atoms, registry, entity_rows, goal, horizon, tick, config,
    target_marginal=None, memo=None, program_cache=None,
):
    """Plan in the misspecified model that cannot learn program or theta."""
    target_marginal = (
        static_marginal(atoms) if target_marginal is None else target_marginal
    )
    memo = {} if memo is None else memo
    program_cache = {} if program_cache is None else program_cache
    if horizon == 0:
        return {"value": terminal_value(atoms, goal), "terminal": True}
    key = (
        "static-disabled", horizon, tick, goal["atom"], goal["value"],
        belief_key(atoms), tuple(sorted(target_marginal.items())),
    )
    if key in memo:
        return memo[key]
    rows = []
    for candidate in candidate_actions(entity_rows):
        branches = disabled_static_step(
            atoms, registry, entity_rows, candidate["action"], tick,
            target_marginal, program_cache,
        )
        children = {}
        future = 0.0
        for outcome, branch in branches.items():
            child = plan_static_update_disabled(
                branch["atoms"], registry, entity_rows, goal,
                horizon - 1, tick + 1, config, target_marginal,
                memo, program_cache,
            )
            children[outcome] = child
            future += branch["probability"] * child["value"]
        rows.append({
            "action": candidate["action"],
            "action_key": candidate["key"],
            "value": -action_cost(candidate["action"], config) + future,
            "branches": children,
        })
    maximum, selected, optimal = _select_action(
        rows, config["planningModel"]["tieTolerance"]
    )
    result = {
        "value": maximum,
        "terminal": False,
        "selected_action": selected["action"],
        "selected_action_key": selected["action_key"],
        "optimal_action_keys": optimal,
        "action_values": {row["action_key"]: row["value"] for row in rows},
        "branches": selected["branches"],
    }
    memo[key] = result
    return result


def evaluate_static_update_disabled_policy(
    actual_atoms, controlled_atoms, registry, entity_rows, goal,
    horizon, tick, config, target_marginal=None,
):
    """Evaluate the disabled-learning controller under the true joint belief."""
    if horizon == 0:
        return terminal_value(actual_atoms, goal)
    target_marginal = (
        static_marginal(controlled_atoms)
        if target_marginal is None else target_marginal
    )
    policy = plan_static_update_disabled(
        controlled_atoms, registry, entity_rows, goal, horizon, tick, config,
        target_marginal,
    )
    action = policy["selected_action"]
    actual_branches = step_belief(
        actual_atoms, registry, entity_rows, action, tick, {}
    )
    controlled_branches = disabled_static_step(
        controlled_atoms, registry, entity_rows, action, tick, target_marginal, {}
    )
    value = -action_cost(action, config)
    for outcome, actual_branch in actual_branches.items():
        if outcome not in controlled_branches:
            raise RuntimeError("V55 disabled control omitted an actual observation")
        value += actual_branch["probability"] * evaluate_static_update_disabled_policy(
            actual_branch["atoms"], controlled_branches[outcome]["atoms"],
            registry, entity_rows, goal, horizon - 1, tick + 1, config,
            target_marginal,
        )
    return value


def clairvoyant_value(atoms, registry, entity_rows, goal, horizon, tick, config):
    """Posterior mean value after revealing the complete root latent atom."""
    cache = {}
    value = 0.0
    for atom in atoms:
        key = (
            atom["program_index"], float(atom["theta"]).hex(),
            atom["configuration_key"],
        )
        if key not in cache:
            revealed = [{**atom, "weight": 1.0}]
            cache[key] = plan_exact(
                revealed, registry, entity_rows, goal, horizon, tick, config
            )["value"]
        value += atom["weight"] * cache[key]
    return value


def attempted_future_outcome_leak(root_belief, future_observation):
    del root_belief, future_observation
    raise PermissionError(
        "V55 root-action firewall rejects access to future observations"
    )
