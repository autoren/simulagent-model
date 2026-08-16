"""Independent bounded execution and explicit-model utilities for V61."""
from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict, deque
from fractions import Fraction
from typing import Any, Sequence

from v22_relational import canonical_json
from v42_stateful import world_signature
from v53_smc2 import continuous_unit_transition, instantiate_program
from v55_planning import candidate_actions
from v56_verification import (
    _add_state,
    _add_transition,
    _apply_effect_independent,
    _evaluate_expression_independent,
    _group_transitions,
    _resolve_atom_independent,
    _validate_action_independent,
    configuration_key,
    prove_support_equivalence,
    validate_world_queue_action,
)


def _public_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def independent_deployment_action(
    node, action_rows: list[dict], history: list[dict], fallback_seed: int,
    mutation: str | None = None,
) -> dict:
    """Reimplement V59 deployment without calling its deployment evaluator."""
    by_key = {row["key"]: row for row in action_rows}
    if node is not None:
        visited = [
            (key, stats) for key, stats in node.actions.items()
            if stats.visits > 0
        ]
        if visited:
            if mutation == "choose_maximum_mean_instead_of_maximum_visits":
                key, _ = min(
                    visited,
                    key=lambda item: (-item[1].mean_return, -item[1].visits, item[0]),
                )
            else:
                key, _ = min(
                    visited,
                    key=lambda item: (-item[1].visits, -item[1].mean_return, item[0]),
                )
            return by_key[key]
    seed = fallback_seed + int(mutation == "change_public_history_fallback_seed")
    token = canonical_json(history)
    index = _public_seed("v59-fallback", seed, token) % len(action_rows)
    return sorted(action_rows, key=lambda row: row["key"])[index]


def independent_tree_child(node, action_key: str, observation: str):
    if node is None or action_key not in node.actions:
        return None
    return node.actions[action_key].children.get(observation)


def independent_transition_distribution(
    program: dict[str, Any], entity_rows: Sequence[dict[str, str]],
    world: dict[str, bool], queue: Sequence[dict[str, Any]],
    action: dict[str, Any], tick: int, mutation: str | None = None,
) -> list[dict[str, Any]]:
    """Independent exact one-step DSL interpreter, including branch mass."""
    if not validate_world_queue_action(entity_rows, world, queue, None, tick):
        raise ValueError("invalid V61 independent transition source")
    binding_mutation = mutation if mutation == "swap_actor_and_target" else None
    action_id, binding = _validate_action_independent(
        action, entity_rows, binding_mutation
    )
    delivered = dict(world)
    remaining = []
    for event in sorted(queue, key=canonical_json):
        if event["due"] == tick:
            if mutation != "omit_due_queue_delivery":
                delivered = _apply_effect_independent(
                    event["effect"], delivered, event["binding"]
                )
        else:
            remaining.append(copy.deepcopy(event))
    if action_id == "wait":
        return [{"world": delivered, "queue": remaining, "mass": Fraction(1)}]

    rule = next(row for row in program["rules"] if row["action"] == action_id)
    base = dict(delivered)
    for effect in rule["deterministic_immediate"]:
        base = _apply_effect_independent(effect, base, binding)
    stochastic = rule["stochastic_immediate"] or rule["stochastic_delayed"]
    if not stochastic:
        return [{"world": base, "queue": remaining, "mass": Fraction(1)}]
    branch = stochastic[0]
    if (
        "condition" in branch
        and not _evaluate_expression_independent(
            branch["condition"], entity_rows, delivered, binding
        )
    ):
        return [{"world": base, "queue": remaining, "mass": Fraction(1)}]
    probability = Fraction(branch["probability"])
    if mutation == "complement_stochastic_probability":
        probability = 1 - probability
    rows = [{
        "world": base,
        "queue": remaining,
        "mass": 1 - probability,
    }]
    if rule["stochastic_immediate"]:
        success_world = _apply_effect_independent(
            branch["effect"], base, binding
        )
        success_queue = remaining
    else:
        success_world = base
        success_queue = [
            *remaining,
            {
                "due": tick + int(branch["delay"]),
                "effect": copy.deepcopy(branch["effect"]),
                "binding": dict(binding),
            },
        ]
    rows.append({
        "world": success_world,
        "queue": success_queue,
        "mass": probability,
    })
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["mass"]:
            continue
        key = configuration_key(row["world"], row["queue"])
        if key not in grouped:
            grouped[key] = {
                "world": copy.deepcopy(row["world"]),
                "queue": copy.deepcopy(row["queue"]),
                "mass": Fraction(0),
            }
        grouped[key]["mass"] += row["mass"]
    result = [grouped[key] for key in sorted(grouped)]
    if mutation == "drop_one_stochastic_successor" and len(result) > 1:
        result = result[:-1]
    return result


def formal_transition_distribution(
    program: dict[str, Any], entity_rows: Sequence[dict[str, str]],
    world: dict[str, bool], queue: Sequence[dict[str, Any]],
    action: dict[str, Any], tick: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for branch in continuous_unit_transition(
        program, entity_rows, world, list(queue), action, tick
    ).values():
        if not branch["mass"]:
            continue
        key = configuration_key(branch["world"], branch["queue"])
        if key not in grouped:
            grouped[key] = {
                "world": copy.deepcopy(branch["world"]),
                "queue": copy.deepcopy(branch["queue"]),
                "mass": Fraction(0),
            }
        grouped[key]["mass"] += Fraction(branch["mass"])
    return [grouped[key] for key in sorted(grouped)]


def _action_cost(action: dict, config: dict) -> float:
    costs = config.get("formalExecutor", {}).get("actionCosts")
    if costs is None:
        costs = config["planningModel"]["actionCost"]
    return float(costs[action["id"]])


def compile_search_policy_dtmc(
    atoms: list[dict], search, registry: list[dict],
    entity_rows: list[dict[str, str]], goal: dict, horizon: int, tick: int,
    config: dict, mutation: str | None = None,
) -> dict:
    if horizon not in {1, 2, 3, 5, 7}:
        raise ValueError("V61 supports fixture horizons and frozen horizons 3, 5, 7")
    total = sum(float(atom["weight"]) for atom in atoms)
    if not atoms or abs(total - 1.0) > 1e-10:
        raise ValueError("V61 exact root belief must normalize")
    action_rows = candidate_actions(entity_rows)
    model: dict[str, Any] = {
        "states": [],
        "state_index": {},
        "transition_map": {},
        "program_cache": {},
        "normalization_checks": [],
        "runtime_tree_nodes": {canonical_json([]): search.root},
        "action_rows": action_rows,
        "fallback_seed": config["evaluation"]["evaluationSeed"],
    }
    root = _add_state(model, ("root",), {"kind": "root", "depth": -1})
    root_targets: dict[int, float] = defaultdict(float)
    work = deque()
    for atom in atoms:
        history: list[dict] = []
        history_token = canonical_json(history)
        key = (
            "execution", 0, history_token, atom["program_index"],
            atom["node_index"], float(atom["theta"]).hex(),
            configuration_key(atom["world"], atom["queue"]),
        )
        state = _add_state(model, key, {
            "kind": "execution", "depth": 0, "tick": tick,
            "history": history, "history_token": history_token,
            "program_index": atom["program_index"],
            "node_index": atom["node_index"], "theta": atom["theta"],
            "world": copy.deepcopy(atom["world"]),
            "queue": copy.deepcopy(atom["queue"]),
        })
        root_targets[state] += float(atom["weight"]) / total
    root_scale = 0.9 if mutation == "corrupt_initial_distribution_mass" else 1.0
    for state, probability in sorted(root_targets.items()):
        _add_transition(model, root, state, probability * root_scale, 0.0)
        work.append(state)
    model["normalization_checks"].append(sum(root_targets.values()) * root_scale)
    visited: set[int] = set()
    done_key = ("done",)
    dropped = False
    while work:
        state_id = work.popleft()
        if state_id in visited:
            continue
        visited.add(state_id)
        state = model["states"][state_id]
        depth = state["depth"]
        if depth == horizon:
            success = state["world"][goal["atom"]] is bool(goal["value"])
            if mutation == "flip_terminal_success_label":
                success = not success
            state["kind"] = "terminal"
            state["success"] = success
            done = _add_state(
                model, done_key, {"kind": "done", "depth": horizon + 1}
            )
            _add_transition(model, state_id, done, 1.0, 1.0 if success else 0.0)
            model["normalization_checks"].append(1.0)
            continue

        node = model["runtime_tree_nodes"].get(state["history_token"])
        # The compiler uses the frozen implementation; the independent selector
        # is checked separately on every reachable state.
        from v59_planning import _deployment_action
        row = _deployment_action(
            node, action_rows, state["history"], model["fallback_seed"]
        )
        state["selected_action"] = copy.deepcopy(row["action"])
        state["selected_action_key"] = row["key"]
        independent = independent_deployment_action(
            node, action_rows, state["history"], model["fallback_seed"]
        )
        state["independent_selected_action_key"] = independent["key"]
        program_key = (
            state["program_index"], state["node_index"],
            float(state["theta"]).hex(),
        )
        if program_key not in model["program_cache"]:
            model["program_cache"][program_key] = instantiate_program(
                registry[state["program_index"]]["template"], state["theta"]
            )
        branches = formal_transition_distribution(
            model["program_cache"][program_key], entity_rows,
            state["world"], state["queue"], row["action"], state["tick"],
        )
        if mutation == "drop_one_stochastic_successor" and len(branches) > 1 and not dropped:
            branches = branches[:-1]
            dropped = True
        outgoing = 0.0
        for branch in branches:
            probability = float(branch["mass"])
            observation = world_signature(branch["world"])
            routed = "*" if mutation == "merge_observation_routes" else observation
            child = None
            if node is not None and row["key"] in node.actions:
                child = node.actions[row["key"]].children.get(routed)
            next_history = [
                *state["history"],
                {"action_key": row["key"], "observation": observation},
            ]
            history_token = canonical_json(next_history)
            if history_token in model["runtime_tree_nodes"]:
                if model["runtime_tree_nodes"][history_token] is not child:
                    raise RuntimeError("V61 tree history collision")
            else:
                model["runtime_tree_nodes"][history_token] = child
            child_key = (
                "execution", depth + 1, history_token,
                state["program_index"], state["node_index"],
                float(state["theta"]).hex(),
                configuration_key(branch["world"], branch["queue"]),
            )
            child_state = _add_state(model, child_key, {
                "kind": "execution", "depth": depth + 1,
                "tick": state["tick"] + 1,
                "history": next_history, "history_token": history_token,
                "program_index": state["program_index"],
                "node_index": state["node_index"], "theta": state["theta"],
                "world": copy.deepcopy(branch["world"]),
                "queue": copy.deepcopy(branch["queue"]),
            })
            reward = 0.0 if mutation == "omit_action_cost_reward" else -_action_cost(
                row["action"], config
            )
            _add_transition(
                model, state_id, child_state, probability, reward,
                {"observation": observation},
            )
            outgoing += probability
            work.append(child_state)
        model["normalization_checks"].append(outgoing)

    done = model["state_index"].get(done_key)
    if done is None:
        raise RuntimeError("V61 compiled policy has no terminal state")
    _add_transition(model, done, done, 1.0, 0.0)
    model["normalization_checks"].append(1.0)
    transitions = sorted(
        model["transition_map"].values(),
        key=lambda edge: (edge["source"], edge["target"]),
    )
    for edge in transitions:
        edge["annotations"].sort(key=canonical_json)
    return {
        "states": model["states"], "transitions": transitions,
        "root_state": root, "done_state": done,
        "normalization_checks": model["normalization_checks"],
        "goal": copy.deepcopy(goal), "entity_rows": copy.deepcopy(entity_rows),
        "horizon": horizon, "start_tick": tick, "registry": copy.deepcopy(registry),
        "runtime_tree_nodes": model["runtime_tree_nodes"],
        "action_rows": action_rows, "fallback_seed": model["fallback_seed"],
    }


def independent_policy_statistics(
    atoms: list[dict], search, registry: list[dict],
    entity_rows: list[dict[str, str]], goal: dict, horizon: int, tick: int,
    config: dict, deployment_mutation: str | None = None,
    transition_mutation: str | None = None,
) -> dict[str, float | int]:
    action_rows = candidate_actions(entity_rows)
    fallback_seed = config["evaluation"]["evaluationSeed"]
    cache: dict[tuple, dict] = {}
    calls = 0
    normalization_error = 0.0

    def visit(atom, node, history, depth):
        nonlocal calls, normalization_error
        calls += 1
        if depth == horizon:
            success = float(atom["world"][goal["atom"]] is bool(goal["value"]))
            return success, success
        row = independent_deployment_action(
            node, action_rows, history, fallback_seed, deployment_mutation
        )
        key = (
            atom["program_index"], atom["node_index"],
            float(atom["theta"]).hex(),
        )
        if key not in cache:
            cache[key] = instantiate_program(
                registry[atom["program_index"]]["template"], atom["theta"]
            )
        branches = independent_transition_distribution(
            cache[key], entity_rows, atom["world"], atom["queue"],
            row["action"], tick + depth, transition_mutation,
        )
        mass = sum((branch["mass"] for branch in branches), Fraction(0))
        normalization_error = max(normalization_error, abs(float(mass) - 1.0))
        success = 0.0
        value = -_action_cost(row["action"], config)
        for branch in branches:
            observation = world_signature(branch["world"])
            child = independent_tree_child(node, row["key"], observation)
            next_history = [
                *history,
                {"action_key": row["key"], "observation": observation},
            ]
            child_atom = {
                **atom,
                "world": branch["world"],
                "queue": branch["queue"],
            }
            child_success, child_value = visit(
                child_atom, child, next_history, depth + 1
            )
            probability = float(branch["mass"])
            success += probability * child_success
            value += probability * child_value
        return success, value

    total = sum(float(atom["weight"]) for atom in atoms)
    success = value = 0.0
    for atom in atoms:
        weight = float(atom["weight"]) / total
        atom_success, atom_value = visit(atom, search.root, [], 0)
        success += weight * atom_success
        value += weight * atom_value
    return {
        "success_probability": success,
        "expected_return": value,
        "recursive_calls": calls,
        "maximum_transition_normalization_error": normalization_error,
    }


def verify_compiled_model_symbolically(model: dict) -> dict[str, Any]:
    states = {row["id"]: row for row in model["states"]}
    grouped = _group_transitions(model["transitions"])
    invariant_checks = invariant_passes = 0
    support_checks = support_passes = probability_passes = 0
    totality_checks = totality_passes = 0
    deployment_checks = deployment_passes = 0
    unknown = deadlocks = 0
    maximum_probability_error = 0.0
    counterexamples = []
    program_cache: dict[tuple, dict] = {}
    for state in model["states"]:
        if state["kind"] not in {"execution", "terminal"}:
            continue
        action = state.get("selected_action")
        invariant_checks += 1
        valid = validate_world_queue_action(
            model["entity_rows"], state["world"], state["queue"], action,
            state["tick"],
        )
        invariant_passes += int(valid)
        if state["kind"] == "terminal":
            expected = state["world"][model["goal"]["atom"]] is bool(
                model["goal"]["value"]
            )
            invariant_checks += 1
            invariant_passes += int(state.get("success") is expected)
            continue
        node = model["runtime_tree_nodes"].get(state["history_token"])
        independent_action = independent_deployment_action(
            node, model["action_rows"], state["history"], model["fallback_seed"]
        )
        deployment_checks += 1
        deployment_passes += int(
            independent_action["key"] == state["selected_action_key"]
            == state["independent_selected_action_key"]
        )
        outgoing = grouped.get(state["id"], [])
        if not outgoing:
            deadlocks += 1
            continue
        program_key = (
            state["program_index"], state["node_index"],
            float(state["theta"]).hex(),
        )
        if program_key not in program_cache:
            program_cache[program_key] = instantiate_program(
                model["registry"][state["program_index"]]["template"],
                state["theta"],
            )
        independent = independent_transition_distribution(
            program_cache[program_key], model["entity_rows"], state["world"],
            state["queue"], state["selected_action"], state["tick"],
        )
        independent_map = {
            configuration_key(row["world"], row["queue"]): row
            for row in independent
        }
        emitted_map: dict[str, dict] = {}
        emitted_mass: dict[str, float] = defaultdict(float)
        for edge in outgoing:
            target = states[edge["target"]]
            if target["kind"] not in {"execution", "terminal"}:
                continue
            key = configuration_key(target["world"], target["queue"])
            emitted_map[key] = {"world": target["world"], "queue": target["queue"]}
            emitted_mass[key] += float(edge["probability"])
            totality_checks += 1
            observations = {row.get("observation") for row in edge["annotations"]}
            observation = world_signature(target["world"])
            expected_history = [
                *state["history"],
                {"action_key": state["selected_action_key"], "observation": observation},
            ]
            totality_passes += int(
                observation in observations and target["history"] == expected_history
            )
        proof = prove_support_equivalence(
            [
                {"world": row["world"], "queue": row["queue"]}
                for row in independent
            ],
            [emitted_map[key] for key in sorted(emitted_map)],
        )
        support_checks += 1
        support_passes += int(proof["equivalent"])
        unknown += int(proof["status"] == "unknown")
        keys = set(independent_map) | set(emitted_mass)
        error = max(
            (
                abs(float(independent_map[key]["mass"]) - emitted_mass[key])
                if key in independent_map else abs(emitted_mass[key])
            )
            for key in keys
        ) if keys else math.inf
        maximum_probability_error = max(maximum_probability_error, error)
        probability_passes += int(error <= 1e-15)
        if (not proof["equivalent"] or error > 1e-15) and len(counterexamples) < 10:
            counterexamples.append({
                "state": state["id"], "support": proof,
                "probability_error": error,
            })
    return {
        "invariant_checks": invariant_checks,
        "invariant_passes": invariant_passes,
        "support_checks": support_checks,
        "support_passes": support_passes,
        "probability_passes": probability_passes,
        "maximum_probability_error": maximum_probability_error,
        "totality_checks": totality_checks,
        "totality_passes": totality_passes,
        "deployment_checks": deployment_checks,
        "deployment_passes": deployment_passes,
        "z3_unknown_count": unknown,
        "nonterminal_deadlock_count": deadlocks,
        "counterexamples": counterexamples,
    }


def strip_runtime_model(model: dict) -> dict:
    return {
        key: value for key, value in model.items()
        if key not in {
            "runtime_tree_nodes", "action_rows", "fallback_seed", "registry",
        }
    }


def hoeffding_radius(config: dict, horizon: int) -> float:
    row = config["probabilisticVerification"]["monteCarloBound"]
    reward_range = float(row["rewardRangeByHorizon"][str(horizon)])
    return reward_range * math.sqrt(
        math.log(2 * int(row["comparisons"]) / float(row["familywiseAlpha"]))
        / (2 * int(row["episodesPerPolicy"]))
    )
