"""Independent symbolic and probabilistic policy-verification utilities for V56."""
from __future__ import annotations

import json
import math
import re
import subprocess
from collections import defaultdict, deque
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import z3

from v22_relational import canonical_json, relation_atom, unary_atom
from v42_stateful import atom_universe, world_signature
from v53_smc2 import continuous_unit_transition, instantiate_program


def tool_versions() -> dict[str, str]:
    completed = subprocess.run(
        ["storm", "--version"], check=True, capture_output=True, text=True
    )
    match = re.search(r"Storm\s+(\d+\.\d+\.\d+)", completed.stdout)
    if not match:
        raise RuntimeError("Unable to parse Storm version")
    return {"storm": match.group(1), "z3": z3.get_version_string()}


def _queue_key(queue: Sequence[dict[str, Any]]) -> str:
    return canonical_json(sorted(queue, key=canonical_json))


def configuration_key(world: dict[str, bool], queue: Sequence[dict[str, Any]]) -> str:
    return canonical_json({
        "world": sorted(world.items()),
        "queue": sorted(queue, key=canonical_json),
    })


def _validate_action_independent(
    action: dict[str, Any], entity_rows: Sequence[dict[str, str]],
    mutation: str | None = None,
) -> tuple[str, dict[str, str]]:
    action_id = action.get("id")
    binding = action.get("binding", {})
    if action_id not in {"pulse", "route", "wait"}:
        raise ValueError("unknown action")
    if action_id == "wait":
        if binding:
            raise ValueError("wait must not have a binding")
        return action_id, {}
    identifiers = {row["id"] for row in entity_rows}
    valid = (
        set(binding) == {"actor", "target"}
        and set(binding.values()) <= identifiers
        and (
            mutation == "permit_self_binding"
            or binding.get("actor") != binding.get("target")
        )
    )
    if not valid:
        raise ValueError("invalid bound action")
    if mutation == "swap_actor_and_target":
        binding = {"actor": binding["target"], "target": binding["actor"]}
    return action_id, dict(binding)


def _resolve_atom_independent(
    expression: dict[str, Any], binding: dict[str, str],
    local: dict[str, str] | None = None,
) -> str:
    local = {} if local is None else local

    def resolve(name: str) -> str:
        if name in local:
            return local[name]
        if name in binding:
            return binding[name]
        raise ValueError(f"unbound variable {name}")

    if expression["op"] == "unary":
        return unary_atom(expression["predicate"], resolve(expression["var"]))
    if expression["op"] == "relation":
        return relation_atom(
            expression["predicate"],
            resolve(expression["source"]),
            resolve(expression["target"]),
        )
    raise ValueError("effect target is not an atom")


def _evaluate_expression_independent(
    expression: dict[str, Any], entity_rows: Sequence[dict[str, str]],
    world: dict[str, bool], binding: dict[str, str],
    local: dict[str, str] | None = None,
) -> bool:
    local = {} if local is None else dict(local)
    op = expression["op"]
    if op in {"unary", "relation"}:
        return world[_resolve_atom_independent(expression, binding, local)]
    if op == "not":
        return not _evaluate_expression_independent(
            expression["arg"], entity_rows, world, binding, local
        )
    if op in {"and", "or", "xor"}:
        values = [
            _evaluate_expression_independent(
                child, entity_rows, world, binding, local
            )
            for child in expression["args"]
        ]
        if op == "and":
            return all(values)
        if op == "or":
            return any(values)
        return values[0] != values[1]
    if op == "exists":
        excluded = {
            local.get(name, binding.get(name, name))
            for name in expression.get("distinct_from", [])
        }
        for entity in sorted(entity_rows, key=lambda row: row["id"]):
            if (
                entity["entity_type"] == expression["entity_type"]
                and entity["id"] not in excluded
                and _evaluate_expression_independent(
                    expression["where"], entity_rows, world, binding,
                    {**local, expression["var"]: entity["id"]},
                )
            ):
                return True
        return False
    raise ValueError(f"unsupported condition operator {op}")


def _apply_effect_independent(
    effect: dict[str, Any], world: dict[str, bool], binding: dict[str, str]
) -> dict[str, bool]:
    result = dict(world)
    target = _resolve_atom_independent(effect["target"], binding)
    operation = effect["op"]
    if operation == "set_true":
        result[target] = True
    elif operation == "set_false":
        result[target] = False
    elif operation == "toggle":
        result[target] = not result[target]
    elif operation == "copy":
        source = _resolve_atom_independent(effect["source"], binding)
        result[target] = result[source]
    else:
        raise ValueError(f"unsupported effect {operation}")
    return result


def validate_world_queue_action(
    entity_rows: Sequence[dict[str, str]], world: dict[str, bool],
    queue: Sequence[dict[str, Any]], action: dict[str, Any] | None, tick: int,
) -> bool:
    if set(world) != set(atom_universe(entity_rows)):
        return False
    if any(type(value) is not bool for value in world.values()):
        return False
    identifiers = {row["id"] for row in entity_rows}
    for event in queue:
        if set(event) != {"due", "effect", "binding"}:
            return False
        if type(event["due"]) is not int or event["due"] < tick:
            return False
        binding = event["binding"]
        if (
            set(binding) != {"actor", "target"}
            or binding["actor"] == binding["target"]
            or not set(binding.values()) <= identifiers
        ):
            return False
        if event["effect"].get("op") not in {
            "set_true", "set_false", "toggle", "copy"
        }:
            return False
        try:
            _resolve_atom_independent(event["effect"]["target"], binding)
        except (KeyError, ValueError):
            return False
    if action is not None:
        try:
            _validate_action_independent(action, entity_rows)
        except ValueError:
            return False
    return True


def independent_transition_support(
    program: dict[str, Any], entity_rows: Sequence[dict[str, str]],
    world: dict[str, bool], queue: Sequence[dict[str, Any]],
    action: dict[str, Any], tick: int, mutation: str | None = None,
) -> list[dict[str, Any]]:
    """A separate DSL interpreter used only to encode logical support."""
    if not validate_world_queue_action(entity_rows, world, queue, None, tick):
        raise ValueError("invalid independent transition source")
    action_id, binding = _validate_action_independent(
        action, entity_rows, mutation
    )
    delivered = dict(world)
    remaining = []
    due = [event for event in queue if event["due"] == tick]
    for event in sorted(due, key=canonical_json):
        if mutation != "omit_due_queue_delivery":
            delivered = _apply_effect_independent(
                event["effect"], delivered, event["binding"]
            )
    remaining.extend(event for event in queue if event["due"] != tick)
    if action_id == "wait":
        return [{"world": delivered, "queue": remaining}]

    rule = next(row for row in program["rules"] if row["action"] == action_id)
    base = dict(delivered)
    for effect in rule["deterministic_immediate"]:
        base = _apply_effect_independent(effect, base, binding)
    branches = rule["stochastic_immediate"] or rule["stochastic_delayed"]
    if not branches:
        return [{"world": base, "queue": remaining}]
    branch = branches[0]
    condition_world = (
        base if mutation == "evaluate_condition_after_deterministic_effect"
        else delivered
    )
    if (
        "condition" in branch
        and not _evaluate_expression_independent(
            branch["condition"], entity_rows, condition_world, binding
        )
    ):
        return [{"world": base, "queue": remaining}]
    results = [{"world": base, "queue": remaining}]
    if rule["stochastic_immediate"]:
        results.append({
            "world": _apply_effect_independent(branch["effect"], base, binding),
            "queue": remaining,
        })
    else:
        due_tick = tick + branch["delay"]
        if mutation == "shift_delayed_due_tick_by_one":
            due_tick += 1
        event = {
            "due": due_tick,
            "effect": branch["effect"],
            "binding": dict(binding),
        }
        results.append({"world": base, "queue": [*remaining, event]})
    unique = {
        configuration_key(row["world"], row["queue"]): row for row in results
    }
    return [unique[key] for key in sorted(unique)]


def formal_transition_support(
    program: dict[str, Any], entity_rows: Sequence[dict[str, str]],
    world: dict[str, bool], queue: Sequence[dict[str, Any]],
    action: dict[str, Any], tick: int,
) -> list[dict[str, Any]]:
    branches = continuous_unit_transition(
        program, entity_rows, world, list(queue), action, tick
    )
    unique = {
        configuration_key(row["world"], row["queue"]): {
            "world": dict(row["world"]), "queue": list(row["queue"])
        }
        for row in branches.values() if row["mass"]
    }
    return [unique[key] for key in sorted(unique)]


def _configuration_formula(
    row: dict[str, Any], atom_vars: dict[str, z3.BoolRef],
    queue_vars: dict[str, z3.BoolRef],
) -> z3.BoolRef:
    terms: list[z3.BoolRef] = []
    for atom, variable in atom_vars.items():
        terms.append(variable == bool(row["world"][atom]))
    present = {canonical_json(event) for event in row["queue"]}
    for token, variable in queue_vars.items():
        terms.append(variable == (token in present))
    return z3.And(*terms) if terms else z3.BoolVal(True)


def prove_support_equivalence(
    independent: Sequence[dict[str, Any]], emitted: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if not independent and not emitted:
        return {"status": "unsat", "equivalent": True, "counterexample": None}
    sample = (independent or emitted)[0]
    atoms = sorted(sample["world"])
    queue_tokens = sorted({
        canonical_json(event)
        for row in [*independent, *emitted]
        for event in row["queue"]
    })
    atom_vars = {atom: z3.Bool(f"next_atom_{index}") for index, atom in enumerate(atoms)}
    queue_vars = {token: z3.Bool(f"next_event_{index}") for index, token in enumerate(queue_tokens)}

    def relation(rows):
        terms = [
            _configuration_formula(row, atom_vars, queue_vars) for row in rows
        ]
        return z3.Or(*terms) if terms else z3.BoolVal(False)

    solver = z3.Solver()
    solver.add(z3.Xor(relation(independent), relation(emitted)))
    status = solver.check()
    if status == z3.unsat:
        return {"status": "unsat", "equivalent": True, "counterexample": None}
    if status == z3.unknown:
        return {
            "status": "unknown", "equivalent": False,
            "counterexample": solver.reason_unknown(),
        }
    model = solver.model()
    return {
        "status": "sat",
        "equivalent": False,
        "counterexample": {
            "world": {
                atom: z3.is_true(model.eval(variable, model_completion=True))
                for atom, variable in atom_vars.items()
            },
            "queue": [
                token for token, variable in queue_vars.items()
                if z3.is_true(model.eval(variable, model_completion=True))
            ],
        },
    }


def _add_state(model: dict[str, Any], key: tuple, row: dict[str, Any]) -> int:
    if key in model["state_index"]:
        return model["state_index"][key]
    index = len(model["states"])
    model["state_index"][key] = index
    model["states"].append({"id": index, **row})
    return index


def _add_transition(
    model: dict[str, Any], source: int, target: int,
    probability: float, reward: float, annotation: dict[str, Any] | None = None,
):
    key = (source, target)
    if key not in model["transition_map"]:
        model["transition_map"][key] = {
            "source": source,
            "target": target,
            "probability": 0.0,
            "reward": reward,
            "annotations": [],
        }
    row = model["transition_map"][key]
    if abs(row["reward"] - reward) > 1e-15:
        raise RuntimeError("merged DTMC transitions disagree on reward")
    row["probability"] += probability
    if annotation is not None:
        row["annotations"].append(annotation)


def _action_cost(action: dict[str, Any], config: dict[str, Any]) -> float:
    costs = config.get("formalExecutor", {}).get("actionCosts")
    if costs is None:
        costs = config["planningModel"]["actionCost"]
    return float(costs[action["id"]])


def compile_policy_dtmc(
    atoms: Sequence[dict[str, Any]], policy: dict[str, Any],
    registry: Sequence[dict[str, Any]], entity_rows: Sequence[dict[str, str]],
    goal: dict[str, Any], horizon: int, tick: int, config: dict[str, Any],
) -> dict[str, Any]:
    if horizon != 3:
        raise ValueError("V56 compiles only the frozen three-action horizon")
    if abs(sum(float(atom["weight"]) for atom in atoms) - 1.0) > 1e-12:
        raise ValueError("root belief does not normalize")
    model: dict[str, Any] = {
        "states": [],
        "state_index": {},
        "transition_map": {},
        "policy_nodes": {(): policy},
        "program_cache": {},
        "normalization_checks": [],
    }
    root = _add_state(model, ("root",), {"kind": "root", "depth": -1})
    work = deque()
    root_targets: dict[int, float] = defaultdict(float)
    total_weight = sum(float(atom["weight"]) for atom in atoms)
    for atom in atoms:
        path: tuple[str, ...] = ()
        key = (
            "execution", 0, path, atom["program_index"], atom["node_index"],
            float(atom["theta"]).hex(), configuration_key(atom["world"], atom["queue"]),
        )
        state = _add_state(model, key, {
            "kind": "execution",
            "depth": 0,
            "tick": tick,
            "policy_path": list(path),
            "program_index": atom["program_index"],
            "node_index": atom["node_index"],
            "theta": atom["theta"],
            "world": dict(atom["world"]),
            "queue": list(atom["queue"]),
        })
        root_targets[state] += float(atom["weight"]) / total_weight
    for state, probability in sorted(root_targets.items()):
        _add_transition(model, root, state, probability, 0.0)
        work.append(state)
    model["normalization_checks"].append(sum(root_targets.values()))
    visited = set()
    done_key = ("done",)

    while work:
        state_id = work.popleft()
        if state_id in visited:
            continue
        visited.add(state_id)
        state = model["states"][state_id]
        depth = state["depth"]
        path = tuple(state["policy_path"])
        if depth == horizon:
            success = state["world"][goal["atom"]] is bool(goal["value"])
            state["kind"] = "terminal"
            state["success"] = success
            done = _add_state(model, done_key, {"kind": "done", "depth": horizon + 1})
            _add_transition(model, state_id, done, 1.0, 1.0 if success else 0.0)
            model["normalization_checks"].append(1.0)
            continue
        policy_node = model["policy_nodes"].get(path)
        if policy_node is None or policy_node.get("terminal"):
            raise RuntimeError("reachable nonterminal execution state lacks a policy node")
        action = policy_node["selected_action"]
        if not validate_world_queue_action(
            entity_rows, state["world"], state["queue"], action, state["tick"]
        ):
            raise RuntimeError("compiler encountered an invalid reachable state")
        state["selected_action"] = action
        program_key = (
            state["program_index"], state["node_index"], float(state["theta"]).hex()
        )
        if program_key not in model["program_cache"]:
            model["program_cache"][program_key] = instantiate_program(
                registry[state["program_index"]]["template"], state["theta"]
            )
        branches = continuous_unit_transition(
            model["program_cache"][program_key], entity_rows,
            state["world"], state["queue"], action, state["tick"],
        )
        outgoing = 0.0
        for branch in branches.values():
            probability = float(branch["mass"])
            if probability <= 0:
                continue
            observation = world_signature(branch["world"])
            if observation not in policy_node["branches"]:
                raise RuntimeError("frozen policy omits a reachable observation")
            child_path = (*path, observation)
            child = policy_node["branches"][observation]
            if (
                child_path in model["policy_nodes"]
                and model["policy_nodes"][child_path] != child
            ):
                raise RuntimeError("policy path collision")
            model["policy_nodes"][child_path] = child
            child_key = (
                "execution", depth + 1, child_path,
                state["program_index"], state["node_index"],
                float(state["theta"]).hex(),
                configuration_key(branch["world"], branch["queue"]),
            )
            child_state = _add_state(model, child_key, {
                "kind": "execution",
                "depth": depth + 1,
                "tick": state["tick"] + 1,
                "policy_path": list(child_path),
                "program_index": state["program_index"],
                "node_index": state["node_index"],
                "theta": state["theta"],
                "world": dict(branch["world"]),
                "queue": list(branch["queue"]),
            })
            _add_transition(
                model, state_id, child_state, probability,
                -_action_cost(action, config),
                {"observation": observation},
            )
            outgoing += probability
            work.append(child_state)
        model["normalization_checks"].append(outgoing)

    done = model["state_index"].get(done_key)
    if done is None:
        raise RuntimeError("compiled policy has no terminal state")
    _add_transition(model, done, done, 1.0, 0.0)
    model["normalization_checks"].append(1.0)
    transitions = sorted(
        model["transition_map"].values(),
        key=lambda row: (row["source"], row["target"]),
    )
    for row in transitions:
        row["annotations"].sort(key=canonical_json)
    for source, rows in _group_transitions(transitions).items():
        total = sum(row["probability"] for row in rows)
        if abs(total - 1.0) > 1e-12:
            raise RuntimeError(f"DTMC row {source} does not normalize: {total}")
    return {
        "states": model["states"],
        "transitions": transitions,
        "root_state": root,
        "done_state": done,
        "normalization_checks": model["normalization_checks"],
        "goal": goal,
        "entity_rows": list(entity_rows),
        "horizon": horizon,
        "start_tick": tick,
        "registry": list(registry),
    }


def _group_transitions(transitions):
    result = defaultdict(list)
    for row in transitions:
        result[row["source"]].append(row)
    return result


def direct_policy_statistics(
    atoms: Sequence[dict[str, Any]], policy: dict[str, Any],
    registry: Sequence[dict[str, Any]], entity_rows: Sequence[dict[str, str]],
    goal: dict[str, Any], horizon: int, tick: int, config: dict[str, Any],
) -> dict[str, float]:
    cache = {}

    def visit(atom, node, remaining, current_tick):
        if remaining == 0:
            success = float(atom["world"][goal["atom"]] is bool(goal["value"]))
            return success, success
        key = (
            atom["program_index"], atom["node_index"], float(atom["theta"]).hex()
        )
        if key not in cache:
            cache[key] = instantiate_program(
                registry[atom["program_index"]]["template"], atom["theta"]
            )
        action = node["selected_action"]
        branches = continuous_unit_transition(
            cache[key], entity_rows, atom["world"], atom["queue"],
            action, current_tick,
        )
        success, value = 0.0, -_action_cost(action, config)
        for branch in branches.values():
            probability = float(branch["mass"])
            observation = world_signature(branch["world"])
            child_atom = {
                **atom, "world": branch["world"], "queue": branch["queue"]
            }
            child_success, child_value = visit(
                child_atom, node["branches"][observation], remaining - 1,
                current_tick + 1,
            )
            success += probability * child_success
            value += probability * child_value
        return success, value

    success, value = 0.0, 0.0
    total = sum(float(atom["weight"]) for atom in atoms)
    for atom in atoms:
        weight = float(atom["weight"]) / total
        atom_success, atom_value = visit(atom, policy, horizon, tick)
        success += weight * atom_success
        value += weight * atom_value
    return {"success_probability": success, "expected_return": value}


def model_statistics(model: dict[str, Any]) -> dict[str, float]:
    transitions = _group_transitions(model["transitions"])
    states = {row["id"]: row for row in model["states"]}
    memo = {}

    def visit(state_id):
        if state_id == model["done_state"]:
            return 0.0, 0.0, 1.0
        if state_id in memo:
            return memo[state_id]
        success, reward, termination = 0.0, 0.0, 0.0
        for edge in transitions[state_id]:
            target = states[edge["target"]]
            if target["kind"] == "done":
                child_success = float(states[state_id].get("success", False))
                child_reward, child_termination = 0.0, 1.0
            else:
                child_success, child_reward, child_termination = visit(edge["target"])
            success += edge["probability"] * child_success
            reward += edge["probability"] * (edge["reward"] + child_reward)
            termination += edge["probability"] * child_termination
        memo[state_id] = success, reward, termination
        return memo[state_id]

    success, reward, termination = visit(model["root_state"])
    return {
        "success_probability": success,
        "expected_return": reward,
        "termination_probability": termination,
    }


def write_explicit_model(model: dict[str, Any], directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    transitions = model["transitions"]
    (directory / "model.tra").write_text(
        "dtmc\n" + "".join(
            f'{row["source"]} {row["target"]} {row["probability"]:.17g}\n'
            for row in transitions
        )
    )
    labels = ["#DECLARATION", "init success done", "#END"]
    for state in model["states"]:
        values = []
        if state["id"] == model["root_state"]:
            values.append("init")
        if state.get("success"):
            values.append("success")
        if state["id"] == model["done_state"]:
            values.append("done")
        if values:
            labels.append(f'{state["id"]} {" ".join(values)}')
    (directory / "model.lab").write_text("\n".join(labels) + "\n")
    (directory / "model.rew").write_text("".join(
        f'{row["source"]} {row["target"]} {row["reward"]:.17g}\n'
        for row in transitions
    ))


def write_policy_bundle(
    model: dict[str, Any], policy: dict[str, Any], directory: Path,
    metadata: dict[str, Any],
):
    write_explicit_model(model, directory)
    serializable_model = {
        key: value for key, value in model.items()
        if key not in {"registry"}
    }
    (directory / "model.meta.json").write_text(json.dumps(
        {**metadata, "model": serializable_model, "registry": model["registry"]},
        indent=2, sort_keys=True,
    ) + "\n")
    (directory / "policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n"
    )


def run_storm_property(directory: Path, property_text: str) -> float:
    completed = subprocess.run(
        [
            "storm", "--explicit", str(directory / "model.tra"),
            str(directory / "model.lab"), "--transrew",
            str(directory / "model.rew"), "--prop", property_text,
        ],
        check=True, capture_output=True, text=True,
    )
    matches = re.findall(
        r"Result \((?:for )?initial states?\):\s*([-+0-9.eE/]+)",
        completed.stdout,
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Unable to parse exactly one Storm result for {property_text}: "
            f"{completed.stdout[-2000:]}"
        )
    token = matches[0]
    return float(Fraction(token)) if "/" in token else float(token)


def run_storm_properties(directory: Path) -> dict[str, float]:
    return {
        "termination_probability": run_storm_property(
            directory, 'P=? [F "done"]'
        ),
        "success_probability": run_storm_property(
            directory, 'P=? [F "success"]'
        ),
        "expected_return": run_storm_property(
            directory, 'R=? [F "done"]'
        ),
    }


def verify_compiled_model_symbolically(model: dict[str, Any]) -> dict[str, Any]:
    states = {row["id"]: row for row in model["states"]}
    transitions = _group_transitions(model["transitions"])
    invariant_checks = invariant_passes = 0
    support_checks = support_passes = totality_checks = totality_passes = 0
    unknown = deadlocks = 0
    counterexamples = []
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
        outgoing = transitions.get(state["id"], [])
        if not outgoing:
            deadlocks += 1
            continue
        program = instantiate_program(
            model["registry"][state["program_index"]]["template"], state["theta"]
        )
        independent = independent_transition_support(
            program, model["entity_rows"], state["world"], state["queue"],
            action, state["tick"],
        )
        emitted_map = {}
        for edge in outgoing:
            target = states[edge["target"]]
            if target["kind"] not in {"execution", "terminal"}:
                continue
            emitted_map[configuration_key(target["world"], target["queue"])] = {
                "world": target["world"], "queue": target["queue"]
            }
            totality_checks += 1
            observation = world_signature(target["world"])
            annotations = {row.get("observation") for row in edge["annotations"]}
            totality_passes += int(observation in annotations)
        proof = prove_support_equivalence(
            independent, [emitted_map[key] for key in sorted(emitted_map)]
        )
        support_checks += 1
        support_passes += int(proof["equivalent"])
        unknown += int(proof["status"] == "unknown")
        if not proof["equivalent"] and len(counterexamples) < 10:
            counterexamples.append({"state": state["id"], **proof})
    return {
        "invariant_checks": invariant_checks,
        "invariant_passes": invariant_passes,
        "support_checks": support_checks,
        "support_passes": support_passes,
        "totality_checks": totality_checks,
        "totality_passes": totality_passes,
        "z3_unknown_count": unknown,
        "nonterminal_deadlock_count": deadlocks,
        "counterexamples": counterexamples,
    }


def finite_model(model: dict[str, Any]) -> bool:
    return all(
        math.isfinite(row[key])
        for row in model["transitions"]
        for key in ("probability", "reward")
    )


def transition_rows_normalize(model: dict[str, Any], tolerance: float = 1e-12) -> bool:
    grouped = _group_transitions(model["transitions"])
    expected_sources = {row["id"] for row in model["states"]}
    return set(grouped) == expected_sources and all(
        abs(sum(row["probability"] for row in rows) - 1.0) <= tolerance
        for rows in grouped.values()
    )
