"""Exact rational stochastic transition DSL and executor for V46."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Any, Sequence

from v22_relational import canonical_expression, canonical_json, evaluate_expression, sha256_text
from v42_stateful import (
    ONTOLOGY, _effect_assignment, atom_universe, canonical_effect, effect, effect_key,
    negate, relation, unary, world_signature,
)


ACTIONS = ("pulse", "route", "wait")
RULE_ACTIONS = ("pulse", "route")
PROBABILITIES = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
PAYLOAD_OPS = ("set_true", "set_false", "toggle")


def stochastic(probability: str, payload: dict[str, Any], condition: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"probability": probability, "effect": payload}
    if condition is not None:
        row["condition"] = condition
    return row


def delayed(delay: int, branch: dict[str, Any]) -> dict[str, Any]:
    return {"delay": delay, **branch}


def _rule(action: str, deterministic_immediate=(), stochastic_immediate=(), stochastic_delayed=()):
    return {
        "action": action,
        "deterministic_immediate": list(deterministic_immediate),
        "stochastic_immediate": list(stochastic_immediate),
        "stochastic_delayed": list(stochastic_delayed),
    }


def _fraction(value: str) -> Fraction:
    result = Fraction(value)
    if result not in PROBABILITIES:
        raise ValueError("V46 probability lies outside the declared vocabulary")
    return result


def canonical_branch(value: dict[str, Any], delayed_branch: bool = False) -> dict[str, Any]:
    probability = _fraction(str(value.get("probability")))
    payload = canonical_effect(value.get("effect", {}))
    if payload["op"] not in PAYLOAD_OPS:
        raise ValueError("V46 stochastic payload must be set, clear, or toggle")
    row: dict[str, Any] = {
        "probability": f"{probability.numerator}/{probability.denominator}",
        "effect": payload,
    }
    if delayed_branch:
        if value.get("delay") not in (1, 2):
            raise ValueError("V46 delayed branch requires one or two ticks")
        row["delay"] = value["delay"]
    if "condition" in value:
        row["condition"] = canonical_expression(value["condition"])
    return row


def canonical_program(program: dict[str, Any]) -> dict[str, Any]:
    rules, seen, stochastic_count, scheduled_count = [], set(), 0, 0
    for source in program.get("rules", []):
        action = source.get("action")
        if action not in RULE_ACTIONS or action in seen:
            raise ValueError("V46 requires unique pulse and route rules")
        seen.add(action)
        deterministic = [canonical_effect(row) for row in source.get("deterministic_immediate", [])]
        immediate = [canonical_branch(row) for row in source.get("stochastic_immediate", [])]
        scheduled = [canonical_branch(row, delayed_branch=True) for row in source.get("stochastic_delayed", [])]
        if len(deterministic) > 1 or len(immediate) > 1 or len(scheduled) > 1 or (immediate and scheduled):
            raise ValueError("V46 permits one deterministic effect and one stochastic immediate-or-delayed branch per action")
        stochastic_count += len(immediate) + len(scheduled)
        scheduled_count += len(scheduled)
        deterministic.sort(key=effect_key)
        rules.append({
            "action": action,
            "deterministic_immediate": deterministic,
            "stochastic_immediate": immediate,
            "stochastic_delayed": scheduled,
        })
    if seen != set(RULE_ACTIONS) or stochastic_count < 1:
        raise ValueError("V46 programs require both action rules and at least one stochastic branch")
    if scheduled_count > 1:
        raise ValueError("V46 permits one delayed branch per program to forbid same-tick delivery conflicts")
    rules.sort(key=lambda row: row["action"])
    return {"dsl_version": 4, "rules": rules}


def program_key(program: dict[str, Any]) -> str:
    return canonical_json(canonical_program(program))


def _validate_action(action: dict[str, Any], entities: Sequence[dict[str, str]]) -> tuple[str, dict[str, str]]:
    action_id, binding = action.get("id"), action.get("binding", {})
    if action_id not in ACTIONS:
        raise ValueError("Unknown V46 action")
    if action_id == "wait":
        if binding:
            raise ValueError("V46 wait takes no binding")
        return action_id, binding
    identifiers = {row["id"] for row in entities}
    if set(binding) != {"actor", "target"} or binding["actor"] == binding["target"] or not set(binding.values()) <= identifiers:
        raise ValueError("V46 bound action requires two distinct known entities")
    return action_id, binding


def _condition_holds(branch, world, binding, entities):
    return "condition" not in branch or evaluate_expression(branch["condition"], ONTOLOGY, entities, world, binding)


def _event_key(event):
    return canonical_json(event)


def _configuration_key(world, queue):
    return canonical_json({"world": sorted(world.items()), "queue": sorted(queue, key=_event_key)})


def _accumulate(target, world, queue, mass):
    if not mass:
        return
    key = _configuration_key(world, queue)
    if key not in target:
        target[key] = {"world": dict(world), "queue": list(queue), "mass": Fraction(0)}
    target[key]["mass"] += mass


def _apply_payload(payload, world, binding, entities):
    assigned = _effect_assignment(payload, world, binding, entities)
    if assigned is None:
        return dict(world)
    atom, value = assigned
    return {**world, atom: value}


def _deliver(queue, tick, world, entities):
    current, remaining = dict(world), []
    due = [event for event in queue if event["due"] == tick]
    for event in sorted(due, key=_event_key):
        current = _apply_payload(event["effect"], current, event["binding"], entities)
    remaining.extend(event for event in queue if event["due"] != tick)
    return current, remaining


def _distribution_rows(world_mass: dict[str, Fraction]) -> list[dict[str, Any]]:
    return [
        {
            "world": signature,
            "mass": {"numerator": mass.numerator, "denominator": mass.denominator},
        }
        for signature, mass in sorted(world_mass.items()) if mass
    ]


def execute_distribution(
    program: dict[str, Any], entities: Sequence[dict[str, str]], initial_world: dict[str, bool],
    actions: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if set(initial_world) != set(atom_universe(entities)):
        raise ValueError("V46 requires a complete Boolean initial world")
    normalized = canonical_program(program)
    rules = {row["action"]: row for row in normalized["rules"]}
    configurations = {_configuration_key(initial_world, []): {"world": dict(initial_world), "queue": [], "mass": Fraction(1)}}
    trajectory = []
    for tick, action in enumerate(actions):
        action_id, binding = _validate_action(action, entities)
        successors = {}
        for configuration in configurations.values():
            world, queue = _deliver(configuration["queue"], tick, configuration["world"], entities)
            mass = configuration["mass"]
            if action_id == "wait":
                _accumulate(successors, world, queue, mass)
                continue
            rule = rules[action_id]
            base = dict(world)
            for payload in rule["deterministic_immediate"]:
                base = _apply_payload(payload, base, binding, entities)
            branch = (rule["stochastic_immediate"] or rule["stochastic_delayed"])
            if not branch or not _condition_holds(branch[0], world, binding, entities):
                _accumulate(successors, base, queue, mass)
                continue
            branch = branch[0]
            probability = _fraction(branch["probability"])
            _accumulate(successors, base, queue, mass * (1 - probability))
            if rule["stochastic_immediate"]:
                applied = _apply_payload(branch["effect"], base, binding, entities)
                _accumulate(successors, applied, queue, mass * probability)
            else:
                event = {"due": tick + branch["delay"], "effect": branch["effect"], "binding": dict(binding)}
                _accumulate(successors, base, [*queue, event], mass * probability)
        configurations = successors
        if sum(row["mass"] for row in configurations.values()) != 1:
            raise RuntimeError("V46 configuration mass does not normalize")
        marginal: dict[str, Fraction] = {}
        for configuration in configurations.values():
            signature = world_signature(configuration["world"])
            marginal[signature] = marginal.get(signature, Fraction(0)) + configuration["mass"]
        if sum(marginal.values()) != 1:
            raise RuntimeError("V46 observation mass does not normalize")
        trajectory.append(_distribution_rows(marginal))
    if not trajectory:
        raise ValueError("V46 sequences must contain an action")
    return trajectory


def distribution_key(trajectory) -> str:
    return canonical_json(trajectory)


def uniformized(trajectory):
    result = []
    for step in trajectory:
        count = len(step)
        mass = Fraction(1, count)
        result.append([{"world": row["world"], "mass": {"numerator": mass.numerator, "denominator": mass.denominator}} for row in step])
    return result


def map_determinized(trajectory):
    result = []
    for step in trajectory:
        selected = min(
            step,
            key=lambda row: (-Fraction(row["mass"]["numerator"], row["mass"]["denominator"]), row["world"]),
        )
        result.append([{"world": selected["world"], "mass": {"numerator": 1, "denominator": 1}}])
    return result


def total_variation(left, right) -> Fraction:
    left_map = {row["world"]: Fraction(row["mass"]["numerator"], row["mass"]["denominator"]) for row in left}
    right_map = {row["world"]: Fraction(row["mass"]["numerator"], row["mass"]["denominator"]) for row in right}
    return sum((abs(left_map.get(key, 0) - right_map.get(key, 0)) for key in set(left_map) | set(right_map)), Fraction(0)) / 2


def mechanic_registry():
    mechanics = []
    targets = [unary(predicate, variable) for predicate, variable in product(("active", "marked", "ready"), ("actor", "target"))]
    for index in range(10):
        probability = PROBABILITIES[index % 3]
        operation = PAYLOAD_OPS[(index // 3) % 3]
        target = targets[index % len(targets)]
        program = {"rules": [
            _rule("pulse", stochastic_immediate=[stochastic(str(probability), effect(operation, target))]),
            _rule("route", deterministic_immediate=[effect("toggle", relation("actor", "target"))]),
        ]}
        mechanics.append({"family": "immediate_bernoulli_mutation", "ordinal": index, "probability": str(probability), "timing": "immediate", "program": canonical_program(program)})
    delayed_targets = (relation("actor", "target"), unary("ready", "target"))
    for index in range(10):
        probability = PROBABILITIES[(index + 1) % 3]
        delay_ticks = 1 + ((index // 3) % 2)
        operation = PAYLOAD_OPS[(index // 6) % 3]
        target = delayed_targets[index % len(delayed_targets)]
        program = {"rules": [
            _rule("pulse", stochastic_delayed=[delayed(delay_ticks, stochastic(str(probability), effect(operation, target)))]),
            _rule("route", deterministic_immediate=[effect("toggle", unary("marked", "actor"))]),
        ]}
        mechanics.append({"family": "delayed_bernoulli_scheduling", "ordinal": index, "probability": str(probability), "timing": "delayed", "program": canonical_program(program)})
    conditions = (unary("active", "actor"), negate(unary("active", "actor")), unary("marked", "target"), relation("actor", "target"), negate(relation("actor", "target")))
    for index in range(10):
        condition = conditions[index % len(conditions)]
        probability = PROBABILITIES[(index + 2) % 3]
        program = {"rules": [
            _rule("pulse", deterministic_immediate=[effect("toggle", unary("active", "actor"))]),
            _rule("route", stochastic_immediate=[stochastic(str(probability), effect("toggle", unary("ready", "target")), condition)]),
        ]}
        mechanics.append({"family": "state_conditional_probability", "ordinal": index, "probability": str(probability), "timing": "immediate", "program": canonical_program(program)})
    for index in range(10):
        probability = PROBABILITIES[index % 3]
        delay_ticks = 1 + ((index // 3) % 2)
        operation = ("set_true", "toggle")[(index // 6) % 2]
        predicate = ("active", "marked")[index % 2]
        target = unary(predicate, "target")
        program = {"rules": [
            _rule("pulse", stochastic_delayed=[delayed(delay_ticks, stochastic(str(probability), effect(operation, target)))]),
            _rule("route", deterministic_immediate=[effect("toggle", target)]),
        ]}
        mechanics.append({"family": "interleaved_deterministic_and_stochastic", "ordinal": index, "probability": str(probability), "timing": "delayed", "program": canonical_program(program)})
    keys = [program_key(row["program"]) for row in mechanics]
    if len(mechanics) != 40 or len(set(keys)) != 40:
        raise RuntimeError("V46 registry must contain 40 unique programs")
    for row, key in zip(mechanics, keys, strict=True):
        row["key"] = key
        row["id"] = f"mechanic_{sha256_text(key)[:16]}"
    return mechanics
