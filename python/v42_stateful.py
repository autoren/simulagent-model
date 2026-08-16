"""Typed deterministic sequential-state DSL and exact executor for V42."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Sequence

from v22_relational import (
    canonical_expression,
    canonical_json,
    evaluate_expression,
    expression_key,
    parse_atom,
    relation_atom,
    sha256_text,
    unary_atom,
)


ONTOLOGY: dict[str, Any] = {
    "entityTypes": ["unit"],
    "unaryPredicates": [
        {"id": "active", "entityType": "unit"},
        {"id": "marked", "entityType": "unit"},
        {"id": "ready", "entityType": "unit"},
    ],
    "relations": [
        {"id": "linked", "sourceType": "unit", "targetType": "unit", "allowSelf": False},
    ],
    "action": {
        "id": "sequential_interaction",
        "parameters": [
            {"id": "actor", "entityType": "unit"},
            {"id": "target", "entityType": "unit"},
        ],
        "distinctParameters": True,
    },
}

ACTIONS = ("pulse", "route")
MUTATION_OPS = ("set_true", "set_false", "toggle")


def entities(count: int) -> list[dict[str, str]]:
    if count < 2:
        raise ValueError("V42 requires at least two entities")
    return [{"id": f"unit_{index}", "entity_type": "unit"} for index in range(count)]


def atom_universe(values: Sequence[dict[str, str]]) -> tuple[str, ...]:
    identifiers = [row["id"] for row in values]
    atoms = [unary_atom(predicate, identifier) for predicate in ("active", "marked", "ready") for identifier in identifiers]
    atoms.extend(
        relation_atom("linked", source, target)
        for source in identifiers for target in identifiers if source != target
    )
    return tuple(sorted(atoms))


def deterministic_world(values: Sequence[dict[str, str]], token: str) -> dict[str, bool]:
    return {
        atom: int(sha256_text(f"v42|{token}|{atom}")[:8], 16) % 2 == 1
        for atom in atom_universe(values)
    }


def action_bindings(values: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    identifiers = sorted(row["id"] for row in values)
    return [
        {"actor": actor, "target": target}
        for actor in identifiers for target in identifiers if actor != target
    ]


def unary(predicate: str, variable: str) -> dict[str, str]:
    return {"op": "unary", "predicate": predicate, "var": variable}


def relation(source: str, target: str) -> dict[str, str]:
    return {"op": "relation", "predicate": "linked", "source": source, "target": target}


def negate(argument: dict[str, Any]) -> dict[str, Any]:
    return {"op": "not", "arg": argument}


def effect(operation: str, target: dict[str, Any], source: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"op": operation, "target": target}
    if source is not None:
        result["source"] = source
    return result


def conditional(condition: dict[str, Any], nested: dict[str, Any]) -> dict[str, Any]:
    return {"op": "conditional_effect", "condition": condition, "effect": nested}


def _canonical_target(target: dict[str, Any]) -> dict[str, Any]:
    normalized = canonical_expression(target)
    if normalized["op"] not in {"unary", "relation"}:
        raise ValueError("V42 effect targets must be atoms")
    variables = {normalized.get("var"), normalized.get("source"), normalized.get("target")} - {None}
    if not variables <= {"actor", "target"}:
        raise ValueError("V42 effects may target action parameters only")
    return normalized


def canonical_effect(value: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    operation = value.get("op")
    if operation in MUTATION_OPS:
        return {"op": operation, "target": _canonical_target(value.get("target", {}))}
    if operation == "copy":
        source = canonical_expression(value.get("source", {}))
        if source["op"] not in {"unary", "relation"}:
            raise ValueError("V42 copy source must be an atom")
        return {"op": "copy", "target": _canonical_target(value.get("target", {})), "source": source}
    if operation == "conditional_effect":
        if depth >= 1:
            raise ValueError("V42 conditional effects cannot nest")
        return {
            "op": "conditional_effect",
            "condition": canonical_expression(value.get("condition", {})),
            "effect": canonical_effect(value.get("effect", {}), depth + 1),
        }
    raise ValueError(f"Unsupported V42 effect operation: {operation}")


def effect_key(value: dict[str, Any]) -> str:
    return canonical_json(canonical_effect(value))


def canonical_program(program: dict[str, Any]) -> dict[str, Any]:
    rules = []
    seen = set()
    for rule in program.get("rules", []):
        action = rule.get("action")
        if action not in ACTIONS or action in seen:
            raise ValueError("V42 requires at most one rule for each registered action")
        seen.add(action)
        effects = [canonical_effect(value) for value in rule.get("effects", [])]
        if not 1 <= len(effects) <= 2:
            raise ValueError("V42 actions require one or two effects")
        effects.sort(key=effect_key)
        rules.append({"action": action, "effects": effects})
    if seen != set(ACTIONS):
        raise ValueError("V42 programs require pulse and route rules")
    rules.sort(key=lambda row: row["action"])
    return {"dsl_version": 2, "rules": rules}


def program_key(program: dict[str, Any]) -> str:
    return canonical_json(canonical_program(program))


def _resolve_atom(expression: dict[str, Any], binding: dict[str, str]) -> str:
    normalized = canonical_expression(expression)
    if normalized["op"] == "unary":
        return unary_atom(normalized["predicate"], binding[normalized["var"]])
    if normalized["op"] == "relation":
        return relation_atom(
            normalized["predicate"], binding[normalized["source"]], binding[normalized["target"]]
        )
    raise ValueError("V42 atom resolver received a non-atom expression")


def _effect_assignment(
    value: dict[str, Any], world: dict[str, bool], binding: dict[str, str],
    entity_rows: Sequence[dict[str, str]],
) -> tuple[str, bool] | None:
    normalized = canonical_effect(value)
    if normalized["op"] == "conditional_effect":
        if not evaluate_expression(normalized["condition"], ONTOLOGY, entity_rows, world, binding):
            return None
        return _effect_assignment(normalized["effect"], world, binding, entity_rows)
    target = _resolve_atom(normalized["target"], binding)
    if normalized["op"] == "set_true":
        return target, True
    if normalized["op"] == "set_false":
        return target, False
    if normalized["op"] == "toggle":
        return target, not world[target]
    source = _resolve_atom(normalized["source"], binding)
    return target, world[source]


def apply_action(
    program: dict[str, Any], entity_rows: Sequence[dict[str, str]], world: dict[str, bool],
    action: dict[str, Any],
) -> dict[str, bool]:
    if set(world) != set(atom_universe(entity_rows)) or any(type(value) is not bool for value in world.values()):
        raise ValueError("V42 action requires a complete typed Boolean world")
    action_id = action.get("id")
    if action_id not in ACTIONS:
        raise ValueError(f"Unknown V42 action: {action_id}")
    binding = action.get("binding", {})
    if set(binding) != {"actor", "target"} or binding["actor"] == binding["target"]:
        raise ValueError("V42 action binding must contain two distinct parameters")
    identifiers = {row["id"] for row in entity_rows}
    if not set(binding.values()) <= identifiers:
        raise ValueError("V42 action binding references an unknown entity")
    rule = next(row for row in canonical_program(program)["rules"] if row["action"] == action_id)
    assignments: dict[str, bool] = {}
    for value in rule["effects"]:
        assignment = _effect_assignment(value, world, binding, entity_rows)
        if assignment is None:
            continue
        atom, truth = assignment
        if atom in assignments and assignments[atom] != truth:
            raise ValueError("V42 simultaneous effects conflict")
        assignments[atom] = truth
    return {**world, **assignments}


def execute_sequence(
    program: dict[str, Any], entity_rows: Sequence[dict[str, str]], world: dict[str, bool],
    actions: Sequence[dict[str, Any]],
) -> list[dict[str, bool]]:
    current = dict(world)
    trajectory = []
    for action in actions:
        current = apply_action(program, entity_rows, current, action)
        trajectory.append(current)
    if not trajectory:
        raise ValueError("V42 sequences must contain at least one action")
    return trajectory


def memoryless_execute_sequence(
    program: dict[str, Any], entity_rows: Sequence[dict[str, str]], world: dict[str, bool],
    actions: Sequence[dict[str, Any]],
) -> list[dict[str, bool]]:
    """Oracle program with the V22-style no-persistent-state limitation."""
    return [apply_action(program, entity_rows, world, action) for action in actions]


def world_signature(world: dict[str, bool]) -> str:
    return canonical_json([{"atom": atom, "value": value} for atom, value in sorted(world.items())])


def epistemic_rows(world: dict[str, bool], unknown_atoms: Iterable[str] = ()) -> list[dict[str, Any]]:
    unknown = set(unknown_atoms)
    if not unknown <= set(world):
        raise ValueError("V42 unknown atom lies outside the world")
    return [
        {"atom": atom, "allowed_values": [False, True] if atom in unknown else [value]}
        for atom, value in sorted(world.items())
    ]


def compatible_worlds(rows: Sequence[dict[str, Any]]) -> list[dict[str, bool]]:
    atoms = [row["atom"] for row in rows]
    if len(atoms) != len(set(atoms)):
        raise ValueError("V42 epistemic state repeats an atom")
    choices = []
    for row in rows:
        values = tuple(row["allowed_values"])
        if values not in ((False,), (True,), (False, True)):
            raise ValueError("Invalid V42 epistemic value set")
        choices.append(values)
    return [dict(zip(atoms, values, strict=True)) for values in product(*choices)]


def execute_partial(
    programs: Sequence[dict[str, Any]], entity_rows: Sequence[dict[str, str]],
    initial_state: Sequence[dict[str, Any]], actions: Sequence[dict[str, Any]],
    memoryless: bool = False,
) -> dict[str, Any]:
    if not programs:
        return {"possible_step_states": [], "possible_final_observations": [], "identifiable": False}
    step_sets = [set() for _ in actions]
    executor = memoryless_execute_sequence if memoryless else execute_sequence
    for world in compatible_worlds(initial_state):
        for program in programs:
            trajectory = executor(program, entity_rows, world, actions)
            for index, state in enumerate(trajectory):
                step_sets[index].add(world_signature(state))
    steps = [sorted(values) for values in step_sets]
    final = steps[-1] if steps else []
    return {
        "possible_step_states": steps,
        "possible_final_observations": final,
        "identifiable": len(final) == 1,
    }


def _rule(action: str, effects: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"action": action, "effects": list(effects)}


def mechanic_registry() -> list[dict[str, Any]]:
    mechanics: list[dict[str, Any]] = []

    for index, (pulse_op, route_op) in enumerate(product(MUTATION_OPS, repeat=2)):
        program = {
            "rules": [
                _rule("pulse", [effect(pulse_op, unary("active", "actor"))]),
                _rule("route", [effect(route_op, unary("marked", "target"))]),
            ]
        }
        mechanics.append({"family": "set_clear_toggle", "ordinal": index, "program": canonical_program(program)})
    mechanics.append({
        "family": "set_clear_toggle", "ordinal": 9,
        "program": canonical_program({"rules": [
            _rule("pulse", [effect("toggle", relation("actor", "target"))]),
            _rule("route", [effect("toggle", unary("ready", "target"))]),
        ]}),
    })

    conditions = (
        unary("active", "actor"), negate(unary("active", "actor")),
        unary("marked", "target"), relation("actor", "target"),
        {"op": "exists", "var": "witness", "entity_type": "unit", "distinct_from": ["actor", "target"], "where": unary("active", "witness")},
    )
    for index, (condition_value, result_op) in enumerate(product(conditions, ("set_true", "toggle"))):
        pulse_effect = (
            effect("toggle", unary("active", "actor"))
            if result_op == "set_true" else effect("set_true", unary("marked", "actor"))
        )
        program = {"rules": [
            _rule("pulse", [pulse_effect]),
            _rule("route", [conditional(condition_value, effect(result_op, unary("ready", "target")))]),
        ]}
        mechanics.append({"family": "state_conditional_effect", "ordinal": index, "program": canonical_program(program)})

    propagation_pairs = (
        ("active", "marked"), ("active", "ready"), ("marked", "active"),
        ("marked", "ready"), ("ready", "active"),
    )
    for index, (direction, pair) in enumerate(product(("direct", "reverse"), propagation_pairs)):
        source_predicate, target_predicate = pair
        edge = relation("actor", "target") if direction == "direct" else relation("target", "actor")
        program = {"rules": [
            _rule("pulse", [effect("toggle", edge)]),
            _rule("route", [conditional(
                edge,
                effect("copy", unary(target_predicate, "target"), unary(source_predicate, "actor")),
            )]),
        ]}
        mechanics.append({"family": "directed_relational_propagation", "ordinal": index, "program": canonical_program(program)})

    order_variants = list(product(MUTATION_OPS, (False, True), ("set_true", "toggle")))[:10]
    for index, (trigger_op, negate_condition, result_op) in enumerate(order_variants):
        condition_value = unary("active", "actor")
        if negate_condition:
            condition_value = negate(condition_value)
        program = {"rules": [
            _rule("pulse", [effect(trigger_op, unary("active", "actor"))]),
            _rule("route", [conditional(
                condition_value, effect(result_op, unary("marked", "target"))
            )]),
        ]}
        mechanics.append({"family": "order_sensitive_composition", "ordinal": index, "program": canonical_program(program)})

    keys = [program_key(row["program"]) for row in mechanics]
    if len(mechanics) != 40 or len(set(keys)) != 40:
        raise RuntimeError("V42 mechanic registry must contain 40 unique programs")
    for row, key in zip(mechanics, keys, strict=True):
        row["key"] = key
        row["id"] = f"mechanic_{sha256_text(key)[:16]}"
    return mechanics
