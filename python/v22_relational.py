"""Typed relational Boolean states and a finite lifted one-step action DSL for V22.

The module deliberately separates complete-world truth from epistemic uncertainty.  A complete
world supplies every well-typed atom.  An epistemic state also supplies every atom, but may assign
the explicit set ``(False, True)``.  Missing keys are always an error.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Iterable, Sequence


COMMUTATIVE = frozenset({"and", "or", "xor"})
BOOLEAN_OPERATORS = frozenset({"not", *COMMUTATIVE})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def unary_atom(predicate: str, entity: str) -> str:
    return f"u:{predicate}:{entity}"


def relation_atom(predicate: str, source: str, target: str) -> str:
    return f"r:{predicate}:{source}:{target}"


def parse_atom(atom: str) -> tuple[str, ...]:
    values = tuple(atom.split(":"))
    if len(values) == 3 and values[0] == "u":
        return values
    if len(values) == 4 and values[0] == "r":
        return values
    raise ValueError(f"Invalid atom key {atom}")


def entities_for_layout(layout: dict[str, int]) -> list[dict[str, str]]:
    entities = [
        {"id": f"unit_{index}", "entity_type": "unit"}
        for index in range(int(layout["units"]))
    ]
    entities.extend(
        {"id": f"hub_{index}", "entity_type": "hub"}
        for index in range(int(layout["hubs"]))
    )
    return entities


def layout_key(entities: Sequence[dict[str, str]]) -> tuple[int, int]:
    return (
        sum(value["entity_type"] == "unit" for value in entities),
        sum(value["entity_type"] == "hub" for value in entities),
    )


def entity_types(entities: Sequence[dict[str, str]]) -> dict[str, str]:
    result = {value["id"]: value["entity_type"] for value in entities}
    if len(result) != len(entities):
        raise ValueError("Entity identifiers must be unique")
    return result


def atom_universe(config: dict[str, Any], entities: Sequence[dict[str, str]]) -> tuple[str, ...]:
    types = entity_types(entities)
    atoms = []
    for predicate in config["unaryPredicates"]:
        atoms.extend(
            unary_atom(predicate["id"], identifier)
            for identifier, entity_type in types.items()
            if entity_type == predicate["entityType"]
        )
    for relation in config["relations"]:
        for source, source_type in types.items():
            if source_type != relation["sourceType"]:
                continue
            for target, target_type in types.items():
                if target_type != relation["targetType"]:
                    continue
                if not relation["allowSelf"] and source == target:
                    continue
                atoms.append(relation_atom(relation["id"], source, target))
    return tuple(sorted(atoms))


def action_bindings(config: dict[str, Any], entities: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    types = entity_types(entities)
    parameters = config["action"]["parameters"]
    choices = [
        sorted(identifier for identifier, entity_type in types.items() if entity_type == parameter["entityType"])
        for parameter in parameters
    ]
    bindings = []
    for values in product(*choices):
        if config["action"]["distinctParameters"] and len(set(values)) != len(values):
            continue
        bindings.append({
            parameter["id"]: value
            for parameter, value in zip(parameters, values, strict=True)
        })
    return bindings


def validate_binding(
    config: dict[str, Any], entities: Sequence[dict[str, str]], binding: dict[str, str]
) -> None:
    types = entity_types(entities)
    parameters = config["action"]["parameters"]
    if set(binding) != {value["id"] for value in parameters}:
        raise ValueError("Action binding parameters differ from the schema")
    for parameter in parameters:
        entity = binding[parameter["id"]]
        if types.get(entity) != parameter["entityType"]:
            raise ValueError(f"Ill-typed binding for {parameter['id']}")
    if config["action"]["distinctParameters"] and len(set(binding.values())) != len(binding):
        raise ValueError("Action parameters must bind distinct entities")


def validate_complete_world(
    config: dict[str, Any], entities: Sequence[dict[str, str]], world: dict[str, bool]
) -> None:
    universe = set(atom_universe(config, entities))
    if set(world) != universe:
        missing = sorted(universe - set(world))
        extra = sorted(set(world) - universe)
        raise ValueError(f"Complete world atom mismatch; missing={missing}, extra={extra}")
    if any(type(value) is not bool for value in world.values()):
        raise ValueError("Complete world values must be Boolean")


def validate_epistemic_state(
    config: dict[str, Any], entities: Sequence[dict[str, str]], state: dict[str, tuple[bool, ...]]
) -> None:
    universe = set(atom_universe(config, entities))
    if set(state) != universe:
        missing = sorted(universe - set(state))
        extra = sorted(set(state) - universe)
        raise ValueError(f"Epistemic atom mismatch; missing={missing}, extra={extra}")
    for atom, values in state.items():
        if values not in ((False,), (True,), (False, True)):
            raise ValueError(f"Invalid epistemic values for {atom}: {values}")


def epistemic_from_world(
    config: dict[str, Any], entities: Sequence[dict[str, str]], world: dict[str, bool],
    unknown_atoms: Iterable[str] = (),
) -> dict[str, tuple[bool, ...]]:
    validate_complete_world(config, entities, world)
    unknown = set(unknown_atoms)
    if not unknown <= set(world):
        raise ValueError("Unknown-atom request includes an atom outside the world")
    return {
        atom: (False, True) if atom in unknown else (value,)
        for atom, value in world.items()
    }


def epistemic_rows(state: dict[str, tuple[bool, ...]]) -> list[dict[str, Any]]:
    return [
        {"atom": atom, "allowed_values": list(values)}
        for atom, values in sorted(state.items())
    ]


def rows_to_epistemic(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[bool, ...]]:
    result: dict[str, tuple[bool, ...]] = {}
    for row in rows:
        atom = row["atom"]
        if atom in result:
            raise ValueError(f"Repeated epistemic atom {atom}")
        result[atom] = tuple(row["allowed_values"])
    return result


def canonical_expression(expression: dict[str, Any], bound: dict[str, str] | None = None) -> dict[str, Any]:
    bound = {} if bound is None else dict(bound)
    operator = expression.get("op")
    if operator == "unary":
        predicate = expression.get("predicate")
        variable = expression.get("var")
        if not isinstance(predicate, str) or not isinstance(variable, str):
            raise ValueError("Unary atom requires predicate and variable")
        return {"op": "unary", "predicate": predicate, "var": bound.get(variable, variable)}
    if operator == "relation":
        predicate = expression.get("predicate")
        source = expression.get("source")
        target = expression.get("target")
        if not all(isinstance(value, str) for value in (predicate, source, target)):
            raise ValueError("Relation atom requires predicate, source, and target")
        return {
            "op": "relation", "predicate": predicate,
            "source": bound.get(source, source), "target": bound.get(target, target),
        }
    if operator == "not":
        return {"op": "not", "arg": canonical_expression(expression.get("arg", {}), bound)}
    if operator in COMMUTATIVE:
        arguments = expression.get("args")
        if not isinstance(arguments, list) or len(arguments) != 2:
            raise ValueError(f"{operator} requires exactly two arguments")
        normalized = [canonical_expression(value, bound) for value in arguments]
        normalized.sort(key=expression_key)
        return {"op": operator, "args": normalized}
    if operator == "exists":
        variable = expression.get("var")
        variable_type = expression.get("entity_type")
        if not isinstance(variable, str) or not variable or variable in {"actor", "target"}:
            raise ValueError("Existential variable must be local and non-empty")
        if not isinstance(variable_type, str) or not variable_type:
            raise ValueError("Existential variable requires an entity type")
        canonical_variable = f"$q{len(bound)}"
        nested = {**bound, variable: canonical_variable}
        distinct = sorted({bound.get(value, value) for value in expression.get("distinct_from", [])})
        return {
            "op": "exists", "var": canonical_variable, "entity_type": variable_type,
            "distinct_from": distinct,
            "where": canonical_expression(expression.get("where", {}), nested),
        }
    raise ValueError(f"Unsupported relational expression operator {operator}")


def expression_key(expression: dict[str, Any]) -> str:
    return canonical_json(canonical_expression(expression))


def expression_quantifier_depth(expression: dict[str, Any]) -> int:
    expression = canonical_expression(expression)
    operator = expression["op"]
    if operator in {"unary", "relation"}:
        return 0
    if operator == "not":
        return expression_quantifier_depth(expression["arg"])
    if operator in COMMUTATIVE:
        return max(expression_quantifier_depth(value) for value in expression["args"])
    return 1 + expression_quantifier_depth(expression["where"])


def canonical_program(program: dict[str, Any]) -> dict[str, Any]:
    bits = program.get("output_bits")
    if not isinstance(bits, list) or not 1 <= len(bits) <= 2:
        raise ValueError("Relational program requires one or two output bits")
    return {
        "dsl_version": 1,
        "action": "inspect_pair",
        "parameters": [
            {"id": "actor", "entity_type": "unit"},
            {"id": "target", "entity_type": "unit"},
        ],
        "output_bits": [canonical_expression(value) for value in bits],
    }


def program_key(program: dict[str, Any]) -> str:
    return canonical_json(canonical_program(program))


def resolve_variable(variable: str, binding: dict[str, str], local: dict[str, str]) -> str:
    if variable in local:
        return local[variable]
    if variable in binding:
        return binding[variable]
    raise ValueError(f"Unbound relational variable {variable}")


def evaluate_expression(
    expression: dict[str, Any], config: dict[str, Any], entities: Sequence[dict[str, str]],
    world: dict[str, bool], binding: dict[str, str], local: dict[str, str] | None = None,
) -> bool:
    expression = canonical_expression(expression)
    local = {} if local is None else dict(local)
    operator = expression["op"]
    if operator == "unary":
        return world[unary_atom(
            expression["predicate"], resolve_variable(expression["var"], binding, local)
        )]
    if operator == "relation":
        return world[relation_atom(
            expression["predicate"],
            resolve_variable(expression["source"], binding, local),
            resolve_variable(expression["target"], binding, local),
        )]
    if operator == "not":
        return not evaluate_expression(expression["arg"], config, entities, world, binding, local)
    if operator in COMMUTATIVE:
        left = evaluate_expression(expression["args"][0], config, entities, world, binding, local)
        right = evaluate_expression(expression["args"][1], config, entities, world, binding, local)
        if operator == "and":
            return left and right
        if operator == "or":
            return left or right
        return left != right
    if operator == "exists":
        types = entity_types(entities)
        excluded = {
            resolve_variable(value, binding, local) for value in expression["distinct_from"]
        }
        for identifier, entity_type in sorted(types.items()):
            if entity_type != expression["entity_type"] or identifier in excluded:
                continue
            if evaluate_expression(
                expression["where"], config, entities, world, binding,
                {**local, expression["var"]: identifier},
            ):
                return True
        return False
    raise AssertionError(f"Unreachable operator {operator}")


def evaluate_program(
    program: dict[str, Any], config: dict[str, Any], entities: Sequence[dict[str, str]],
    world: dict[str, bool], binding: dict[str, str],
) -> str:
    validate_complete_world(config, entities, world)
    validate_binding(config, entities, binding)
    program = canonical_program(program)
    bits = [
        evaluate_expression(value, config, entities, world, binding)
        for value in program["output_bits"]
    ]
    return "transition_" + "".join("1" if value else "0" for value in bits)


def expression_atoms(
    expression: dict[str, Any], config: dict[str, Any], entities: Sequence[dict[str, str]],
    binding: dict[str, str], local: dict[str, str] | None = None,
) -> set[str]:
    expression = canonical_expression(expression)
    local = {} if local is None else dict(local)
    operator = expression["op"]
    if operator == "unary":
        return {unary_atom(
            expression["predicate"], resolve_variable(expression["var"], binding, local)
        )}
    if operator == "relation":
        return {relation_atom(
            expression["predicate"],
            resolve_variable(expression["source"], binding, local),
            resolve_variable(expression["target"], binding, local),
        )}
    if operator == "not":
        return expression_atoms(expression["arg"], config, entities, binding, local)
    if operator in COMMUTATIVE:
        return set().union(*(
            expression_atoms(value, config, entities, binding, local)
            for value in expression["args"]
        ))
    types = entity_types(entities)
    excluded = {resolve_variable(value, binding, local) for value in expression["distinct_from"]}
    result: set[str] = set()
    for identifier, entity_type in sorted(types.items()):
        if entity_type == expression["entity_type"] and identifier not in excluded:
            result.update(expression_atoms(
                expression["where"], config, entities, binding,
                {**local, expression["var"]: identifier},
            ))
    return result


def compatible_worlds(
    config: dict[str, Any], entities: Sequence[dict[str, str]],
    state: dict[str, tuple[bool, ...]], maximum_unknown: int | None = None,
) -> Iterable[dict[str, bool]]:
    validate_epistemic_state(config, entities, state)
    unknown = [atom for atom, values in state.items() if len(values) == 2]
    if maximum_unknown is not None and len(unknown) > maximum_unknown:
        raise ValueError(f"Epistemic state has {len(unknown)} unknown atoms; limit is {maximum_unknown}")
    fixed = {atom: values[0] for atom, values in state.items() if len(values) == 1}
    for values in product((False, True), repeat=len(unknown)):
        yield {**fixed, **dict(zip(unknown, values, strict=True))}


def execute_partial(
    programs: Sequence[dict[str, Any]], config: dict[str, Any],
    entities: Sequence[dict[str, str]], state: dict[str, tuple[bool, ...]],
    binding: dict[str, str], maximum_unknown: int | None = None,
) -> dict[str, Any]:
    possible = sorted({
        evaluate_program(program, config, entities, world, binding)
        for program in programs
        for world in compatible_worlds(config, entities, state, maximum_unknown)
    })
    return {"possible_transition_codes": possible, "identifiable": len(possible) == 1}


def _unary(predicate: str, variable: str) -> dict[str, Any]:
    return {"op": "unary", "predicate": predicate, "var": variable}


def _relation(predicate: str, source: str, target: str) -> dict[str, Any]:
    return {"op": "relation", "predicate": predicate, "source": source, "target": target}


def _not(argument: dict[str, Any]) -> dict[str, Any]:
    return {"op": "not", "arg": argument}


def _binary(operator: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": operator, "args": [left, right]}


def _exists(
    variable: str, entity_type: str, where: dict[str, Any], distinct_from: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "op": "exists", "var": variable, "entity_type": entity_type,
        "distinct_from": list(distinct_from), "where": where,
    }


@dataclass(frozen=True)
class ExpressionHypothesis:
    family: str
    name: str
    expression: dict[str, Any]
    key: str


@dataclass(frozen=True)
class ProgramHypothesis:
    program: dict[str, Any]
    component_families: tuple[str, ...]
    component_names: tuple[str, ...]
    key: str


def expression_catalog() -> tuple[ExpressionHypothesis, ...]:
    path = lambda start, middle, end: _binary(
        "and", _relation("linked", start, middle), _relation("linked", middle, end)
    )
    values: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "unary_selection": [
            ("actor_stable", _unary("stable", "actor")),
            ("actor_charged", _unary("charged", "actor")),
            ("target_stable", _unary("stable", "target")),
            ("target_charged", _unary("charged", "target")),
            ("actor_stable_and_target_charged", _binary(
                "and", _unary("stable", "actor"), _unary("charged", "target")
            )),
            ("actor_charged_xor_target_stable", _binary(
                "xor", _unary("charged", "actor"), _unary("stable", "target")
            )),
        ],
        "relation_conditioned": [
            ("actor_links_target", _relation("linked", "actor", "target")),
            ("target_links_actor", _relation("linked", "target", "actor")),
            ("actor_links_stable_target", _binary(
                "and", _relation("linked", "actor", "target"), _unary("stable", "target")
            )),
            ("target_links_charged_actor", _binary(
                "and", _relation("linked", "target", "actor"), _unary("charged", "actor")
            )),
            ("one_way_xor", _binary(
                "xor", _relation("linked", "actor", "target"),
                _relation("linked", "target", "actor")
            )),
            ("strict_actor_to_target", _binary(
                "and", _relation("linked", "actor", "target"),
                _not(_relation("linked", "target", "actor"))
            )),
        ],
        "two_hop_composition": [
            ("actor_via_mid_to_target", _exists(
                "mid", "unit", path("actor", "mid", "target"), ("actor", "target")
            )),
            ("target_via_mid_to_actor", _exists(
                "mid", "unit", path("target", "mid", "actor"), ("actor", "target")
            )),
            ("common_parent", _exists(
                "mid", "unit", _binary(
                    "and", _relation("linked", "mid", "actor"),
                    _relation("linked", "mid", "target")
                ), ("actor", "target")
            )),
            ("common_child", _exists(
                "mid", "unit", _binary(
                    "and", _relation("linked", "actor", "mid"),
                    _relation("linked", "target", "mid")
                ), ("actor", "target")
            )),
            ("stable_intermediate_path", _exists(
                "mid", "unit", _binary(
                    "and", path("actor", "mid", "target"), _unary("stable", "mid")
                ), ("actor", "target")
            )),
            ("charged_intermediate_reverse_path", _exists(
                "mid", "unit", _binary(
                    "and", path("target", "mid", "actor"), _unary("charged", "mid")
                ), ("actor", "target")
            )),
        ],
        "existential_aggregation": [
            ("stable_out_neighbor", _exists(
                "member", "unit", _binary(
                    "and", _relation("linked", "actor", "member"), _unary("stable", "member")
                ), ("actor",)
            )),
            ("charged_in_neighbor", _exists(
                "member", "unit", _binary(
                    "and", _relation("linked", "member", "target"), _unary("charged", "member")
                ), ("target",)
            )),
            ("online_hub_feeds_actor", _exists(
                "member", "hub", _binary(
                    "and", _relation("feeds", "member", "actor"), _unary("online", "member")
                )
            )),
            ("offline_hub_feeds_target", _exists(
                "member", "hub", _binary(
                    "and", _relation("feeds", "member", "target"),
                    _not(_unary("online", "member"))
                )
            )),
            ("stable_in_neighbor", _exists(
                "member", "unit", _binary(
                    "and", _relation("linked", "member", "actor"), _unary("stable", "member")
                ), ("actor",)
            )),
            ("charged_out_neighbor", _exists(
                "member", "unit", _binary(
                    "and", _relation("linked", "target", "member"), _unary("charged", "member")
                ), ("target",)
            )),
        ],
    }
    result = []
    for family, entries in values.items():
        for name, expression in entries:
            canonical = canonical_expression(expression)
            result.append(ExpressionHypothesis(family, name, canonical, expression_key(canonical)))
    return tuple(result)


def enumerate_program_hypotheses(output_bits: int) -> tuple[ProgramHypothesis, ...]:
    catalog = expression_catalog()
    if output_bits == 1:
        combinations = ((value,) for value in catalog)
    elif output_bits == 2:
        combinations = (
            (left, right) for left in catalog for right in catalog if left.key != right.key
        )
    else:
        raise ValueError("V22 supports one or two visible outcome bits")
    result = []
    for components in combinations:
        program = canonical_program({"output_bits": [value.expression for value in components]})
        result.append(ProgramHypothesis(
            program=program,
            component_families=tuple(value.family for value in components),
            component_names=tuple(value.name for value in components),
            key=program_key(program),
        ))
    return tuple(result)


def target_hypotheses(family: str, output_bits: int) -> tuple[ProgramHypothesis, ...]:
    return tuple(
        value for value in enumerate_program_hypotheses(output_bits)
        if all(component == family for component in value.component_families)
    )


def find_expression_counterexample(
    left: dict[str, Any], right: dict[str, Any], config: dict[str, Any],
    layouts: Sequence[dict[str, int]], maximum_atoms: int = 16,
) -> dict[str, Any] | None:
    left = canonical_expression(left)
    right = canonical_expression(right)
    for layout in layouts:
        entities = entities_for_layout(layout)
        universe = atom_universe(config, entities)
        for binding in action_bindings(config, entities):
            relevant = sorted(
                expression_atoms(left, config, entities, binding)
                | expression_atoms(right, config, entities, binding)
            )
            if len(relevant) > maximum_atoms:
                raise RuntimeError(
                    f"Equivalence check needs {len(relevant)} atoms; declared limit is {maximum_atoms}"
                )
            for values in product((False, True), repeat=len(relevant)):
                world = {atom: False for atom in universe}
                world.update(dict(zip(relevant, values, strict=True)))
                if evaluate_expression(left, config, entities, world, binding) != evaluate_expression(
                    right, config, entities, world, binding
                ):
                    return {
                        "entities": entities,
                        "binding": binding,
                        "world": world,
                        "layout": dict(layout),
                        "relevant_atoms": len(relevant),
                    }
    return None


def find_program_counterexample(
    left: ProgramHypothesis, right: ProgramHypothesis, config: dict[str, Any],
    layouts: Sequence[dict[str, int]], cache: dict[tuple[str, str, str], dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    cache = {} if cache is None else cache
    layout_identity = canonical_json(layouts)
    for left_bit, right_bit in zip(
        left.program["output_bits"], right.program["output_bits"], strict=True
    ):
        if expression_key(left_bit) == expression_key(right_bit):
            continue
        pair = tuple(sorted((expression_key(left_bit), expression_key(right_bit))))
        key = (pair[0], pair[1], layout_identity)
        if key not in cache:
            cache[key] = find_expression_counterexample(
                left_bit, right_bit, config, layouts,
                config["limits"]["maximumTruthTableAtomsPerEquivalenceCheck"],
            )
        if cache[key] is not None:
            return cache[key]
    return None


def trace_consistent_hypotheses(
    hypotheses: Sequence[ProgramHypothesis], traces: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> list[ProgramHypothesis]:
    return [
        hypothesis for hypothesis in hypotheses
        if all(
            evaluate_program(
                hypothesis.program, config, trace["entities"], trace["world"], trace["binding"]
            ) == trace["transition_code"]
            for trace in traces
        )
    ]


def greedy_identifying_support(
    target: ProgramHypothesis, hypotheses: Sequence[ProgramHypothesis], config: dict[str, Any],
    layouts: Sequence[dict[str, int]], maximum_traces: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}
    witnesses: dict[str, dict[str, Any]] = {}
    support_equivalent = []
    for hypothesis in hypotheses:
        if hypothesis.key == target.key:
            continue
        witness = find_program_counterexample(target, hypothesis, config, layouts, cache)
        if witness is None:
            support_equivalent.append(hypothesis.key)
            continue
        identity = canonical_state_hash(
            config, witness["entities"], witness["world"], witness["binding"]
        )
        witnesses.setdefault(identity, witness)
    if support_equivalent:
        raise RuntimeError(
            f"Target is not identifiable below four entities; equivalent candidates={len(support_equivalent)}"
        )
    candidates = list(hypotheses)
    support = []
    unused = list(witnesses.values())
    while len(candidates) > 1:
        best = None
        best_remaining = None
        for witness in unused:
            target_code = evaluate_program(
                target.program, config, witness["entities"], witness["world"], witness["binding"]
            )
            remaining = [
                hypothesis for hypothesis in candidates
                if evaluate_program(
                    hypothesis.program, config, witness["entities"], witness["world"], witness["binding"]
                ) == target_code
            ]
            if best_remaining is None or len(remaining) < len(best_remaining):
                best = witness
                best_remaining = remaining
        if best is None or best_remaining is None or len(best_remaining) == len(candidates):
            raise RuntimeError("Counterexample pool cannot identify the target")
        trace = {
            **best,
            "transition_code": evaluate_program(
                target.program, config, best["entities"], best["world"], best["binding"]
            ),
        }
        support.append(trace)
        candidates = best_remaining
        unused.remove(best)
        if len(support) > maximum_traces:
            raise RuntimeError(f"Target needs more than {maximum_traces} identifying traces")
    if candidates[0].key != target.key:
        raise RuntimeError("Support uniquely selected the wrong target")
    return support, {
        "initial_hypotheses": len(hypotheses),
        "counterexample_worlds": len(witnesses),
        "support_traces": len(support),
        "remaining_hypotheses": 1,
        "equivalence_cache_entries": len(cache),
    }


def hashed_world(
    config: dict[str, Any], entities: Sequence[dict[str, str]], token: str,
) -> dict[str, bool]:
    return {
        atom: int(sha256_text(f"{token}|{atom}")[-2:], 16) % 2 == 1
        for atom in atom_universe(config, entities)
    }


def rename_atom(atom: str, mapping: dict[str, str]) -> str:
    values = parse_atom(atom)
    if values[0] == "u":
        return unary_atom(values[1], mapping[values[2]])
    return relation_atom(values[1], mapping[values[2]], mapping[values[3]])


def rename_state(
    entities: Sequence[dict[str, str]], world: dict[str, bool], binding: dict[str, str],
    mapping: dict[str, str],
) -> tuple[list[dict[str, str]], dict[str, bool], dict[str, str]]:
    if set(mapping) != {value["id"] for value in entities} or len(set(mapping.values())) != len(mapping):
        raise ValueError("Entity renaming must be a bijection")
    renamed_entities = [
        {"id": mapping[value["id"]], "entity_type": value["entity_type"]}
        for value in entities
    ]
    renamed_world = {rename_atom(atom, mapping): value for atom, value in world.items()}
    renamed_binding = {parameter: mapping[entity] for parameter, entity in binding.items()}
    return renamed_entities, renamed_world, renamed_binding


def canonical_state_hash(
    config: dict[str, Any], entities: Sequence[dict[str, str]], world: dict[str, bool],
    binding: dict[str, str] | None = None,
) -> str:
    validate_complete_world(config, entities, world)
    types = entity_types(entities)
    by_type = {
        entity_type: sorted(identifier for identifier, value in types.items() if value == entity_type)
        for entity_type in sorted(set(types.values()))
    }
    permutation_groups = [list(permutations(values)) for values in by_type.values()]
    representations = []
    for selected in product(*permutation_groups):
        mapping: dict[str, str] = {}
        for (entity_type, _), ordering in zip(by_type.items(), selected, strict=True):
            mapping.update({
                identifier: f"{entity_type}_{index}"
                for index, identifier in enumerate(ordering)
            })
        renamed = sorted((rename_atom(atom, mapping), value) for atom, value in world.items())
        payload: dict[str, Any] = {
            "layout": sorted((key, len(value)) for key, value in by_type.items()),
            "world": renamed,
        }
        if binding is not None:
            payload["binding"] = sorted((key, mapping[value]) for key, value in binding.items())
        representations.append(canonical_json(payload))
    return sha256_text(min(representations))


def extend_with_inert_entity(
    config: dict[str, Any], entities: Sequence[dict[str, str]], world: dict[str, bool],
    entity_type: str = "unit",
) -> tuple[list[dict[str, str]], dict[str, bool], str]:
    types = entity_types(entities)
    index = 0
    while f"{entity_type}_{index}" in types:
        index += 1
    identifier = f"{entity_type}_{index}"
    extended_entities = [*entities, {"id": identifier, "entity_type": entity_type}]
    extended_world = {
        atom: world.get(atom, False) for atom in atom_universe(config, extended_entities)
    }
    return extended_entities, extended_world, identifier


def _predicate_maps(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    return (
        {value["id"]: value for value in config["unaryPredicates"]},
        {value["id"]: value for value in config["relations"]},
    )


def _literal_phrase(
    atom: str, truth: bool, orientation: str, config: dict[str, Any]
) -> str:
    unary, relations = _predicate_maps(config)
    values = parse_atom(atom)
    if values[0] == "u":
        predicate = unary[values[1]]
        label = predicate["trueLabel"] if truth else predicate["falseLabel"]
        return f"{values[2]} is {label}"
    relation = relations[values[1]]
    source, target = values[2], values[3]
    if relation["id"] == "linked":
        if orientation == "inverse":
            return (
                f"{target} receives a link from {source}" if truth
                else f"{target} does not receive a link from {source}"
            )
        return f"{source} is linked to {target}" if truth else f"{source} is not linked to {target}"
    if orientation == "inverse":
        return f"{target} is fed by {source}" if truth else f"{target} is not fed by {source}"
    return f"{source} feeds {target}" if truth else f"{source} does not feed {target}"


def render_observation(
    config: dict[str, Any], entities: Sequence[dict[str, str]],
    state: dict[str, tuple[bool, ...]], token: str,
) -> tuple[str, list[dict[str, Any]]]:
    validate_epistemic_state(config, entities, state)
    declarations = ", ".join(
        f"{value['id']} ({value['entity_type']})" for value in entities
    )
    clauses = [f"Entities: {declarations}."]
    signatures = []
    operators = ("affirmative_gold", "negated_opposite", "contrastive_both")
    for index, (atom, allowed) in enumerate(sorted(state.items())):
        values = parse_atom(atom)
        orientation = "inverse" if values[0] == "r" and (index + int(token[-1], 16)) % 2 else "direct"
        if len(allowed) == 2:
            positive = _literal_phrase(atom, True, orientation, config)
            clause = f"Current evidence leaves unresolved whether {positive}."
            operator = "explicit_unknown"
        else:
            truth = allowed[0]
            operator = operators[(index + int(token[0], 16)) % len(operators)]
            actual = _literal_phrase(atom, truth, orientation, config)
            opposite = _literal_phrase(atom, not truth, orientation, config)
            if operator == "affirmative_gold":
                clause = f"The current inspection establishes that {actual}."
            elif operator == "negated_opposite":
                clause = f"It is not true that {opposite}."
            else:
                clause = f"{opposite.capitalize()} is not the case; instead, {actual}."
        clauses.append(clause)
        signatures.append({
            "atom": atom,
            "predicate_kind": "unary" if values[0] == "u" else "relation",
            "predicate": values[1],
            "arguments": list(values[2:]),
            "allowed_values": list(allowed),
            "semantic_operator": operator,
            "relation_orientation": orientation if values[0] == "r" else None,
        })
    return "\n".join(clauses), signatures
