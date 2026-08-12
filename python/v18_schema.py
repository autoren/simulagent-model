"""Executable Boolean transition DSL and exact V18 schema induction utilities.

V18 deliberately separates two questions:

* grounding: which Boolean values are supported by a natural-language trace; and
* dynamics: which executable program maps those values to a visible transition.

The functions in this module address only the second question.  They never inspect
V17 and never receive the target action-dependency table as model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product
from typing import Any, Iterable, Sequence


BOOLEAN_VALUES = ("inactive", "active")
COMMUTATIVE_OPERATORS = frozenset({"and", "or", "xor"})
DSL_OPERATORS = ("var", "not", "and", "or", "xor")


@dataclass(frozen=True)
class ExpressionHypothesis:
    expression: dict[str, Any]
    family: str
    depth: int
    variables: tuple[str, ...]
    signature: tuple[bool, ...]


@dataclass(frozen=True)
class ProgramHypothesis:
    program: dict[str, Any]
    component_families: tuple[str, ...]
    max_depth: int
    relevant_determinants: tuple[str, ...]
    signature: tuple[str, ...]


def variable(identifier: str) -> dict[str, Any]:
    return {"op": "var", "id": identifier}


def unary_not(argument: dict[str, Any]) -> dict[str, Any]:
    return canonical_expression({"op": "not", "arg": argument})


def binary(operator: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if operator not in COMMUTATIVE_OPERATORS:
        raise ValueError(f"Unsupported binary operator {operator}")
    return canonical_expression({"op": operator, "args": [left, right]})


def canonical_expression(expression: dict[str, Any]) -> dict[str, Any]:
    """Return a stable syntactic representative without claiming semantic uniqueness."""

    operator = expression.get("op")
    if operator == "var":
        identifier = expression.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Variable expression requires a non-empty id")
        return {"op": "var", "id": identifier}
    if operator == "not":
        argument = canonical_expression(expression.get("arg", {}))
        if argument["op"] == "not":
            return canonical_expression(argument["arg"])
        return {"op": "not", "arg": argument}
    if operator in COMMUTATIVE_OPERATORS:
        arguments = expression.get("args")
        if not isinstance(arguments, list) or len(arguments) != 2:
            raise ValueError(f"{operator} expression requires exactly two arguments")
        normalized = [canonical_expression(value) for value in arguments]
        normalized.sort(key=expression_key)
        return {"op": operator, "args": normalized}
    raise ValueError(f"Unsupported expression operator {operator}")


def expression_key(expression: dict[str, Any]) -> str:
    expression = canonical_expression(expression)
    operator = expression["op"]
    if operator == "var":
        return f"v:{expression['id']}"
    if operator == "not":
        return f"n({expression_key(expression['arg'])})"
    return f"{operator}({','.join(expression_key(value) for value in expression['args'])})"


def evaluate_expression(expression: dict[str, Any], assignment: dict[str, bool]) -> bool:
    expression = canonical_expression(expression)
    operator = expression["op"]
    if operator == "var":
        identifier = expression["id"]
        if identifier not in assignment:
            raise ValueError(f"Assignment omits determinant {identifier}")
        return bool(assignment[identifier])
    if operator == "not":
        return not evaluate_expression(expression["arg"], assignment)
    left, right = (
        evaluate_expression(value, assignment) for value in expression["args"]
    )
    if operator == "and":
        return left and right
    if operator == "or":
        return left or right
    if operator == "xor":
        return left != right
    raise AssertionError(f"Unreachable operator {operator}")


def expression_variables(expression: dict[str, Any]) -> tuple[str, ...]:
    expression = canonical_expression(expression)
    if expression["op"] == "var":
        return (expression["id"],)
    if expression["op"] == "not":
        return expression_variables(expression["arg"])
    return tuple(sorted(set().union(*(expression_variables(value) for value in expression["args"]))))


def expression_depth(expression: dict[str, Any]) -> int:
    expression = canonical_expression(expression)
    if expression["op"] == "var":
        return 0
    if expression["op"] == "not":
        return 1 + expression_depth(expression["arg"])
    return 1 + max(expression_depth(value) for value in expression["args"])


def all_assignments(determinant_ids: Sequence[str]) -> list[dict[str, bool]]:
    return [
        dict(zip(determinant_ids, values, strict=True))
        for values in product((False, True), repeat=len(determinant_ids))
    ]


def transition_code(bits: Iterable[bool]) -> str:
    return "transition_" + "".join("1" if value else "0" for value in bits)


def evaluate_program(program: dict[str, Any], assignment: dict[str, bool]) -> str:
    determinant_ids = tuple(value["id"] for value in program["determinants"])
    if set(assignment) != set(determinant_ids):
        raise ValueError("Assignment keys differ from the program determinant set")
    return transition_code(
        evaluate_expression(expression, assignment)
        for expression in program["output_bits"]
    )


def program_signature(program: dict[str, Any]) -> tuple[str, ...]:
    determinant_ids = tuple(value["id"] for value in program["determinants"])
    return tuple(evaluate_program(program, value) for value in all_assignments(determinant_ids))


def behaviorally_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_ids = tuple(value["id"] for value in left["determinants"])
    right_ids = tuple(value["id"] for value in right["determinants"])
    return left_ids == right_ids and program_signature(left) == program_signature(right)


def execute_query(program: dict[str, Any], allowed_values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    determinant_ids = tuple(value["id"] for value in program["determinants"])
    compatible = compatible_assignments(determinant_ids, allowed_values)
    possible = sorted({evaluate_program(program, assignment) for assignment in compatible})
    return {
        "compatible_assignments": len(compatible),
        "possible_transition_codes": possible,
        "identifiable": len(possible) == 1,
    }


def compatible_assignments(
    determinant_ids: Sequence[str], allowed_values: Sequence[dict[str, Any]]
) -> list[dict[str, bool]]:
    supplied: dict[str, tuple[bool, ...]] = {}
    for grounding in allowed_values:
        identifier = grounding["determinant_id"]
        if identifier not in determinant_ids:
            raise ValueError(f"Query contains unknown determinant {identifier}")
        if identifier in supplied:
            raise ValueError(f"Query repeats determinant {identifier}")
        raw = grounding["allowed_values"]
        values = tuple(value == "active" for value in BOOLEAN_VALUES if value in raw)
        if not values or any(value not in BOOLEAN_VALUES for value in raw):
            raise ValueError(f"Invalid allowed values for {identifier}")
        supplied[identifier] = values
    if set(supplied) != set(determinant_ids):
        raise ValueError("Query determinant set differs from the program determinant set")
    return [
        dict(zip(determinant_ids, values, strict=True))
        for values in product(*(supplied[identifier] for identifier in determinant_ids))
    ]


def semantic_relevant_determinants(program: dict[str, Any]) -> tuple[str, ...]:
    determinant_ids = tuple(value["id"] for value in program["determinants"])
    relevant = []
    for identifier in determinant_ids:
        others = [value for value in determinant_ids if value != identifier]
        changes = False
        for assignment in all_assignments(others):
            inactive = {**assignment, identifier: False}
            active = {**assignment, identifier: True}
            if evaluate_program(program, inactive) != evaluate_program(program, active):
                changes = True
                break
        if changes:
            relevant.append(identifier)
    return tuple(relevant)


def expression_for_family(family: str, identifiers: Sequence[str]) -> dict[str, Any]:
    values = [variable(identifier) for identifier in identifiers]
    if family == "var" and len(values) == 1:
        return values[0]
    if family == "not" and len(values) == 1:
        return unary_not(values[0])
    if family in {"and", "or", "xor"} and len(values) == 2:
        return binary(family, values[0], values[1])
    if family == "and_not" and len(values) == 2:
        return binary("and", values[0], unary_not(values[1]))
    if family == "or_of_and" and len(values) == 3:
        return binary("or", binary("and", values[0], values[1]), values[2])
    if family == "and_of_or" and len(values) == 3:
        return binary("and", binary("or", values[0], values[1]), values[2])
    if family == "xor_of_and" and len(values) == 3:
        return binary("xor", binary("and", values[0], values[1]), values[2])
    if family == "majority" and len(values) == 3:
        return binary(
            "or",
            binary("and", values[0], values[1]),
            binary(
                "or",
                binary("and", values[0], values[2]),
                binary("and", values[1], values[2]),
            ),
        )
    if family == "mux" and len(values) == 3:
        return binary(
            "or",
            binary("and", values[0], values[1]),
            binary("and", unary_not(values[0]), values[2]),
        )
    if family == "deep_xor" and len(values) == 4:
        return binary(
            "xor",
            binary("or", binary("and", values[0], values[1]), values[2]),
            values[3],
        )
    raise ValueError(f"Family {family} does not accept {len(values)} identifiers")


def build_program(
    determinant_ids: Sequence[str],
    components: Sequence[tuple[str, Sequence[str]]],
) -> dict[str, Any]:
    if not components:
        raise ValueError("A transition program needs at least one output bit")
    return {
        "dsl_version": 1,
        "determinants": [{"id": identifier, "type": "boolean"} for identifier in determinant_ids],
        "output_bits": [expression_for_family(family, values) for family, values in components],
    }


def _expression_candidates(determinant_ids: tuple[str, ...]) -> tuple[ExpressionHypothesis, ...]:
    assignments = all_assignments(determinant_ids)
    specs: list[tuple[str, tuple[str, ...]]] = []
    for identifier in determinant_ids:
        specs.extend((("var", (identifier,)), ("not", (identifier,))))
    for left, right in combinations(determinant_ids, 2):
        specs.extend((operator, (left, right)) for operator in ("and", "or", "xor"))
    specs.extend(("and_not", values) for values in permutations(determinant_ids, 2))
    for values in permutations(determinant_ids, 3):
        specs.extend((family, values) for family in (
            "or_of_and", "and_of_or", "xor_of_and", "majority", "mux"
        ))
    specs.extend(("deep_xor", values) for values in permutations(determinant_ids, 4))

    by_signature: dict[tuple[bool, ...], ExpressionHypothesis] = {}
    for family, values in specs:
        expression = expression_for_family(family, values)
        signature = tuple(evaluate_expression(expression, assignment) for assignment in assignments)
        candidate = ExpressionHypothesis(
            expression=expression,
            family=family,
            depth=expression_depth(expression),
            variables=expression_variables(expression),
            signature=signature,
        )
        existing = by_signature.get(signature)
        if existing is None or (candidate.depth, family, expression_key(expression)) < (
            existing.depth,
            existing.family,
            expression_key(existing.expression),
        ):
            by_signature[signature] = candidate
    return tuple(sorted(by_signature.values(), key=lambda value: value.signature))


@lru_cache(maxsize=16)
def enumerate_program_hypotheses(
    determinant_ids: tuple[str, ...],
    output_bits: int,
) -> tuple[ProgramHypothesis, ...]:
    """Enumerate one representative for every behavior in the bounded V18 grammar."""

    if output_bits not in (1, 2):
        raise ValueError("V18 supports one or two visible outcome bits")
    expressions = _expression_candidates(determinant_ids)
    assignments = all_assignments(determinant_ids)
    combinations_to_try = product(expressions, repeat=output_bits)
    by_signature: dict[tuple[str, ...], ProgramHypothesis] = {}
    for components in combinations_to_try:
        signature = tuple(
            transition_code(component.signature[index] for component in components)
            for index in range(len(assignments))
        )
        if signature in by_signature:
            continue
        program = {
            "dsl_version": 1,
            "determinants": [
                {"id": identifier, "type": "boolean"} for identifier in determinant_ids
            ],
            "output_bits": [value.expression for value in components],
        }
        by_signature[signature] = ProgramHypothesis(
            program=program,
            component_families=tuple(value.family for value in components),
            max_depth=max(value.depth for value in components),
            relevant_determinants=semantic_relevant_determinants(program),
            signature=signature,
        )
    return tuple(sorted(by_signature.values(), key=lambda value: value.signature))


def trace_consistent_hypotheses(
    hypotheses: Sequence[ProgramHypothesis],
    support: Sequence[dict[str, Any]],
    determinant_ids: Sequence[str],
) -> list[ProgramHypothesis]:
    assignment_index = {
        tuple(assignment[identifier] for identifier in determinant_ids): index
        for index, assignment in enumerate(all_assignments(determinant_ids))
    }
    consistent = []
    for hypothesis in hypotheses:
        matches = True
        for trace in support:
            assignment = trace["assignment"]
            key = tuple(bool(assignment[identifier]) for identifier in determinant_ids)
            if hypothesis.signature[assignment_index[key]] != trace["transition_code"]:
                matches = False
                break
        if matches:
            consistent.append(hypothesis)
    return consistent


def allowed_trace_consistent_hypotheses(
    hypotheses: Sequence[ProgramHypothesis],
    support: Sequence[dict[str, Any]],
    determinant_ids: Sequence[str],
) -> list[ProgramHypothesis]:
    """Retain programs with at least one grounded assignment explaining each trace outcome."""

    assignments = all_assignments(determinant_ids)
    index_by_assignment = {
        tuple(value[identifier] for identifier in determinant_ids): index
        for index, value in enumerate(assignments)
    }
    trace_indices = []
    for trace in support:
        indices = [
            index_by_assignment[tuple(value[identifier] for identifier in determinant_ids)]
            for value in compatible_assignments(determinant_ids, trace["allowed_values"])
        ]
        trace_indices.append((indices, trace["transition_code"]))
    return [
        hypothesis for hypothesis in hypotheses
        if all(any(hypothesis.signature[index] == outcome for index in indices) for indices, outcome in trace_indices)
    ]


def greedy_distinguishing_support(
    target: dict[str, Any],
    hypotheses: Sequence[ProgramHypothesis],
) -> list[dict[str, Any]]:
    """Choose a deterministic trace sequence until target behavior is identified."""

    determinant_ids = tuple(value["id"] for value in target["determinants"])
    target_signature = program_signature(target)
    remaining = list(hypotheses)
    unused = list(enumerate(all_assignments(determinant_ids)))
    support: list[dict[str, Any]] = []
    while len(remaining) > 1:
        best: tuple[int, int, dict[str, bool]] | None = None
        for index, assignment in unused:
            outcome = target_signature[index]
            survivors = sum(value.signature[index] == outcome for value in remaining)
            proposal = (survivors, index, assignment)
            if best is None or (proposal[0], proposal[1]) < (best[0], best[1]):
                best = proposal
        if best is None or best[0] == len(remaining):
            raise ValueError("No remaining assignment distinguishes the target program")
        _, index, assignment = best
        trace = {"assignment": assignment, "transition_code": target_signature[index]}
        support.append(trace)
        unused = [value for value in unused if value[0] != index]
        remaining = trace_consistent_hypotheses(remaining, [trace], determinant_ids)
    if not remaining or remaining[0].signature != target_signature:
        raise ValueError("Greedy support selected the wrong behavioral equivalence class")
    return support


def version_space_query(
    hypotheses: Sequence[ProgramHypothesis],
    allowed_values: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not hypotheses:
        raise ValueError("Cannot answer from an empty program version space")
    possible: set[str] = set()
    compatible_assignments = 0
    for hypothesis in hypotheses:
        answer = execute_query(hypothesis.program, allowed_values)
        possible.update(answer["possible_transition_codes"])
        compatible_assignments = answer["compatible_assignments"]
    values = sorted(possible)
    return {
        "compatible_assignments": compatible_assignments,
        "possible_transition_codes": values,
        "identifiable": len(values) == 1,
        "behavioral_hypotheses": len(hypotheses),
    }
