"""V41 controlled-language rendering and graph assembly around the frozen V39 compiler."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Sequence

from v22_relational import parse_atom, relation_atom, unary_atom
from v38_focus_parser import ontology_with_lexical_forms, render_form
from v39_compiler import compile_agent_input


SEPARATORS = {"period": ". ", "semicolon": "; ", "em_dash": " — "}
TRUE_CELLS = (("positive", "assert"), ("negative", "deny"), ("positive", "double_deny"), ("positive", "contrast_select"))
FALSE_CELLS = (("negative", "assert"), ("positive", "deny"), ("negative", "double_deny"), ("negative", "contrast_select"))
UNKNOWN_CELLS = (("positive", "unresolved"), ("negative", "unresolved"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def truth_status(values: Sequence[bool]) -> str:
    value = tuple(values)
    if value == (True,):
        return "true"
    if value == (False,):
        return "false"
    if value == (False, True):
        return "unknown"
    raise ValueError(f"Invalid epistemic values: {values}")


def fresh_operator_ontology(episode_token: str) -> tuple[dict[str, Any], dict[str, str]]:
    suffix = sha256_text(episode_token)[:8]
    roots = {
        "assert": "accepted entry", "deny": "rejected entry",
        "double_deny": "rejection reversed", "contrast_select": "selected entry",
        "unresolved": "pending entry",
    }
    cues = {operation: f"{root} {suffix}" for operation, root in roots.items()}
    ontology = {
        "operations": [{"id": operation, "cues": [cue]} for operation, cue in cues.items()],
        "grammar": {
            "roles": {"focus": "Focal report:", "operation": "Operation cue:", "context": "Context only:"},
            "productions": [
                "{focus_label}{literal}{separator}{operation_label}{cue}{separator}{context_label}{aside}.",
                "{context_label}{aside}{separator}{operation_label}{cue}{separator}{focus_label}{literal}.",
            ],
            "separators": SEPARATORS,
            "constraint": "exactly one focus, operation, and context field; one separator realization per record",
        },
    }
    return ontology, cues


def entity_aliases(entities: Sequence[dict[str, str]], token: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    mapping = {
        row["id"]: f"e{sha256_text(f'{token}|{row['id']}')[:10]}"
        for row in entities
    }
    public = [{"id": mapping[row["id"]], "entity_type": row["entity_type"]} for row in reversed(entities)]
    return public, mapping


def _predicate_spec(predicate_ontology: dict[str, Any], predicate: str, kind: str) -> dict[str, Any]:
    registry = "unary_predicates" if kind == "u" else "relations"
    return next(row for row in predicate_ontology[registry] if row["id"] == predicate)


def literal_text(
    atom: str, sign: str, orientation: str, mapping: dict[str, str],
    predicate_ontology: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    parsed = parse_atom(atom)
    spec = _predicate_spec(predicate_ontology, parsed[1], parsed[0])
    if parsed[0] == "u":
        arguments = [mapping[parsed[2]]]
        text = render_form(spec[f"{sign}_form"], arguments)
    else:
        arguments = [mapping[parsed[2]], mapping[parsed[3]]]
        text = render_form(spec[f"{orientation}_{sign}_form"], arguments)
    return text, {"predicate": parsed[1], "arguments": arguments, "lexical_sign": sign}


def render_packet(focus: str, decoy: str, cue: str, focus_order: str, punctuation: str) -> str:
    separator = SEPARATORS[punctuation]
    if focus_order == "focus_first":
        return f"Focal report: {focus}{separator}Operation cue: {cue}{separator}Context only: {decoy}."
    return f"Context only: {decoy}{separator}Operation cue: {cue}{separator}Focal report: {focus}."


def language_scene(
    scene: dict[str, Any], v32_config: dict[str, Any], episode_token: str,
    cell_counts: Counter, scene_ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    entities = scene["entities"]
    public_entities, mapping = entity_aliases(entities, f"{episode_token}|{scene['id']}")
    reverse = {value: key for key, value in mapping.items()}
    predicate_ontology = ontology_with_lexical_forms(v32_config)
    operator_ontology, cues = fresh_operator_ontology(episode_token)
    state_rows = scene["epistemic_state"]
    packets = []
    references = []
    for atom_index, state_row in enumerate(state_rows):
        atom = state_row["atom"]
        status = truth_status(state_row["allowed_values"])
        eligible = {"true": TRUE_CELLS, "false": FALSE_CELLS, "unknown": UNKNOWN_CELLS}[status]
        minimum = min(cell_counts[(status, sign, operation)] for sign, operation in eligible)
        choices = [cell for cell in eligible if cell_counts[(status, *cell)] == minimum]
        choices.sort(key=lambda cell: sha256_text(f"{episode_token}|{scene['id']}|{atom}|{cell}"))
        sign, operation = choices[0]
        cell_counts[(status, sign, operation)] += 1
        parsed = parse_atom(atom)
        orientation = "unary" if parsed[0] == "u" else ("direct" if int(sha256_text(f"orientation|{scene['id']}|{atom}")[:2], 16) % 2 == 0 else "inverse")
        focus, expected = literal_text(atom, sign, orientation, mapping, predicate_ontology)
        expected["outer_operation"] = operation
        decoy_kind = ("exact_opposite", "different_grounded_atom", "non_state_distractor")[(scene_ordinal + atom_index) % 3]
        if decoy_kind == "exact_opposite":
            decoy, _ = literal_text(atom, "negative" if sign == "positive" else "positive", orientation, mapping, predicate_ontology)
        elif decoy_kind == "different_grounded_atom":
            other = state_rows[(atom_index + 1) % len(state_rows)]["atom"]
            other_parsed = parse_atom(other)
            other_orientation = "unary" if other_parsed[0] == "u" else "direct"
            decoy, _ = literal_text(other, "positive", other_orientation, mapping, predicate_ontology)
        else:
            decoy = "an inventory note lists spare seals"
        punctuation = tuple(SEPARATORS)[(scene_ordinal + atom_index) % 3]
        focus_order = "focus_first" if (scene_ordinal + atom_index) % 2 == 0 else "focus_second"
        packet_id = f"clause_{sha256_text(f'{scene['id']}|{atom}')[:16]}"
        packets.append({
            "id": packet_id,
            "evidence_text": render_packet(focus, decoy, cues[operation], focus_order, punctuation),
        })
        references.append({
            "id": packet_id, "atom": atom, "allowed_values": list(state_row["allowed_values"]),
            "expected_parse": expected, "truth_status": status,
            "orientation": orientation, "decoy_kind": decoy_kind,
        })
    public = {
        "id": scene["id"],
        "entities": public_entities,
        "action": {"id": "inspect_pair", "binding": {key: mapping[value] for key, value in scene["action_binding"].items()}},
        "predicate_ontology": predicate_ontology,
        "operator_ontology": operator_ontology,
        "evidence_packets": list(reversed(packets)),
    }
    reference = {
        "id": scene["id"], "entity_alias_to_canonical": reverse,
        "clause_references": references,
    }
    return public, reference


def compile_language_scene(public: dict[str, Any], v32_config: dict[str, Any]) -> dict[str, Any]:
    reverse = None
    # Canonicalization is supplied separately to avoid exposing it to the compiler.
    results = []
    for packet in public["evidence_packets"]:
        agent_input = {
            "entities": public["entities"],
            "predicate_ontology": public["predicate_ontology"],
            "operator_ontology": public["operator_ontology"],
            "evidence_text": packet["evidence_text"],
        }
        results.append({"id": packet["id"], "compiler_result": compile_agent_input(agent_input)})
    return {"id": public["id"], "clauses": results}


def assemble_epistemic_graph(
    public: dict[str, Any], compiled: dict[str, Any], alias_to_canonical: dict[str, str],
    v32_config: dict[str, Any],
) -> dict[str, Any]:
    truth_table = v32_config["factorization"]["truthCompiler"]
    state: dict[str, tuple[bool, ...]] = {}
    exact_status = True
    for row in compiled["clauses"]:
        result = row["compiler_result"]
        if result.get("status") != "ok":
            exact_status = False
            continue
        parsed = result["parse"]
        try:
            arguments = [alias_to_canonical[value] for value in parsed["arguments"]]
        except KeyError:
            exact_status = False
            continue
        atom = unary_atom(parsed["predicate"], arguments[0]) if len(arguments) == 1 else relation_atom(parsed["predicate"], arguments[0], arguments[1])
        status = truth_table[parsed["outer_operation"]][parsed["lexical_sign"]]
        values = {"true": (True,), "false": (False,), "unknown": (False, True)}[status]
        if atom in state:
            exact_status = False
        state[atom] = values
    return {
        "id": public["id"], "complete": exact_status,
        "epistemic_state": [{"atom": atom, "allowed_values": list(values)} for atom, values in sorted(state.items())],
        "compiled_clauses": compiled["clauses"],
    }
