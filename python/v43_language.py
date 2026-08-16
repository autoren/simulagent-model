"""Declared state and ordered-action language interface for V43."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Sequence

from v22_relational import parse_atom, relation_atom, sha256_text, unary_atom
from v38_focus_parser import render_form
from v39_compiler import compile_agent_input, render_declared_evidence


TRUTH_TABLE = {
    "assert": {"positive": "true", "negative": "false"},
    "deny": {"positive": "false", "negative": "true"},
    "double_deny": {"positive": "true", "negative": "false"},
    "contrast_select": {"positive": "true", "negative": "false"},
    "unresolved": {"positive": "unknown", "negative": "unknown"},
}
TRUE_CELLS = (("positive", "assert"), ("negative", "deny"), ("positive", "double_deny"), ("positive", "contrast_select"))
FALSE_CELLS = (("negative", "assert"), ("positive", "deny"), ("negative", "double_deny"), ("negative", "contrast_select"))
UNKNOWN_CELLS = (("positive", "unresolved"), ("negative", "unresolved"))
SEPARATORS = {"period": ". ", "semicolon": "; ", "em_dash": " — "}
ACTION_TEMPLATES = {
    "actor_first": "{actor} performs {cue} toward {target}",
    "target_first": "{target} receives {cue} from {actor}",
}
STEP_PATTERN = re.compile(r"^Step (?P<ordinal>[1-9][0-9]*): (?P<command>.+)\.$")


def episode_aliases(record: dict[str, Any]) -> dict[str, str]:
    identifiers = {
        entity["id"]
        for section in (record["agent_input"]["support_sequences"], record["agent_input"]["queries"])
        for sequence in section
        for entity in sequence["entities"]
    }
    return {
        identifier: f"e{sha256_text(f'{record['id']}|{identifier}')[:10]}"
        for identifier in sorted(identifiers)
    }


def predicate_ontology(episode_token: str) -> dict[str, Any]:
    suffix = sha256_text(f"predicate|{episode_token}")[:8]
    unary_roots = {"active": "amber", "marked": "cobalt", "ready": "silver"}
    return {
        "entity_types": ["unit"],
        "unary_predicates": [
            {
                "id": predicate,
                "entity_type": "unit",
                "positive_form": f"{{entity}} bears {root}-signal-{suffix}",
                "negative_form": f"{{entity}} bears void-{root}-signal-{suffix}",
            }
            for predicate, root in unary_roots.items()
        ],
        "relations": [
            {
                "id": "linked",
                "source_type": "unit",
                "target_type": "unit",
                "direct_positive_form": f"{{source}} sends arc-{suffix} to {{target}}",
                "direct_negative_form": f"{{source}} sends void-arc-{suffix} to {{target}}",
                "inverse_positive_form": f"{{target}} receives arc-{suffix} from {{source}}",
                "inverse_negative_form": f"{{target}} receives void-arc-{suffix} from {{source}}",
            }
        ],
    }


def operator_ontology(episode_token: str) -> tuple[dict[str, Any], dict[str, str]]:
    suffix = sha256_text(f"operator|{episode_token}")[:8]
    roots = {
        "assert": "accepted-entry",
        "deny": "rejected-entry",
        "double_deny": "rejection-reversed",
        "contrast_select": "selected-entry",
        "unresolved": "pending-entry",
    }
    cues = {operation: f"{root}-{suffix}" for operation, root in roots.items()}
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


def action_ontology(episode_token: str) -> tuple[dict[str, Any], dict[str, str]]:
    suffix = sha256_text(f"action|{episode_token}")[:8]
    cues = {"pulse": f"pulse-token-{suffix}", "route": f"route-token-{suffix}"}
    ontology = {
        "entity_types": ["unit"],
        "actions": [
            {
                "id": action,
                "cue": cue,
                "parameters": [
                    {"id": "actor", "entity_type": "unit"},
                    {"id": "target", "entity_type": "unit"},
                ],
                "distinct_parameters": True,
            }
            for action, cue in cues.items()
        ],
        "templates": ACTION_TEMPLATES,
        "sequence_grammar": {
            "step": "Step {ordinal}: {command}.",
            "separator": "newline",
            "ordinals": "contiguous_one_based",
        },
    }
    return ontology, cues


def public_entities(entities: Sequence[dict[str, str]], aliases: dict[str, str]) -> list[dict[str, str]]:
    return [{"id": aliases[row["id"]], "entity_type": row["entity_type"]} for row in reversed(entities)]


def alias_atom(atom: str, aliases: dict[str, str]) -> str:
    parsed = parse_atom(atom)
    if parsed[0] == "u":
        return unary_atom(parsed[1], aliases[parsed[2]])
    return relation_atom(parsed[1], aliases[parsed[2]], aliases[parsed[3]])


def _status(row: dict[str, Any]) -> str:
    values = tuple(row.get("allowed_values", [row.get("value")]))
    if values == (True,):
        return "true"
    if values == (False,):
        return "false"
    if values in ((False, True), (True, False)):
        return "unknown"
    raise ValueError(f"Unsupported V43 state values: {values}")


def _predicate_spec(ontology: dict[str, Any], predicate: str, kind: str) -> dict[str, Any]:
    registry = "unary_predicates" if kind == "u" else "relations"
    return next(row for row in ontology[registry] if row["id"] == predicate)


def literal_text(
    atom: str, sign: str, orientation: str, aliases: dict[str, str], ontology: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    parsed = parse_atom(atom)
    spec = _predicate_spec(ontology, parsed[1], parsed[0])
    if parsed[0] == "u":
        arguments = [aliases[parsed[2]]]
        text = render_form(spec[f"{sign}_form"], arguments)
    else:
        arguments = [aliases[parsed[2]], aliases[parsed[3]]]
        text = render_form(spec[f"{orientation}_{sign}_form"], arguments)
    return text, {"predicate": parsed[1], "arguments": arguments, "lexical_sign": sign}


def render_state(
    state_rows: Sequence[dict[str, Any]], aliases: dict[str, str], ontology: dict[str, Any],
    cues: dict[str, str], token: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    packets, references = [], []
    rows = sorted(state_rows, key=lambda row: row["atom"])
    for index, row in enumerate(rows):
        status = _status(row)
        eligible = {"true": TRUE_CELLS, "false": FALSE_CELLS, "unknown": UNKNOWN_CELLS}[status]
        sign, operation = eligible[int(sha256_text(f"cell|{token}|{row['atom']}")[:8], 16) % len(eligible)]
        parsed = parse_atom(row["atom"])
        orientation = "unary" if parsed[0] == "u" else ("direct" if index % 2 == 0 else "inverse")
        focus, expected = literal_text(row["atom"], sign, orientation, aliases, ontology)
        expected["outer_operation"] = operation
        decoy_kind = ("exact_opposite", "different_atom", "non_state")[(index + int(sha256_text(token)[:2], 16)) % 3]
        if decoy_kind == "exact_opposite":
            decoy, _ = literal_text(row["atom"], "negative" if sign == "positive" else "positive", orientation, aliases, ontology)
        elif decoy_kind == "different_atom":
            other = rows[(index + 1) % len(rows)]["atom"]
            other_parsed = parse_atom(other)
            decoy, _ = literal_text(other, "positive", "unary" if other_parsed[0] == "u" else "direct", aliases, ontology)
        else:
            decoy = "an inventory note lists unused brackets"
        focus_order = "focus_first" if index % 2 == 0 else "focus_second"
        punctuation = tuple(SEPARATORS)[index % len(SEPARATORS)]
        packet_id = f"packet_{sha256_text(f'{token}|{row['atom']}')[:16]}"
        packets.append({
            "id": packet_id,
            "evidence_text": render_declared_evidence(focus, decoy, cues[operation], focus_order, punctuation),
        })
        references.append({
            "id": packet_id,
            "expected_parse": expected,
            "expected_atom": alias_atom(row["atom"], aliases),
            "expected_allowed_values": {"true": [True], "false": [False], "unknown": [False, True]}[status],
            "truth_status": status,
            "orientation": orientation,
            "decoy_kind": decoy_kind,
        })
    return {"evidence_packets": list(reversed(packets))}, {
        "clauses": references,
        "epistemic_state": [
            {"atom": alias_atom(row["atom"], aliases), "allowed_values": {"true": [True], "false": [False], "unknown": [False, True]}[_status(row)]}
            for row in rows
        ],
    }


def compile_state(
    state: dict[str, Any], entities: Sequence[dict[str, str]], predicate: dict[str, Any], operator: dict[str, Any],
) -> dict[str, Any]:
    clauses, assembled = [], {}
    complete = True
    for packet in state.get("evidence_packets", []):
        result = compile_agent_input({
            "entities": list(entities),
            "predicate_ontology": predicate,
            "operator_ontology": operator,
            "evidence_text": packet.get("evidence_text"),
        })
        clauses.append({"id": packet.get("id"), "compiler_result": result})
        if result.get("status") != "ok":
            complete = False
            continue
        parse = result["parse"]
        arguments = parse["arguments"]
        atom = unary_atom(parse["predicate"], arguments[0]) if len(arguments) == 1 else relation_atom(parse["predicate"], arguments[0], arguments[1])
        status = TRUTH_TABLE[parse["outer_operation"]][parse["lexical_sign"]]
        values = {"true": [True], "false": [False], "unknown": [False, True]}[status]
        if atom in assembled:
            complete = False
        assembled[atom] = values
    return {
        "status": "ok" if complete else "abstain",
        "complete": complete,
        "clauses": clauses,
        "epistemic_state": [{"atom": atom, "allowed_values": assembled[atom]} for atom in sorted(assembled)],
    }


def alias_actions(actions: Sequence[dict[str, Any]], aliases: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {"id": row["id"], "binding": {key: aliases[value] for key, value in row["binding"].items()}}
        for row in actions
    ]


def render_action_sequence(
    actions: Sequence[dict[str, Any]], aliases: dict[str, str], cues: dict[str, str], token: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = alias_actions(actions, aliases)
    lines, orientations = [], []
    for index, action in enumerate(expected, start=1):
        orientation = "actor_first" if int(sha256_text(f"orientation|{token}|{index}")[:2], 16) % 2 == 0 else "target_first"
        orientations.append(orientation)
        command = ACTION_TEMPLATES[orientation].format(cue=cues[action["id"]], **action["binding"])
        lines.append(f"Step {index}: {command}.")
    return {"action_sequence_text": "\n".join(lines)}, {"actions": expected, "orientations": orientations}


def _valid_action_ontology(ontology: dict[str, Any]) -> bool:
    if ontology.get("entity_types") != ["unit"] or ontology.get("templates") != ACTION_TEMPLATES:
        return False
    if ontology.get("sequence_grammar") != {"step": "Step {ordinal}: {command}.", "separator": "newline", "ordinals": "contiguous_one_based"}:
        return False
    actions = ontology.get("actions", [])
    return bool(actions) and all(
        row.get("id") and row.get("cue") and row.get("distinct_parameters") is True
        and row.get("parameters") == [{"id": "actor", "entity_type": "unit"}, {"id": "target", "entity_type": "unit"}]
        for row in actions
    )


def compile_action_sequence(
    payload: dict[str, Any], entities: Sequence[dict[str, str]], ontology: dict[str, Any],
) -> dict[str, Any]:
    text = payload.get("action_sequence_text")
    if not isinstance(text, str) or not text or not _valid_action_ontology(ontology):
        return {"status": "abstain", "reason": "missing_text_or_invalid_action_ontology"}
    entity_ids = [row["id"] for row in entities if row.get("entity_type") == "unit"]
    parsed_lines = []
    for line in text.splitlines():
        match = STEP_PATTERN.fullmatch(line)
        if not match:
            return {"status": "abstain", "reason": "declared_action_grammar_mismatch"}
        parsed_lines.append((int(match.group("ordinal")), match.group("command")))
    ordinals = [row[0] for row in parsed_lines]
    if ordinals != list(range(1, len(parsed_lines) + 1)):
        return {"status": "abstain", "reason": "noncontiguous_or_duplicate_step_ordinal"}
    actions = []
    for _, command in parsed_lines:
        matches = []
        for action in ontology["actions"]:
            for actor in entity_ids:
                for target in entity_ids:
                    if actor == target:
                        continue
                    for orientation, template in ontology["templates"].items():
                        rendered = template.format(actor=actor, target=target, cue=action["cue"])
                        if rendered.casefold() == command.casefold():
                            matches.append({
                                "id": action["id"],
                                "binding": {"actor": actor, "target": target},
                                "orientation": orientation,
                            })
        unique = {(row["id"], row["binding"]["actor"], row["binding"]["target"], row["orientation"]): row for row in matches}
        if len(unique) != 1:
            return {"status": "ambiguous" if unique else "abstain", "reason": "unsupported_or_ambiguous_action_command"}
        selected = next(iter(unique.values()))
        actions.append({"id": selected["id"], "binding": selected["binding"]})
    return {"status": "ok", "actions": actions}


def safety_challenges(
    entities: Sequence[dict[str, str]], aliases: dict[str, str], predicate: dict[str, Any], operator: dict[str, Any],
    operator_cues: dict[str, str], action: dict[str, Any], action_cues: dict[str, str],
) -> list[dict[str, Any]]:
    first = entities[0]["id"]
    second = entities[1]["id"]
    focus = predicate["unary_predicates"][0]["positive_form"].format(entity=first)
    valid_state = render_declared_evidence(focus, "an inventory note", operator_cues["assert"], "focus_first", "period")
    valid_command = ACTION_TEMPLATES["actor_first"].format(actor=first, target=second, cue=action_cues["pulse"])
    ambiguous = deepcopy(action)
    ambiguous["actions"][1]["cue"] = ambiguous["actions"][0]["cue"]
    return [
        {"id": "unknown_predicate_lexeme", "kind": "state", "payload": {"evidence_packets": [{"id": "s1", "evidence_text": render_declared_evidence(f"{first} bears unknown-signal", "an inventory note", operator_cues["assert"], "focus_first", "period")}]}, "expected": "abstain"},
        {"id": "unknown_state_operator_cue", "kind": "state", "payload": {"evidence_packets": [{"id": "s2", "evidence_text": valid_state.replace(operator_cues["assert"], "unknown-operation-cue")}]}, "expected": "abstain"},
        {"id": "malformed_declared_grammar", "kind": "state", "payload": {"evidence_packets": [{"id": "s3", "evidence_text": f"Maybe {focus}."}]}, "expected": "abstain"},
        {"id": "unknown_action_cue", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {valid_command.replace(action_cues['pulse'], 'unknown-action-cue')}."}, "action_ontology": action, "expected": "abstain"},
        {"id": "duplicate_step_ordinal", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {valid_command}.\nStep 1: {valid_command}."}, "action_ontology": action, "expected": "abstain"},
        {"id": "missing_step_ordinal", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {valid_command}.\nStep 3: {valid_command}."}, "action_ontology": action, "expected": "abstain"},
        {"id": "ambiguous_action_template", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {valid_command}."}, "action_ontology": ambiguous, "expected": "abstain"},
    ]


def evaluate_safety_challenge(
    challenge: dict[str, Any], entities: Sequence[dict[str, str]], predicate: dict[str, Any], operator: dict[str, Any], action: dict[str, Any],
) -> bool:
    if challenge["kind"] == "state":
        result = compile_state(challenge["payload"], entities, predicate, operator)
    else:
        result = compile_action_sequence(challenge["payload"], entities, challenge.get("action_ontology", action))
    return result.get("status") != "ok"
