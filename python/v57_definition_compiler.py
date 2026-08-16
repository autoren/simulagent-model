"""Controlled definition-conditioned compiler for V57."""
from __future__ import annotations

import json
import re
from typing import Any

from v39_compiler import OPERATOR_CUES


DEFINITION_CLAUSES = {
    "identity": re.compile(
        r"^It defines opaque symbol (?P<opaque_id>[a-z0-9_]+) as kind "
        r"(?P<kind>unary_predicate|binary_relation|bound_action)$"
    ),
    "signature": re.compile(r"^The typed roles are (?P<payload>\{.*\})$"),
    "forms": re.compile(r"^The lexical forms are (?P<payload>\{.*\})$"),
}


def render_controlled_definition(
    opaque_id: str,
    kind: str,
    typed_signature: dict[str, str],
    lexical_forms: dict[str, str],
    template_family: str,
) -> str:
    clauses = {
        "identity": f"It defines opaque symbol {opaque_id} as kind {kind}",
        "signature": "The typed roles are " + json.dumps(
            typed_signature, sort_keys=True, separators=(",", ":")
        ),
        "forms": "The lexical forms are " + json.dumps(
            lexical_forms, sort_keys=True, separators=(",", ":")
        ),
    }
    orders = {
        "signature_first": ("signature", "identity", "forms"),
        "meaning_first": ("identity", "forms", "signature"),
        "example_first": ("forms", "signature", "identity"),
    }
    if template_family not in orders:
        raise ValueError("unknown V57 definition template family")
    return ". ".join(clauses[key] for key in orders[template_family]) + "."


def parse_controlled_definition(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.endswith("."):
        return None
    sentences = text[:-1].split(". ")
    if len(sentences) != 3:
        return None
    parsed: dict[str, Any] = {}
    for sentence in sentences:
        matches = [
            (name, pattern.fullmatch(sentence))
            for name, pattern in DEFINITION_CLAUSES.items()
        ]
        matches = [(name, match) for name, match in matches if match]
        if len(matches) != 1:
            return None
        name, match = matches[0]
        if name in parsed:
            return None
        if name == "identity":
            parsed.update(match.groupdict())
        else:
            try:
                payload = json.loads(match.group("payload"))
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in payload.items()
            ):
                return None
            parsed[name] = payload
    return parsed if set(parsed) == {"opaque_id", "kind", "signature", "forms"} else None


def _required_signature(kind: str) -> set[str]:
    return {
        "unary_predicate": {"entity"},
        "binary_relation": {"source", "target"},
        "bound_action": {"actor", "target"},
    }[kind]


def _required_forms(kind: str) -> set[str]:
    return {
        "unary_predicate": {"positive", "negative"},
        "binary_relation": {
            "direct_positive", "direct_negative",
            "inverse_positive", "inverse_negative",
        },
        "bound_action": {"command"},
    }[kind]


def validate_concept(
    row: dict[str, Any], mutation: str | None = None
) -> dict[str, Any] | None:
    required = {
        "opaque_id", "kind", "typed_signature", "controlled_definition",
        "positive_or_command_form", "lexical_forms", "definition_template_family",
    }
    if set(row) != required:
        return None
    parsed = parse_controlled_definition(row["controlled_definition"])
    if parsed is None and mutation != "ignore_definition_body":
        return None
    if parsed is None:
        parsed = {
            "opaque_id": row["opaque_id"], "kind": row["kind"],
            "signature": row["typed_signature"], "forms": row["lexical_forms"],
        }
    kind = row["kind"]
    if kind not in {"unary_predicate", "binary_relation", "bound_action"}:
        return None
    expected_signature = _required_signature(kind)
    if mutation != "ignore_typed_signature" and (
        set(row["typed_signature"]) != expected_signature
        or set(parsed["signature"]) != expected_signature
    ):
        return None
    if set(row["lexical_forms"]) != _required_forms(kind):
        return None
    if kind != "bound_action" and mutation == "drop_negative_form":
        row = {**row, "lexical_forms": {
            key: value for key, value in row["lexical_forms"].items()
            if "negative" not in key
        }}
    consistent = (
        parsed["opaque_id"] == row["opaque_id"]
        and parsed["kind"] == kind
        and (
            mutation == "ignore_typed_signature"
            or parsed["signature"] == row["typed_signature"]
        )
        and (
            mutation == "ignore_definition_body"
            or parsed["forms"] == row["lexical_forms"]
        )
        and row["positive_or_command_form"]
        == row["lexical_forms"][
            "command" if kind == "bound_action" else (
                "positive" if kind == "unary_predicate" else "direct_positive"
            )
        ]
    )
    return row if consistent else None


def _entity_types(agent_input: dict[str, Any]) -> dict[str, str] | None:
    rows = agent_input.get("entities")
    if not isinstance(rows, list):
        return None
    result = {}
    for row in rows:
        if set(row) != {"id", "entity_type"} or row["id"] in result:
            return None
        result[row["id"]] = row["entity_type"]
    return result


def _format_pattern_regex(pattern: str, roles: list[str]) -> re.Pattern:
    escaped = re.escape(pattern)
    for role in roles:
        escaped = escaped.replace(re.escape("{" + role + "}"), rf"(?P<{role}>[a-z0-9_]+)")
    return re.compile("^" + escaped + "$")


def _predicate_focus(evidence: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"Focal report: (?P<focus>.+); Operation cue: (?P<cue>.+); "
        r"Context only: (?P<context>.+)\.",
        evidence,
    )
    if not match:
        return None
    cue_map = {
        cue.casefold(): operation
        for operation, cues in OPERATOR_CUES.items() for cue in cues
    }
    operation = cue_map.get(match.group("cue").casefold())
    return (match.group("focus"), operation) if operation else None


def _match_concept(
    concept: dict[str, Any], text: str, entity_types: dict[str, str],
    mutation: str | None,
) -> list[dict[str, Any]]:
    kind = concept["kind"]
    signature = concept["typed_signature"]
    forms = concept["lexical_forms"]
    rows = []
    if kind == "unary_predicate":
        candidates = (("positive", "positive"), ("negative", "negative"))
        roles = ["entity"]
    elif kind == "binary_relation":
        candidates = (
            ("direct_positive", "positive"),
            ("direct_negative", "negative"),
            ("inverse_positive", "positive"),
            ("inverse_negative", "negative"),
        )
        roles = ["source", "target"]
    else:
        candidates = (("command", None),)
        roles = ["actor", "target"]
    for form_key, sign in candidates:
        if form_key not in forms:
            continue
        match = _format_pattern_regex(forms[form_key], roles).fullmatch(text)
        if not match:
            continue
        binding = match.groupdict()
        if mutation != "ignore_typed_signature" and any(
            entity_types.get(binding[role]) != signature[role] for role in roles
        ):
            continue
        if len(set(binding.values())) != len(binding):
            continue
        if kind == "unary_predicate":
            arguments = [binding["entity"]]
        else:
            arguments = [binding[roles[0]], binding[roles[1]]]
            if mutation == "swap_relation_roles" and kind == "binary_relation":
                arguments.reverse()
        output_kind = (
            "binary_relation" if mutation == "treat_action_as_relation"
            and kind == "bound_action" else kind
        )
        rows.append({
            "kind": output_kind,
            "symbol": concept["opaque_id"],
            "arguments": arguments,
            "lexical_sign": sign,
            "orientation": (
                form_key.split("_", 1)[0] if kind == "binary_relation" else None
            ),
        })
    return rows


def compile_agent_input(
    agent_input: dict[str, Any], mutation: str | None = None
) -> dict[str, Any]:
    entity_types = _entity_types(agent_input)
    definitions = agent_input.get("concept_definitions")
    evidence = agent_input.get("evidence_text")
    if entity_types is None or not isinstance(definitions, list) or not definitions:
        return {"status": "abstain", "reason": "missing_or_invalid_schema"}
    if not isinstance(evidence, str):
        return {"status": "abstain", "reason": "missing_evidence"}
    concepts = []
    for row in definitions:
        validated = validate_concept(dict(row), mutation)
        if validated is None:
            return {"status": "abstain", "reason": "invalid_definition"}
        concepts.append(validated)
    if mutation != "accept_duplicate_lexeme":
        forms = [
            value for concept in concepts
            for value in concept["lexical_forms"].values()
        ]
        if len(forms) != len(set(value.casefold() for value in forms)):
            return {"status": "ambiguous", "reason": "duplicate_lexical_form"}

    predicate = _predicate_focus(evidence)
    if predicate:
        focus, operation = predicate
        matches = [
            match for concept in concepts
            if concept["kind"] != "bound_action"
            for match in _match_concept(concept, focus, entity_types, mutation)
        ]
        for match in matches:
            match["outer_operation"] = operation
    else:
        action = re.fullmatch(r"Action request: (?P<command>.+)\.", evidence)
        if not action:
            return {"status": "abstain", "reason": "unsupported_evidence_grammar"}
        matches = [
            match for concept in concepts if concept["kind"] == "bound_action"
            for match in _match_concept(
                concept, action.group("command"), entity_types, mutation
            )
        ]
        for match in matches:
            match["outer_operation"] = None
    if len(matches) > 1 and mutation == "accept_duplicate_lexeme":
        return {"status": "ok", "parse": matches[0]}
    if len(matches) != 1:
        return {
            "status": "ambiguous" if len(matches) > 1 else "abstain",
            "reason": "non_unique_grounding",
        }
    return {"status": "ok", "parse": matches[0]}


def compiled_truth(parse: dict[str, Any]) -> str | None:
    if parse["kind"] == "bound_action":
        return None
    table = {
        "assert": {"positive": "true", "negative": "false"},
        "deny": {"positive": "false", "negative": "true"},
        "double_deny": {"positive": "true", "negative": "false"},
        "contrast_select": {"positive": "true", "negative": "false"},
        "unresolved": {"positive": "unknown", "negative": "unknown"},
    }
    return table[parse["outer_operation"]][parse["lexical_sign"]]
