"""Declared state and ordered-action language interface for V45."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Sequence

from v22_relational import sha256_text
from v43_language import (
    ACTION_TEMPLATES,
    compile_state,
    episode_aliases,
    evaluate_safety_challenge as evaluate_v43_safety,
    operator_ontology,
    predicate_ontology,
    public_entities,
    render_state,
)


STEP_PATTERN = re.compile(r"^Step (?P<ordinal>[1-9][0-9]*): (?P<command>.+)\.$")
WAIT_TEMPLATE = "advance time with {cue}"


def action_ontology(episode_token: str) -> tuple[dict[str, Any], dict[str, str]]:
    suffix = sha256_text(f"v45-action|{episode_token}")[:8]
    cues = {
        "pulse": f"pulse-token-{suffix}",
        "route": f"route-token-{suffix}",
        "wait": f"wait-token-{suffix}",
    }
    ontology = {
        "entity_types": ["unit"],
        "actions": [
            {
                "id": action,
                "cue": cues[action],
                "parameters": [
                    {"id": "actor", "entity_type": "unit"},
                    {"id": "target", "entity_type": "unit"},
                ],
                "distinct_parameters": True,
            }
            for action in ("pulse", "route")
        ] + [{"id": "wait", "cue": cues["wait"], "parameters": [], "distinct_parameters": False}],
        "templates": {**ACTION_TEMPLATES, "wait": WAIT_TEMPLATE},
        "sequence_grammar": {
            "step": "Step {ordinal}: {command}.",
            "separator": "newline",
            "ordinals": "contiguous_one_based",
        },
    }
    return ontology, cues


def alias_actions(actions: Sequence[dict[str, Any]], aliases: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "id": action["id"],
            "binding": {key: aliases[value] for key, value in action.get("binding", {}).items()},
        }
        for action in actions
    ]


def render_action_sequence(
    actions: Sequence[dict[str, Any]], aliases: dict[str, str], cues: dict[str, str], token: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = alias_actions(actions, aliases)
    lines, orientations, kinds = [], [], []
    for index, action in enumerate(expected, start=1):
        if action["id"] == "wait":
            orientation = "wait"
            command = WAIT_TEMPLATE.format(cue=cues["wait"])
            kind = "wait"
        else:
            orientation = "actor_first" if int(sha256_text(f"v45-orientation|{token}|{index}")[:2], 16) % 2 == 0 else "target_first"
            command = ACTION_TEMPLATES[orientation].format(cue=cues[action["id"]], **action["binding"])
            kind = "bound"
        orientations.append(orientation)
        kinds.append(kind)
        lines.append(f"Step {index}: {command}.")
    return {"action_sequence_text": "\n".join(lines)}, {
        "actions": expected,
        "orientations": orientations,
        "command_kinds": kinds,
    }


def _valid_action_ontology(ontology: dict[str, Any]) -> bool:
    if ontology.get("entity_types") != ["unit"]:
        return False
    if ontology.get("templates") != {**ACTION_TEMPLATES, "wait": WAIT_TEMPLATE}:
        return False
    if ontology.get("sequence_grammar") != {
        "step": "Step {ordinal}: {command}.", "separator": "newline", "ordinals": "contiguous_one_based",
    }:
        return False
    rows = ontology.get("actions", [])
    if {row.get("id") for row in rows} != {"pulse", "route", "wait"}:
        return False
    by_id = {row["id"]: row for row in rows}
    bound_parameters = [{"id": "actor", "entity_type": "unit"}, {"id": "target", "entity_type": "unit"}]
    return (
        all(by_id[action].get("cue") and by_id[action].get("parameters") == bound_parameters and by_id[action].get("distinct_parameters") is True for action in ("pulse", "route"))
        and bool(by_id["wait"].get("cue"))
        and by_id["wait"].get("parameters") == []
        and by_id["wait"].get("distinct_parameters") is False
    )


def compile_action_sequence(
    payload: dict[str, Any], entities: Sequence[dict[str, str]], ontology: dict[str, Any],
) -> dict[str, Any]:
    text = payload.get("action_sequence_text")
    if not isinstance(text, str) or not text or not _valid_action_ontology(ontology):
        return {"status": "abstain", "reason": "missing_text_or_invalid_action_ontology"}
    parsed = []
    for line in text.splitlines():
        match = STEP_PATTERN.fullmatch(line)
        if not match:
            return {"status": "abstain", "reason": "declared_action_grammar_mismatch"}
        parsed.append((int(match.group("ordinal")), match.group("command")))
    if [row[0] for row in parsed] != list(range(1, len(parsed) + 1)):
        return {"status": "abstain", "reason": "noncontiguous_or_duplicate_step_ordinal"}
    entity_ids = [row["id"] for row in entities if row.get("entity_type") == "unit"]
    actions, kinds = [], []
    for _, command in parsed:
        matches = []
        for action in ontology["actions"]:
            if action["id"] == "wait":
                if WAIT_TEMPLATE.format(cue=action["cue"]).casefold() == command.casefold():
                    matches.append({"id": "wait", "binding": {}, "kind": "wait", "orientation": "wait"})
                continue
            for actor in entity_ids:
                for target in entity_ids:
                    if actor == target:
                        continue
                    for orientation, template in ACTION_TEMPLATES.items():
                        if template.format(actor=actor, target=target, cue=action["cue"]).casefold() == command.casefold():
                            matches.append({
                                "id": action["id"],
                                "binding": {"actor": actor, "target": target},
                                "kind": "bound",
                                "orientation": orientation,
                            })
        unique = {
            (row["id"], tuple(sorted(row["binding"].items())), row["orientation"]): row
            for row in matches
        }
        if len(unique) != 1:
            return {"status": "ambiguous" if unique else "abstain", "reason": "unsupported_or_ambiguous_action_command"}
        selected = next(iter(unique.values()))
        actions.append({"id": selected["id"], "binding": selected["binding"]})
        kinds.append(selected["kind"])
    return {"status": "ok", "actions": actions, "command_kinds": kinds}


def safety_challenges(
    entities: Sequence[dict[str, str]], predicate: dict[str, Any], operator: dict[str, Any],
    operator_cues: dict[str, str], action: dict[str, Any], action_cues: dict[str, str],
) -> list[dict[str, Any]]:
    first, second = entities[0]["id"], entities[1]["id"]
    focus = predicate["unary_predicates"][0]["positive_form"].format(entity=first)
    from v39_compiler import render_declared_evidence
    valid_state = render_declared_evidence(focus, "an inventory note", operator_cues["assert"], "focus_first", "period")
    valid_bound = ACTION_TEMPLATES["actor_first"].format(actor=first, target=second, cue=action_cues["pulse"])
    valid_wait = WAIT_TEMPLATE.format(cue=action_cues["wait"])
    ambiguous = deepcopy(action)
    next(row for row in ambiguous["actions"] if row["id"] == "route")["cue"] = action_cues["pulse"]
    return [
        {"id": "unknown_predicate_lexeme", "kind": "state", "payload": {"evidence_packets": [{"id": "s1", "evidence_text": render_declared_evidence(f"{first} bears unknown-signal", "an inventory note", operator_cues["assert"], "focus_first", "period")}]}, "expected": "abstain"},
        {"id": "unknown_state_operator_cue", "kind": "state", "payload": {"evidence_packets": [{"id": "s2", "evidence_text": valid_state.replace(operator_cues["assert"], "unknown-operation-cue")}]}, "expected": "abstain"},
        {"id": "malformed_declared_grammar", "kind": "state", "payload": {"evidence_packets": [{"id": "s3", "evidence_text": f"Maybe {focus}."}]}, "expected": "abstain"},
        {"id": "unknown_bound_action_cue", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {valid_bound.replace(action_cues['pulse'], 'unknown-action-cue')}."}, "action_ontology": action, "expected": "abstain"},
        {"id": "unknown_wait_cue", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {valid_wait.replace(action_cues['wait'], 'unknown-wait-cue')}."}, "action_ontology": action, "expected": "abstain"},
        {"id": "wait_with_arguments", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {first} performs {action_cues['wait']} toward {second}."}, "action_ontology": action, "expected": "abstain"},
        {"id": "duplicate_step_ordinal", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {valid_wait}.\nStep 1: {valid_bound}."}, "action_ontology": action, "expected": "abstain"},
        {"id": "missing_step_ordinal", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {valid_wait}.\nStep 3: {valid_bound}."}, "action_ontology": action, "expected": "abstain"},
        {"id": "ambiguous_action_template", "kind": "action", "payload": {"action_sequence_text": f"Step 1: {valid_bound}."}, "action_ontology": ambiguous, "expected": "abstain"},
    ]


def evaluate_safety_challenge(
    challenge: dict[str, Any], entities: Sequence[dict[str, str]], predicate: dict[str, Any],
    operator: dict[str, Any], action: dict[str, Any],
) -> bool:
    if challenge["kind"] == "state":
        return evaluate_v43_safety(challenge, entities, predicate, operator, action)
    result = compile_action_sequence(challenge["payload"], entities, challenge.get("action_ontology", action))
    return result.get("status") != "ok"
