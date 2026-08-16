"""V39 exact compiler for the declared controlled evidence language."""

from __future__ import annotations

import re
from typing import Any

from v38_focus_parser import extract_literal_candidates


OPERATOR_CUES: dict[str, tuple[str, str]] = {
    "assert": ("endorsement", "support"),
    "deny": ("rejection", "contradiction"),
    "double_deny": ("rejection-overturning", "denial-blocking"),
    "contrast_select": ("selection", "preference"),
    "unresolved": ("undecided", "withheld-verdict"),
}

SEPARATORS = {"period": ". ", "semicolon": "; ", "em_dash": " — "}
FOCUS_LABEL = "Focal report: "
OPERATION_LABEL = "Operation cue: "
CONTEXT_LABEL = "Context only: "


def declared_operator_ontology() -> dict[str, Any]:
    """Return the complete operator lexicon and grammar exposed to the agent."""
    return {
        "operations": [
            {"id": operation, "cues": list(cues)}
            for operation, cues in OPERATOR_CUES.items()
        ],
        "grammar": {
            "roles": {
                "focus": FOCUS_LABEL.strip(),
                "operation": OPERATION_LABEL.strip(),
                "context": CONTEXT_LABEL.strip(),
            },
            "productions": [
                "{focus_label}{literal}{separator}{operation_label}{cue}{separator}{context_label}{aside}.",
                "{context_label}{aside}{separator}{operation_label}{cue}{separator}{focus_label}{literal}.",
            ],
            "separators": SEPARATORS,
            "constraint": "exactly one focus, operation, and context field; one separator realization per record",
        },
    }


def render_declared_evidence(
    focus: str, decoy: str, cue: str, focus_order: str, punctuation: str,
) -> str:
    separator = SEPARATORS[punctuation]
    if focus_order == "focus_first":
        fields = (f"{FOCUS_LABEL}{focus}", f"{OPERATION_LABEL}{cue}", f"{CONTEXT_LABEL}{decoy}")
    elif focus_order == "focus_second":
        fields = (f"{CONTEXT_LABEL}{decoy}", f"{OPERATION_LABEL}{cue}", f"{FOCUS_LABEL}{focus}")
    else:
        raise ValueError(f"Unsupported V39 focus order: {focus_order}")
    return separator.join(fields) + "."


def _cue_map(operator_ontology: dict[str, Any]) -> dict[str, str] | None:
    mapping: dict[str, str] = {}
    for row in operator_ontology.get("operations", []):
        operation = row.get("id")
        for cue in row.get("cues", []):
            normalized = str(cue).casefold()
            if normalized in mapping or not normalized:
                return None
            mapping[normalized] = str(operation)
    return mapping


def _grammar_matches(evidence: str, separators: dict[str, str]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for punctuation, separator in separators.items():
        escaped = re.escape(separator)
        patterns = (
            (
                "focus_first",
                rf"^{re.escape(FOCUS_LABEL)}(?P<focus>.+?){escaped}{re.escape(OPERATION_LABEL)}(?P<cue>.+?){escaped}{re.escape(CONTEXT_LABEL)}(?P<decoy>.+)\.$",
            ),
            (
                "focus_second",
                rf"^{re.escape(CONTEXT_LABEL)}(?P<decoy>.+?){escaped}{re.escape(OPERATION_LABEL)}(?P<cue>.+?){escaped}{re.escape(FOCUS_LABEL)}(?P<focus>.+)\.$",
            ),
        )
        for focus_order, pattern in patterns:
            match = re.fullmatch(pattern, evidence)
            if match:
                matches.append({**match.groupdict(), "focus_order": focus_order, "punctuation": punctuation})
    return matches


def _ground_exact_literal(agent_input: dict[str, Any], text: str):
    probe = {
        "agent_input": {
            "entities": agent_input.get("entities", []),
            "predicate_ontology": agent_input.get("predicate_ontology", {}),
            "evidence_text": text,
        }
    }
    return [
        candidate for candidate in extract_literal_candidates(probe)
        if candidate.start == 0 and candidate.end == len(text)
    ]


def compile_agent_input(agent_input: dict[str, Any]) -> dict[str, Any]:
    """Compile from exposed fields only; unsupported or ambiguous inputs fail closed."""
    evidence = agent_input.get("evidence_text")
    if not isinstance(evidence, str):
        return {"status": "abstain", "reason": "missing_evidence_text"}
    if evidence.count(FOCUS_LABEL) != 1:
        status = "ambiguous" if evidence.count(FOCUS_LABEL) > 1 else "abstain"
        return {"status": status, "reason": "focus_field_cardinality"}
    if evidence.count(OPERATION_LABEL) != 1 or evidence.count(CONTEXT_LABEL) != 1:
        return {"status": "abstain", "reason": "declared_field_cardinality"}
    operator_ontology = agent_input.get("operator_ontology")
    if not isinstance(operator_ontology, dict):
        return {"status": "abstain", "reason": "missing_operator_ontology"}
    cue_map = _cue_map(operator_ontology)
    if cue_map is None:
        return {"status": "abstain", "reason": "invalid_operator_ontology"}
    grammar = operator_ontology.get("grammar", {})
    separators = grammar.get("separators")
    if not isinstance(separators, dict) or separators != SEPARATORS:
        return {"status": "abstain", "reason": "unsupported_grammar_registry"}
    matches = _grammar_matches(evidence, separators)
    if len(matches) != 1:
        return {"status": "abstain", "reason": "declared_grammar_mismatch"}
    match = matches[0]
    operation = cue_map.get(match["cue"].casefold())
    if operation is None:
        return {"status": "abstain", "reason": "unknown_operator_cue"}
    candidates = _ground_exact_literal(agent_input, match["focus"])
    if len(candidates) != 1:
        status = "ambiguous" if len(candidates) > 1 else "abstain"
        return {"status": status, "reason": "unsupported_or_ambiguous_predicate_lexeme"}
    candidate = candidates[0]
    return {
        "status": "ok",
        "parse": {
            "predicate": candidate.predicate,
            "arguments": list(candidate.arguments),
            "lexical_sign": candidate.sign,
            "outer_operation": operation,
        },
        "diagnostics": {
            "source_orientation": candidate.orientation,
            "focus_order": match["focus_order"],
            "punctuation": match["punctuation"],
        },
    }
