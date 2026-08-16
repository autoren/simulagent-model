"""V38 controlled discourse grammar, ontology matcher, and focus interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence


SURFACE_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "fit": {
        "assert": {
            "focus_a": "The state entry affirms {focus}. The separate comparison {decoy} is not the embedded report.",
            "focus_b": "Set aside the comparison {decoy}; the observation records {focus}.",
        },
        "deny": {
            "focus_a": "Observation rejects the embedded report {focus}. The comparison {decoy} is incidental.",
            "focus_b": "Apart from the incidental comparison {decoy}, the proposition refuted here is {focus}.",
        },
        "double_deny": {
            "focus_a": "The attempted rejection of {focus} is overturned. The comparison {decoy} is incidental.",
            "focus_b": "Ignore the incidental comparison {decoy}; evidence blocks rejecting {focus}.",
        },
        "contrast_select": {
            "focus_a": "Observation favors {focus}; {decoy} is the alternative it sets aside.",
            "focus_b": "The alternative {decoy} is discarded, while {focus} is selected.",
        },
        "unresolved": {
            "focus_a": "The report {focus} remains undecided. The comparison {decoy} has no bearing on that status.",
            "focus_b": "Although {decoy} appears as a comparison, no verdict is available for {focus}.",
        },
    },
    "validation": {
        "assert": {
            "focus_a": "Record {focus} as supported; the aside {decoy} does not supply the proposition at issue.",
            "focus_b": "The aside {decoy} is not at issue. What observation endorses is {focus}.",
        },
        "deny": {
            "focus_a": "Mark the focal description {focus} as contradicted; {decoy} is merely an aside.",
            "focus_b": "The aside {decoy} is not being judged. Evidence rules against {focus}.",
        },
        "double_deny": {
            "focus_a": "A dismissal of the focal claim {focus} would fail; {decoy} is only an aside.",
            "focus_b": "Leaving the aside {decoy} out of account, evidence overturns a rejection of {focus}.",
        },
        "contrast_select": {
            "focus_a": "Keep the focal alternative {focus} and put aside {decoy}.",
            "focus_b": "Put aside {decoy}; the focal member retained by the comparison is {focus}.",
        },
        "unresolved": {
            "focus_a": "No decision can be made about the focal report {focus}; {decoy} is an unrelated aside.",
            "focus_b": "The aside {decoy} settles nothing. The focal report {focus} stays open.",
        },
    },
}

NON_STATE_DECOYS = {
    "focus_first": "a storage note lists spare brackets",
    "focus_second": "a timetable records routine inspections",
}


@dataclass(frozen=True)
class LiteralCandidate:
    predicate: str
    arguments: tuple[str, ...]
    sign: str
    orientation: str
    text: str
    start: int
    end: int


def ontology_with_lexical_forms(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_types": config["ontology"]["entityTypes"],
        "unary_predicates": [
            {
                "id": row["id"], "entity_type": row["entityType"],
                "positive_form": row["trueForm"], "negative_form": row["falseForm"],
            }
            for row in config["ontology"]["unaryPredicates"]
        ],
        "relations": [
            {
                "id": row["id"], "source_type": row["sourceType"], "target_type": row["targetType"],
                "direct_positive_form": row["directTrueForm"], "direct_negative_form": row["directFalseForm"],
                "inverse_positive_form": row["inverseTrueForm"], "inverse_negative_form": row["inverseFalseForm"],
            }
            for row in config["ontology"]["relations"]
        ],
    }


def render_form(template: str, arguments: Sequence[str]) -> str:
    if len(arguments) == 1:
        return template.format(entity=arguments[0])
    return template.format(source=arguments[0], target=arguments[1])


def enumerate_lexicalizations(row: dict[str, Any]) -> list[tuple[str, tuple[str, ...], str, str, str]]:
    ontology = row["agent_input"]["predicate_ontology"]
    by_type: dict[str, list[str]] = {}
    for entity_type in ontology["entity_types"]:
        by_type[entity_type] = [
            entity["id"] for entity in row["agent_input"]["entities"]
            if entity["entity_type"] == entity_type
        ]
    values = []
    for predicate in ontology["unary_predicates"]:
        for entity in by_type[predicate["entity_type"]]:
            for sign, key in (("positive", "positive_form"), ("negative", "negative_form")):
                values.append((predicate["id"], (entity,), sign, "unary", render_form(predicate[key], (entity,))))
    for relation in ontology["relations"]:
        for source in by_type[relation["source_type"]]:
            for target in by_type[relation["target_type"]]:
                if source == target:
                    continue
                for orientation in ("direct", "inverse"):
                    for sign in ("positive", "negative"):
                        key = f"{orientation}_{sign}_form"
                        values.append((
                            relation["id"], (source, target), sign, orientation,
                            render_form(relation[key], (source, target)),
                        ))
    return values


def extract_literal_candidates(row: dict[str, Any]) -> list[LiteralCandidate]:
    evidence = row["agent_input"]["evidence_text"]
    found = []
    for predicate, arguments, sign, orientation, text in enumerate_lexicalizations(row):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(text)}(?![A-Za-z0-9_])"
        for match in re.finditer(pattern, evidence, flags=re.IGNORECASE):
            found.append(LiteralCandidate(predicate, arguments, sign, orientation, text, match.start(), match.end()))
    # Prefer longest lexicalization at an identical span; otherwise preserve textual order.
    unique = {}
    for candidate in sorted(found, key=lambda item: (item.start, -(item.end - item.start), item.text)):
        unique.setdefault((candidate.start, candidate.end), candidate)
    return list(unique.values())


def render_surface(split: str, operation: str, surface: str, focus: str, decoy: str) -> str:
    return SURFACE_TEMPLATES[split][operation][surface].format(focus=focus, decoy=decoy)


def normalized_template(split: str, operation: str, surface: str) -> str:
    template = SURFACE_TEMPLATES[split][operation][surface]
    template = re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())
    return re.sub(r"\s+", " ", template).strip()


def deterministic_focus_index(row: dict[str, Any], candidates: Sequence[LiteralCandidate]) -> int:
    evidence = row["agent_input"]["evidence_text"]
    matches = []
    for split_registry in SURFACE_TEMPLATES.values():
        for operation_registry in split_registry.values():
            for template in operation_registry.values():
                pattern = re.escape(template)
                pattern = pattern.replace(re.escape("{focus}"), "(?P<focus>.+?)")
                pattern = pattern.replace(re.escape("{decoy}"), "(?P<decoy>.+?)")
                match = re.fullmatch(pattern, evidence, flags=re.IGNORECASE)
                if match:
                    matches.append(match.span("focus"))
    if len(matches) != 1:
        raise ValueError(f"V38 deterministic grammar matched {len(matches)} surfaces")
    start, end = matches[0]
    selected = [index for index, candidate in enumerate(candidates) if candidate.start == start and candidate.end == end]
    if len(selected) != 1:
        raise ValueError("V38 focus grammar did not select exactly one grounded literal")
    return selected[0]


def candidate_prompt(row: dict[str, Any], candidate: LiteralCandidate) -> str:
    return (
        f"Declared ontology with lexical definitions: {row['agent_input']['predicate_ontology']}\n"
        f"Evidence: {row['agent_input']['evidence_text']}\n"
        f"Candidate grounded literal: {candidate.text}\n"
        "Does the discourse treat this candidate span as the embedded proposition at issue, rather than as an opposite, comparison, or aside? Answer Yes or No.\n"
        "Answer:"
    )
