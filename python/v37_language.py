"""Immutable V37 validation language and candidate-conditioned prompts."""

from __future__ import annotations

import re
from typing import Any, Sequence

from v30_language import fact_phrases, sha256_text


SURFACE_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "assert": {
        "invariance_a": ("Treat the following embedded description as matching the observation: {literal}.", "long"),
        "invariance_b": ("The observation endorses what this report says—namely, {literal}.", "long"),
    },
    "deny": {
        "invariance_a": ("Observation contradicts, rather than accepts, this embedded description: {literal}.", "long"),
        "invariance_b": ("The proposition that {literal} is the one the evidence rejects.", "medium"),
    },
    "double_deny": {
        "invariance_a": ("Evidence does not permit the embedded report to be rejected: {literal}.", "long"),
        "invariance_b": ("The attempted dismissal of the claim that {literal} is itself overturned.", "long"),
    },
    "contrast_select": {
        "invariance_a": ("Between {literal} and {opposite}, the evidence favors the former.", "medium"),
        "invariance_b": ("Of the opposed alternatives, retain {literal}; do not retain {opposite}.", "long"),
    },
    "unresolved": {
        "invariance_a": ("The observation leaves undecided whether to accept or reject this report: {literal}.", "long"),
        "invariance_b": ("Evidence supplies no basis for choosing between {literal} and {opposite}.", "long"),
    },
}

DISTRACTOR_PREFIX = (
    "A scheduling notice mentions battery-rack inventory and is unrelated to the present observation."
)
DISTRACTOR_SUFFIX = (
    "Separately, an inventory note records shelf counts without describing the current state."
)
NORMALIZATION_VERSION = "v37_slots_lower_whitespace_v1"

SIGN_DEFINITIONS = {
    "positive": "The embedded literal states its canonical proposition without lexical negation.",
    "negative": "The embedded literal itself contains lexical negation of its canonical proposition.",
}

OPERATION_DEFINITIONS = {
    "assert": "The evidence endorses, verifies, or records the embedded literal.",
    "deny": "The evidence rejects, refutes, or contradicts the embedded literal.",
    "double_deny": "The evidence rejects or overturns a rejection of the embedded literal.",
    "contrast_select": "The evidence explicitly selects the embedded literal over its stated opposite.",
    "unresolved": "The evidence withholds a decision in both directions, neither endorsing nor rejecting the embedded literal.",
}


def validate_registry(config: dict[str, Any]) -> None:
    operations = config["interfaces"]["outerOperationClasses"]
    surfaces = config["freshValidation"]["surfaceNamesPerOperation"]
    if list(SURFACE_TEMPLATES) != operations:
        raise ValueError("V37 operation registry differs from the design")
    if any(list(SURFACE_TEMPLATES[operation]) != surfaces for operation in operations):
        raise ValueError("V37 surface registry differs from the design")
    templates = [value[0] for values in SURFACE_TEMPLATES.values() for value in values.values()]
    if len(templates) != 10 or len(set(templates)) != 10:
        raise ValueError("V37 requires exactly ten unique validation templates")
    if set(SIGN_DEFINITIONS) != set(config["interfaces"]["lexicalSignClasses"]):
        raise ValueError("V37 sign definitions differ from the design")
    if set(OPERATION_DEFINITIONS) != set(operations):
        raise ValueError("V37 operation definitions differ from the design")


def normalized_template(operation: str, surface_name: str) -> str:
    template = SURFACE_TEMPLATES[operation][surface_name][0]
    template = re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())
    return re.sub(r"\s+", " ", template).strip()


def construction_hash(operation: str, surface_name: str) -> str:
    return sha256_text(
        f"v37|{NORMALIZATION_VERSION}|{operation}|{normalized_template(operation, surface_name)}"
    )


def render_evidence(
    predicate: str,
    arguments: Sequence[str],
    lexical_sign: str,
    operation: str,
    surface_name: str,
    orientation: str,
    distractor_placement: str,
    v32_config: dict[str, Any],
) -> tuple[str, str]:
    positive, negative = fact_phrases(predicate, arguments, orientation, v32_config)
    literal = positive if lexical_sign == "positive" else negative
    opposite = negative if lexical_sign == "positive" else positive
    template, length = SURFACE_TEMPLATES[operation][surface_name]
    evidence = template.format(literal=literal, opposite=opposite)
    if distractor_placement == "prefix":
        evidence = f"{DISTRACTOR_PREFIX} {evidence}"
    elif distractor_placement == "suffix":
        evidence = f"{evidence} {DISTRACTOR_SUFFIX}"
    elif distractor_placement != "none":
        raise ValueError(f"Unknown V37 distractor placement: {distractor_placement}")
    return evidence, length


def candidate_prompt(row: dict[str, Any], component: str, candidate: str) -> str:
    evidence = row["agent_input"]["evidence_text"]
    if component == "lexical_sign":
        definition = SIGN_DEFINITIONS[candidate]
        question = (
            "Judge the lexical sign of the embedded literal itself before any outer evidence "
            "operation is applied. Ignore unrelated statements."
        )
    elif component == "outer_operation":
        definition = OPERATION_DEFINITIONS[candidate]
        question = (
            "Judge only the outer semantic operation applied to the embedded literal. Preserve "
            "the distinction between rejecting a literal and rejecting its rejection, and ignore "
            "unrelated statements."
        )
    else:
        raise ValueError(f"Unknown V37 component: {component}")
    return (
        f"Evidence statement: {evidence}\n"
        f"Task: {question}\n"
        f"Candidate analysis: {candidate} — {definition}\n"
        "Is this candidate analysis compatible with the evidence? Answer Yes or No.\n"
        "Answer:"
    )
