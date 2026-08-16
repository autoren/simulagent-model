"""Immutable supported-language registry and rendering rules for V36."""

from __future__ import annotations

import re
from typing import Any, Sequence

from v30_language import fact_phrases, sha256_text


SURFACE_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "assert": {
        "confirm_a": ("As an observed fact, the embedded report holds: {literal}.", "medium"),
        "confirm_b": ("The reading to enter as verified is that {literal}.", "medium"),
        "confirm_c": ("Observation ratifies the proposition according to which {literal}.", "long"),
    },
    "deny": {
        "confirm_a": ("The embedded report fails the observation check: {literal}.", "medium"),
        "confirm_b": ("Evidence rules against treating as correct the proposition that {literal}.", "long"),
        "confirm_c": ("The description to mark as refuted is the one saying {literal}.", "long"),
    },
    "double_deny": {
        "confirm_a": ("Evidence overturns the attempted refutation of this report: {literal}.", "long"),
        "confirm_b": ("Rejecting the embedded claim would be the mistaken judgment: {literal}.", "long"),
        "confirm_c": ("It is the denial of this proposition that observation rules out: {literal}.", "long"),
    },
    "contrast_select": {
        "confirm_a": ("Comparing the two readings, observation keeps {literal} and eliminates {opposite}.", "long"),
        "confirm_b": ("The supported member of the pair is {literal}, whereas {opposite} is discarded.", "long"),
        "confirm_c": ("The evidence resolves the opposition toward {literal} instead of {opposite}.", "long"),
    },
    "unresolved": {
        "confirm_a": ("Available observation supports neither settling nor refuting this report: {literal}.", "long"),
        "confirm_b": ("The proposition that {literal} remains open in both directions.", "medium"),
        "confirm_c": ("No verdict for or against the embedded description is warranted: {literal}.", "long"),
    },
}

DISTRACTOR = (
    "An unrelated maintenance bulletin lists storage-bin inspections and makes no claim about "
    "the present state."
)
GENERATOR_SEED = 3601
COLLISION_POLICY = "fail_closed_no_resampling_or_template_replacement"
NORMALIZATION_VERSION = "v36_slots_lower_whitespace_v1"


def validate_registry(config: dict[str, Any]) -> None:
    suite = config["confirmationSuite"]
    operations = suite["outerOperations"]
    surfaces = suite["newSurfaceNamesPerOperation"]
    if list(SURFACE_TEMPLATES) != operations:
        raise ValueError("V36 operation registry differs from the design")
    if any(list(SURFACE_TEMPLATES[operation]) != surfaces for operation in operations):
        raise ValueError("V36 surface-name registry differs from the design")
    templates = [value[0] for rows in SURFACE_TEMPLATES.values() for value in rows.values()]
    if len(templates) != 15 or len(set(templates)) != 15:
        raise ValueError("V36 requires exactly 15 unique surface templates")
    for operation, rows in SURFACE_TEMPLATES.items():
        for surface, (template, length) in rows.items():
            if "{literal}" not in template or length not in ("medium", "long"):
                raise ValueError(f"Invalid V36 template: {operation}.{surface}")
            if operation == "contrast_select" and "{opposite}" not in template:
                raise ValueError(f"V36 contrast template lacks opposite: {surface}")


def normalized_template(operation: str, surface_name: str) -> str:
    template = SURFACE_TEMPLATES[operation][surface_name][0]
    normalized = re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def construction_hash(operation: str, surface_name: str) -> str:
    return sha256_text(f"v36|{NORMALIZATION_VERSION}|{operation}|{normalized_template(operation, surface_name)}")


def render_evidence(
    predicate: str, arguments: Sequence[str], lexical_sign: str, operation: str,
    surface_name: str, orientation: str, distractor: bool, v32_config: dict[str, Any],
) -> tuple[str, str]:
    positive, negative = fact_phrases(predicate, arguments, orientation, v32_config)
    literal = positive if lexical_sign == "positive" else negative
    opposite = negative if lexical_sign == "positive" else positive
    template, length = SURFACE_TEMPLATES[operation][surface_name]
    text = template.format(literal=literal, opposite=opposite)
    if distractor:
        text = f"{DISTRACTOR} {text}"
    return text, length
