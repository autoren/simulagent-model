#!/usr/bin/env python3
"""Strict model-free clarification surfaces for the frozen V79 decision core.

This module is deliberately incapable of accepting generated or arbitrary text for
deployment.  The only deployable surfaces are constants or finite-grammar strings
constructed here.  Inspection of untrusted text returns a non-deployable certificate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any


CLARIFICATION_CODES = (
    "ask_operation",
    "ask_recipient",
    "ask_full_details",
)
AUTHORIZED_SOURCES = ("canonical", "finite_grammar")
FINITE_GRAMMAR_STYLES = (
    "concise",
    "neutral",
    "polite",
    "formal",
    "direct",
    "friendly",
    "explicit",
    "minimal",
)
OPERATION_CHOICE = "schedule the project review or send the project summary"
RECIPIENT_CHOICE = "Alex Chen or Alex Kim"
STRICT_CHOICE_FRAGMENTS = {
    "ask_operation": (OPERATION_CHOICE,),
    "ask_recipient": (RECIPIENT_CHOICE,),
    "ask_full_details": (OPERATION_CHOICE, RECIPIENT_CHOICE),
}
CANONICAL_SURFACES = {
    "ask_operation": "Should I schedule the project review or send the project summary?",
    "ask_recipient": "Should the recipient be Alex Chen or Alex Kim?",
    "ask_full_details": (
        "Should I schedule the project review or send the project summary, and should "
        "the recipient be Alex Chen or Alex Kim?"
    ),
}
FORBIDDEN_EXECUTION_CLAIMS = (
    "i have",
    "i've",
    "i will",
    "i'll",
    "already",
    "completed",
    "executed",
    "scheduled",
    "sent",
    "booked",
    "done",
)
GRAMMAR_WRAPPERS = {
    "concise": "{core}?",
    "neutral": "Please clarify: {core}?",
    "polite": "Could you please clarify: {core}?",
    "formal": "Please specify: {core}?",
    "direct": "Direct clarification: {core}?",
    "friendly": "Could you help me clarify: {core}?",
    "explicit": "For clarity: {core}?",
    "minimal": "{core}?",
}


@dataclass(frozen=True)
class SurfaceCertificate:
    action_code: str
    source: str
    structural_valid: bool
    exact_choice_fragments_valid: bool
    unrequested_choice_fragment_count: int
    forbidden_execution_claim_count: int
    content_valid: bool
    source_authorized: bool
    deployable: bool
    reason: str


@dataclass(frozen=True)
class RenderedClarification:
    action_code: str
    question: str
    source: str
    style: str | None
    certificate: SurfaceCertificate

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value


def _require_code(action_code: str) -> None:
    if action_code not in CLARIFICATION_CODES:
        raise ValueError(f"not a clarification action code: {action_code}")


def certify_surface(
    action_code: str, question: str, source: str
) -> SurfaceCertificate:
    """Certify content and provenance without converting untrusted text to authority."""
    _require_code(action_code)
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    required = STRICT_CHOICE_FRAGMENTS[action_code]
    all_fragments = (OPERATION_CHOICE, RECIPIENT_CHOICE)
    forbidden = tuple(fragment for fragment in all_fragments if fragment not in required)
    structural_valid = bool(
        1 <= len(question) <= 180
        and question.isascii()
        and question.count("?") == 1
        and question.endswith("?")
        and "\n" not in question
        and "\r" not in question
        and "_" not in question
    )
    exact_choice_fragments_valid = all(question.count(fragment) == 1 for fragment in required)
    unrequested_count = sum(question.count(fragment) for fragment in forbidden)
    lower = question.lower()
    claim_count = sum(lower.count(fragment) for fragment in FORBIDDEN_EXECUTION_CLAIMS)
    content_valid = bool(
        structural_valid
        and exact_choice_fragments_valid
        and unrequested_count == 0
        and claim_count == 0
    )
    source_authorized = source in AUTHORIZED_SOURCES
    deployable = bool(content_valid and source_authorized)
    if deployable:
        reason = "authorized_model_free_surface"
    elif not content_valid:
        reason = "invalid_surface_content"
    else:
        reason = "unauthorized_surface_source"
    return SurfaceCertificate(
        action_code=action_code,
        source=source,
        structural_valid=structural_valid,
        exact_choice_fragments_valid=exact_choice_fragments_valid,
        unrequested_choice_fragment_count=unrequested_count,
        forbidden_execution_claim_count=claim_count,
        content_valid=content_valid,
        source_authorized=source_authorized,
        deployable=deployable,
        reason=reason,
    )


def finite_grammar_surface(action_code: str, style: str) -> str:
    _require_code(action_code)
    if style not in FINITE_GRAMMAR_STYLES:
        raise ValueError(f"unknown finite-grammar style: {style}")
    operation = "should I schedule the project review or send the project summary"
    recipient = "should the recipient be Alex Chen or Alex Kim"
    if action_code == "ask_operation":
        core = operation
    elif action_code == "ask_recipient":
        core = recipient
    else:
        core = f"{operation}, and {recipient}"
    question = GRAMMAR_WRAPPERS[style].format(core=core)
    return question[0].upper() + question[1:]


def render_clarification(
    action_code: str,
    *,
    source: str = "canonical",
    style: str | None = None,
) -> RenderedClarification:
    """Render only from the two frozen model-free sources."""
    _require_code(action_code)
    if source not in AUTHORIZED_SOURCES:
        raise PermissionError(f"clarification source is disabled: {source}")
    if source == "canonical":
        if style is not None:
            raise ValueError("canonical rendering does not accept a style")
        question = CANONICAL_SURFACES[action_code]
    else:
        if style is None:
            raise ValueError("finite_grammar rendering requires a style")
        question = finite_grammar_surface(action_code, style)
    certificate = certify_surface(action_code, question, source)
    if not certificate.deployable:
        raise RuntimeError("internal clarification surface failed its frozen certificate")
    return RenderedClarification(
        action_code=action_code,
        question=question,
        source=source,
        style=style,
        certificate=certificate,
    )


def inspect_untrusted_surface(action_code: str, question: str) -> SurfaceCertificate:
    """Inspect arbitrary text; the returned certificate is never deployable."""
    return certify_surface(action_code, question, "untrusted_passthrough")


def decorate_policy_node(
    policy_node: dict[str, Any],
    *,
    source: str = "canonical",
    style: str | None = None,
) -> dict[str, Any]:
    """Attach a surface to an ask node without changing any authoritative field."""
    result = deepcopy(policy_node)
    action = result.get("action")
    if action not in CLARIFICATION_CODES:
        return result
    rendered = render_clarification(action, source=source, style=style)
    result["clarification_surface"] = rendered.to_dict()
    return result


def unsafe_surface_mutations() -> tuple[tuple[str, str], ...]:
    """Locked negative controls covering V82's discovered false-positive class."""
    return (
        ("ask_operation", "Should I schedule the project review and send the project summary?"),
        ("ask_operation", "Should I schedule the project review?"),
        ("ask_recipient", "Should the recipient be Alex Chen and Alex Kim?"),
        ("ask_recipient", "Should the recipient be Alex Chen or Alex Kim, and should I schedule the project review or send the project summary?"),
        ("ask_full_details", CANONICAL_SURFACES["ask_operation"]),
        ("ask_full_details", CANONICAL_SURFACES["ask_recipient"]),
        ("ask_operation", "I will schedule the project review or send the project summary?"),
        ("ask_recipient", "Should the recipient be Alex Chen or Alex Kim??"),
        ("ask_full_details", "ask_full_details: should I schedule the project review or send the project summary, and should the recipient be Alex Chen or Alex Kim?"),
        ("ask_operation", "Should I schedule the project review or send the project summary, schedule the project review or send the project summary?"),
    )
