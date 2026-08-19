#!/usr/bin/env python3
"""V86 hardening: reject individual options from every unrequested slot."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from schema_grounded_interface import (
    AUTHORIZED_SOURCES,
    FINITE_GRAMMAR_STYLES,
    V79_ACTION_TO_REQUEST,
    ClarificationRequest,
    RenderedSchemaClarification,
    SchemaRegistry,
    SchemaSurfaceCertificate,
    canonical_schema_surface,
    certify_schema_surface,
    render_schema_clarification,
)


@dataclass(frozen=True)
class HardenedSchemaSurfaceCertificate:
    typed_request: ClarificationRequest
    source: str
    base_structural_valid: bool
    base_exact_choice_fragments_valid: bool
    complete_unrequested_choice_fragment_count: int
    individual_unrequested_option_surface_count: int
    forbidden_execution_claim_count: int
    content_valid: bool
    source_authorized: bool
    deployable: bool
    reason: str


@dataclass(frozen=True)
class HardenedRenderedSchemaClarification:
    typed_request: ClarificationRequest
    question: str
    source: str
    style: str | None
    certificate: HardenedSchemaSurfaceCertificate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _requested_and_unrequested_slots(registry: SchemaRegistry, request: ClarificationRequest):
    schema = registry.schema(request.schema_id)
    if request.kind == "all":
        if request.slot_id is not None:
            raise ValueError("all requests require null slot_id")
        return schema.slots, ()
    requested = tuple(slot for slot in schema.slots if slot.slot_id == request.slot_id)
    if len(requested) != 1:
        raise ValueError(f"unknown requested slot: {request.slot_id}")
    unrequested = tuple(slot for slot in schema.slots if slot not in requested)
    return requested, unrequested


def hardened_certify_schema_surface(
    registry: SchemaRegistry,
    request: ClarificationRequest,
    question: str,
    source: str,
) -> HardenedSchemaSurfaceCertificate:
    base = certify_schema_surface(registry, request, question, source)
    _, unrequested = _requested_and_unrequested_slots(registry, request)
    individual_count = sum(
        question.count(option.surface)
        for slot in unrequested
        for option in slot.options
    )
    content_valid = bool(base.content_valid and individual_count == 0)
    deployable = bool(content_valid and base.source_authorized)
    reason = (
        "authorized_hardened_model_free_schema_surface" if deployable
        else "unrequested_individual_option_surface" if individual_count
        else "invalid_surface_content" if not content_valid
        else "unauthorized_surface_source"
    )
    return HardenedSchemaSurfaceCertificate(
        typed_request=request,
        source=source,
        base_structural_valid=base.structural_valid,
        base_exact_choice_fragments_valid=base.exact_choice_fragments_valid,
        complete_unrequested_choice_fragment_count=base.unrequested_choice_fragment_count,
        individual_unrequested_option_surface_count=individual_count,
        forbidden_execution_claim_count=base.forbidden_execution_claim_count,
        content_valid=content_valid,
        source_authorized=base.source_authorized,
        deployable=deployable,
        reason=reason,
    )


def render_hardened_schema_clarification(
    registry: SchemaRegistry,
    request: ClarificationRequest,
    *,
    source: str = "canonical",
    style: str | None = None,
) -> HardenedRenderedSchemaClarification:
    if source not in AUTHORIZED_SOURCES:
        raise PermissionError(f"schema surface source is disabled: {source}")
    base: RenderedSchemaClarification = render_schema_clarification(
        registry, request, source=source, style=style
    )
    certificate = hardened_certify_schema_surface(
        registry, request, base.question, source
    )
    if not certificate.deployable:
        raise RuntimeError("internal V86 surface failed hardened certification")
    return HardenedRenderedSchemaClarification(
        request, base.question, source, style, certificate
    )


def inspect_untrusted_hardened_surface(
    registry: SchemaRegistry, request: ClarificationRequest, question: str
) -> HardenedSchemaSurfaceCertificate:
    return hardened_certify_schema_surface(
        registry, request, question, "untrusted_passthrough"
    )


def decorate_v79_policy_node_v86(
    policy_node: dict[str, Any],
    registry: SchemaRegistry,
    *,
    source: str = "canonical",
    style: str | None = None,
) -> dict[str, Any]:
    result = deepcopy(policy_node)
    request = V79_ACTION_TO_REQUEST.get(result.get("action"))
    if request is None:
        return result
    rendered = render_hardened_schema_clarification(
        registry, request, source=source, style=style
    )
    result["schema_clarification_surface"] = rendered.to_dict()
    return result


def partial_option_injection_mutations(
    registry: SchemaRegistry,
) -> tuple[tuple[ClarificationRequest, str, str], ...]:
    rows: list[tuple[ClarificationRequest, str, str]] = []
    for schema in registry.schemas:
        for requested in schema.slots:
            request = ClarificationRequest(schema.schema_id, "slot", requested.slot_id)
            base = canonical_schema_surface(registry, request)[:-1]
            other = next(slot for slot in schema.slots if slot != requested)
            for option in other.options:
                rows.append((request, f"{base}, and {option.surface}?", option.surface))
    return tuple(rows)
