#!/usr/bin/env python3
"""Generic typed-schema clarification renderer with a fail-closed provenance boundary."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable


AUTHORIZED_SOURCES = ("canonical", "finite_grammar")
FINITE_GRAMMAR_STYLES = (
    "concise", "neutral", "polite", "formal",
    "direct", "friendly", "explicit", "minimal",
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
FORBIDDEN_EXECUTION_CLAIMS = (
    "i have", "i've", "i will", "i'll", "already", "completed",
    "executed", "scheduled", "sent", "booked", "done",
)
MACHINE_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SchemaBoundaryError(ValueError):
    """Typed failure at schema/request boundaries; never triggers a fallback guess."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SchemaOption:
    option_id: str
    surface: str


@dataclass(frozen=True)
class SchemaSlot:
    slot_id: str
    question_prefix: str
    options: tuple[SchemaOption, SchemaOption]


@dataclass(frozen=True)
class TypedSchema:
    schema_id: str
    slots: tuple[SchemaSlot, SchemaSlot]


@dataclass(frozen=True)
class SchemaRegistry:
    schemas: tuple[TypedSchema, ...]

    def schema(self, schema_id: str) -> TypedSchema:
        for schema in self.schemas:
            if schema.schema_id == schema_id:
                return schema
        raise SchemaBoundaryError("unknown_schema", f"unknown schema: {schema_id}")


@dataclass(frozen=True)
class ClarificationRequest:
    schema_id: str
    kind: str
    slot_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SchemaSurfaceCertificate:
    typed_request: ClarificationRequest
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
class RenderedSchemaClarification:
    typed_request: ClarificationRequest
    question: str
    source: str
    style: str | None
    certificate: SchemaSurfaceCertificate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_text(value: Any, maximum: int, field: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise SchemaBoundaryError("invalid_schema_text", f"invalid {field}")
    lower = value.lower()
    if (
        not value.isascii()
        or "?" in value
        or "\n" in value
        or "\r" in value
        or "_" in value
        or any(fragment in lower for fragment in FORBIDDEN_EXECUTION_CLAIMS)
    ):
        raise SchemaBoundaryError("unsafe_schema_text", f"unsafe {field}")
    return value


def _machine_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or MACHINE_ID.fullmatch(value) is None:
        raise SchemaBoundaryError("invalid_machine_id", f"invalid {field}")
    return value


def compile_schema_registry(raw_schemas: Any) -> SchemaRegistry:
    if not isinstance(raw_schemas, list) or not raw_schemas:
        raise SchemaBoundaryError("invalid_schema_population", "schemas must be a nonempty list")
    schemas: list[TypedSchema] = []
    seen_schema_ids: set[str] = set()
    for raw_schema in raw_schemas:
        if not isinstance(raw_schema, dict) or set(raw_schema) != {"schemaId", "slots"}:
            raise SchemaBoundaryError("invalid_schema_shape", "schema keys must be exact")
        schema_id = _machine_id(raw_schema["schemaId"], "schemaId")
        if schema_id in seen_schema_ids:
            raise SchemaBoundaryError("duplicate_schema_id", "schema IDs must be unique")
        seen_schema_ids.add(schema_id)
        raw_slots = raw_schema["slots"]
        if not isinstance(raw_slots, list) or len(raw_slots) != 2:
            raise SchemaBoundaryError("invalid_slot_count", "each schema requires exactly two slots")
        slots: list[SchemaSlot] = []
        seen_slot_ids: set[str] = set()
        for raw_slot in raw_slots:
            if not isinstance(raw_slot, dict) or set(raw_slot) != {
                "slotId", "questionPrefix", "options"
            }:
                raise SchemaBoundaryError("invalid_slot_shape", "slot keys must be exact")
            slot_id = _machine_id(raw_slot["slotId"], "slotId")
            if slot_id in seen_slot_ids:
                raise SchemaBoundaryError("duplicate_slot_id", "slot IDs must be unique per schema")
            seen_slot_ids.add(slot_id)
            prefix = _safe_text(raw_slot["questionPrefix"], 48, "questionPrefix")
            raw_options = raw_slot["options"]
            if not isinstance(raw_options, list) or len(raw_options) != 2:
                raise SchemaBoundaryError("invalid_option_count", "each slot requires exactly two options")
            options: list[SchemaOption] = []
            seen_option_ids: set[str] = set()
            seen_surfaces: set[str] = set()
            for raw_option in raw_options:
                if not isinstance(raw_option, dict) or set(raw_option) != {
                    "optionId", "surface"
                }:
                    raise SchemaBoundaryError("invalid_option_shape", "option keys must be exact")
                option_id = _machine_id(raw_option["optionId"], "optionId")
                surface = _safe_text(raw_option["surface"], 64, "option surface")
                if option_id in seen_option_ids:
                    raise SchemaBoundaryError("duplicate_option_id", "option IDs must be unique per slot")
                if surface in seen_surfaces:
                    raise SchemaBoundaryError("duplicate_option_surface", "option surfaces must differ")
                seen_option_ids.add(option_id)
                seen_surfaces.add(surface)
                options.append(SchemaOption(option_id, surface))
            slots.append(SchemaSlot(slot_id, prefix, (options[0], options[1])))
        schemas.append(TypedSchema(schema_id, (slots[0], slots[1])))
    return SchemaRegistry(tuple(schemas))


def parse_clarification_request(raw: Any) -> ClarificationRequest:
    if not isinstance(raw, dict) or set(raw) != {"schema_id", "kind", "slot_id"}:
        raise SchemaBoundaryError("invalid_request_shape", "request fields must be exact")
    schema_id = _machine_id(raw["schema_id"], "schema_id")
    kind = raw["kind"]
    slot_id = raw["slot_id"]
    if kind not in ("slot", "all"):
        raise SchemaBoundaryError("invalid_request_kind", f"unknown request kind: {kind}")
    if kind == "slot":
        if not isinstance(slot_id, str):
            raise SchemaBoundaryError("missing_slot_id", "slot requests require slot_id")
        _machine_id(slot_id, "slot_id")
    elif slot_id is not None:
        raise SchemaBoundaryError("unexpected_slot_id", "all requests require null slot_id")
    return ClarificationRequest(schema_id, kind, slot_id)


def _resolve_slots(
    registry: SchemaRegistry, request: ClarificationRequest
) -> tuple[SchemaSlot, ...]:
    schema = registry.schema(request.schema_id)
    if request.kind == "all":
        if request.slot_id is not None:
            raise SchemaBoundaryError("unexpected_slot_id", "all requests require null slot_id")
        return schema.slots
    if request.kind != "slot" or request.slot_id is None:
        raise SchemaBoundaryError("invalid_request_kind", "slot request is incomplete")
    for slot in schema.slots:
        if slot.slot_id == request.slot_id:
            return (slot,)
    raise SchemaBoundaryError("unknown_slot", f"unknown slot: {request.slot_id}")


def _choice_fragment(slot: SchemaSlot) -> str:
    return f"{slot.options[0].surface} or {slot.options[1].surface}"


def canonical_schema_surface(
    registry: SchemaRegistry, request: ClarificationRequest
) -> str:
    slots = _resolve_slots(registry, request)
    clauses = [f"{slot.question_prefix} {_choice_fragment(slot)}" for slot in slots]
    if len(clauses) == 1:
        return clauses[0] + "?"
    continuation = clauses[1][0].lower() + clauses[1][1:]
    return f"{clauses[0]}, and {continuation}?"


def certify_schema_surface(
    registry: SchemaRegistry,
    request: ClarificationRequest,
    question: str,
    source: str,
) -> SchemaSurfaceCertificate:
    slots = _resolve_slots(registry, request)
    schema = registry.schema(request.schema_id)
    required = tuple(_choice_fragment(slot) for slot in slots)
    forbidden = tuple(
        _choice_fragment(slot) for slot in schema.slots if slot not in slots
    )
    structural = bool(
        isinstance(question, str)
        and 1 <= len(question) <= 320
        and question.isascii()
        and question.count("?") == 1
        and question.endswith("?")
        and "\n" not in question
        and "\r" not in question
        and "_" not in question
    )
    exact_choices = bool(
        isinstance(question, str)
        and all(question.count(fragment) == 1 for fragment in required)
    )
    unrequested = (
        sum(question.count(fragment) for fragment in forbidden)
        if isinstance(question, str) else len(forbidden)
    )
    lower = question.lower() if isinstance(question, str) else ""
    claims = sum(lower.count(fragment) for fragment in FORBIDDEN_EXECUTION_CLAIMS)
    content = bool(structural and exact_choices and unrequested == 0 and claims == 0)
    authorized = source in AUTHORIZED_SOURCES
    deployable = bool(content and authorized)
    reason = (
        "authorized_model_free_schema_surface" if deployable
        else "invalid_surface_content" if not content
        else "unauthorized_surface_source"
    )
    return SchemaSurfaceCertificate(
        typed_request=request,
        source=source,
        structural_valid=structural,
        exact_choice_fragments_valid=exact_choices,
        unrequested_choice_fragment_count=unrequested,
        forbidden_execution_claim_count=claims,
        content_valid=content,
        source_authorized=authorized,
        deployable=deployable,
        reason=reason,
    )


def render_schema_clarification(
    registry: SchemaRegistry,
    request: ClarificationRequest,
    *,
    source: str = "canonical",
    style: str | None = None,
) -> RenderedSchemaClarification:
    if source not in AUTHORIZED_SOURCES:
        raise PermissionError(f"schema surface source is disabled: {source}")
    canonical = canonical_schema_surface(registry, request)
    if source == "canonical":
        if style is not None:
            raise SchemaBoundaryError("unexpected_style", "canonical rendering has no style")
        question = canonical
    else:
        if style not in FINITE_GRAMMAR_STYLES:
            raise SchemaBoundaryError("unknown_style", f"unknown style: {style}")
        core = canonical[:-1]
        core = core[0].lower() + core[1:]
        question = GRAMMAR_WRAPPERS[style].format(core=core)
    certificate = certify_schema_surface(registry, request, question, source)
    if not certificate.deployable:
        raise RuntimeError("internal schema surface failed its frozen certificate")
    return RenderedSchemaClarification(request, question, source, style, certificate)


def inspect_untrusted_schema_surface(
    registry: SchemaRegistry, request: ClarificationRequest, question: str
) -> SchemaSurfaceCertificate:
    return certify_schema_surface(registry, request, question, "untrusted_passthrough")


V79_ACTION_TO_REQUEST = {
    "ask_operation": ClarificationRequest("project_workflow", "slot", "operation"),
    "ask_recipient": ClarificationRequest("project_workflow", "slot", "recipient"),
    "ask_full_details": ClarificationRequest("project_workflow", "all", None),
}


def decorate_v79_policy_node(
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
    rendered = render_schema_clarification(registry, request, source=source, style=style)
    result["schema_clarification_surface"] = rendered.to_dict()
    return result


def invalid_schema_mutations(raw_schemas: list[dict[str, Any]]) -> tuple[tuple[str, list[dict[str, Any]]], ...]:
    def changed() -> list[dict[str, Any]]:
        return deepcopy(raw_schemas)

    rows: list[tuple[str, list[dict[str, Any]]]] = []
    value = changed(); value[1]["schemaId"] = value[0]["schemaId"]; rows.append(("duplicate_schema_id", value))
    value = changed(); value[0]["slots"][1]["slotId"] = value[0]["slots"][0]["slotId"]; rows.append(("duplicate_slot_id", value))
    value = changed(); del value[0]["slots"][0]["slotId"]; rows.append(("missing_slot_id", value))
    value = changed(); value[0]["slots"][0]["options"] = value[0]["slots"][0]["options"][:1]; rows.append(("one_option", value))
    value = changed(); value[0]["slots"][0]["options"].append({"optionId": "third_option", "surface": "third option"}); rows.append(("three_options", value))
    value = changed(); value[0]["slots"][0]["options"][1]["optionId"] = value[0]["slots"][0]["options"][0]["optionId"]; rows.append(("duplicate_option_id", value))
    value = changed(); value[0]["slots"][0]["options"][1]["surface"] = value[0]["slots"][0]["options"][0]["surface"]; rows.append(("duplicate_option_surface", value))
    value = changed(); value[0]["slots"][0]["options"][0]["surface"] = "unsafe\nvalue"; rows.append(("unsafe_surface_newline", value))
    value = changed(); value[0]["slots"][0]["options"][0]["surface"] = "unsafe_value"; rows.append(("unsafe_surface_underscore", value))
    value = changed(); value[0]["slots"][0]["options"][0]["surface"] = "unsafe value?"; rows.append(("unsafe_surface_question_mark", value))
    value = changed(); value[0]["slots"][0]["options"][0]["surface"] = "I will execute"; rows.append(("unsafe_surface_execution_claim", value))
    value = changed(); del value[0]["slots"][0]["questionPrefix"]; rows.append(("missing_question_prefix", value))
    return tuple(rows)


def invalid_request_population(registry: SchemaRegistry) -> tuple[dict[str, Any], ...]:
    first = registry.schemas[0]
    rows: list[dict[str, Any]] = [
        {"schema_id": "unknown_schema", "kind": "slot", "slot_id": first.slots[0].slot_id}
    ]
    rows.extend(
        {"schema_id": schema.schema_id, "kind": "slot", "slot_id": "unknown_slot"}
        for schema in registry.schemas
    )
    rows.extend(
        {"schema_id": schema.schema_id, "kind": "slot", "slot_id": None}
        for schema in registry.schemas
    )
    rows.extend(
        {"schema_id": schema.schema_id, "kind": "all", "slot_id": schema.slots[0].slot_id}
        for schema in registry.schemas
    )
    return tuple(rows)


def unsafe_schema_surface_mutations(
    registry: SchemaRegistry,
) -> tuple[tuple[ClarificationRequest, str], ...]:
    rows: list[tuple[ClarificationRequest, str]] = []
    for schema in registry.schemas:
        first, second = schema.slots
        first_request = ClarificationRequest(schema.schema_id, "slot", first.slot_id)
        second_request = ClarificationRequest(schema.schema_id, "slot", second.slot_id)
        first_and = f"{first.question_prefix} {first.options[0].surface} and {first.options[1].surface}?"
        second_missing = f"{second.question_prefix} {second.options[0].surface}?"
        second_unrequested = (
            f"{second.question_prefix} {_choice_fragment(second)}, and {_choice_fragment(first)}?"
        )
        first_claim = f"I will {_choice_fragment(first)}?"
        rows.extend((
            (first_request, first_and),
            (second_request, second_missing),
            (second_request, second_unrequested),
            (first_request, first_claim),
        ))
    return tuple(rows)
