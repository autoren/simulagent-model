from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def service_family(service_name: str) -> str:
    match = re.fullmatch(r"(.+)_([0-9]+)", service_name)
    if match is None:
        raise ValueError(f"service identifier lacks numeric version suffix: {service_name}")
    return match.group(1)


def _schema_structure(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("schema must be a non-empty list")
    rows = []
    seen: set[str] = set()
    for service in payload:
        if not isinstance(service, dict):
            raise ValueError("service schema must be an object")
        name = service.get("service_name")
        intents = service.get("intents")
        slots = service.get("slots")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("service names must be unique non-empty strings")
        if not isinstance(intents, list) or not isinstance(slots, list):
            raise ValueError(f"service {name} requires intent and slot lists")
        if any(not isinstance(intent, dict) or not isinstance(intent.get("name"), str) for intent in intents):
            raise ValueError(f"service {name} has invalid intents")
        if any(not isinstance(slot, dict) or not isinstance(slot.get("name"), str) for slot in slots):
            raise ValueError(f"service {name} has invalid slots")
        seen.add(name)
        rows.append({
            "service": name,
            "family": service_family(name),
            "intent_count": len(intents),
            "slot_count": len(slots),
        })
    return rows


def build_test_schema_inventory(
    development_schema_payload: Any,
    test_schema_payload: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    development = _schema_structure(development_schema_payload)
    test = _schema_structure(test_schema_payload)
    development_families = frozenset(row["family"] for row in development)
    family_rule = config["familyRule"]
    eligible = [
        row for row in test
        if row["family"] not in development_families
        and row["intent_count"] >= family_rule["minimumTypedIntentCountPerEligibleService"]
        and row["slot_count"] >= family_rule["minimumSlotCountPerEligibleService"]
    ]
    novel_families = sorted({row["family"] for row in eligible})
    inventory_rows = [
        {
            "service": row["service"],
            "family": row["family"],
            "intent_count": row["intent_count"],
            "slot_count": row["slot_count"],
        }
        for row in sorted(eligible, key=lambda item: item["service"])
    ]
    forbidden = {"intent_name", "intent_names", "description", "slot_name", "slot_names", "tokens"}
    keys = set().union(*(row.keys() for row in inventory_rows)) if inventory_rows else set()
    if keys & forbidden:
        raise AssertionError("schema language leaked into structural inventory")
    return {
        "development_service_count": len(development),
        "development_family_count": len(development_families),
        "test_service_count": len(test),
        "test_family_count": len({row["family"] for row in test}),
        "novel_service_families": novel_families,
        "novel_service_family_count": len(novel_families),
        "eligible_novel_services": inventory_rows,
        "eligible_novel_service_count": len(inventory_rows),
        "inventory_sha256": canonical_sha256(inventory_rows),
        "emitted_intent_name_count": 0,
        "emitted_intent_description_count": 0,
        "emitted_slot_name_count": 0,
        "contains_schema_language_or_surface_tokens": False,
    }


__all__ = ["build_test_schema_inventory", "git_blob_sha1", "service_family"]
