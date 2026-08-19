#!/usr/bin/env python3
"""Pinned-source acquisition and text-free structural inventory helpers for V87."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any
from urllib.request import Request, urlopen


PINNED_REPOSITORY = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue"
PINNED_RAW_BASE = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    "dstc8-schema-guided-dialogue/e852981ae34990f4358979625854259302feaa78/"
)


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fetch_pinned_file(path: str, *, expected_size: int, expected_blob_sha1: str) -> bytes:
    if path not in {"dev/schema.json", "dev/dialogues_001.json"}:
        raise ValueError("path is not authorized by the V87 source lock")
    request = Request(PINNED_RAW_BASE + path, headers={"User-Agent": "simulagent-v87-source-audit"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - exact HTTPS origin is pinned above
        data = response.read(expected_size + 1)
    if len(data) != expected_size:
        raise ValueError(f"unexpected byte size for {path}: {len(data)}")
    actual = git_blob_sha1(data)
    if actual != expected_blob_sha1:
        raise ValueError(f"Git blob mismatch for {path}: {actual}")
    return data


def compile_schema_index(schema_payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(schema_payload, list) or not schema_payload:
        raise ValueError("schema payload must be a non-empty list")
    index: dict[str, dict[str, Any]] = {}
    for service in schema_payload:
        if not isinstance(service, dict):
            raise ValueError("every service schema must be an object")
        name = service.get("service_name")
        slots = service.get("slots")
        intents = service.get("intents")
        if not isinstance(name, str) or not name or name in index:
            raise ValueError("service names must be unique non-empty strings")
        if not isinstance(slots, list) or not isinstance(intents, list):
            raise ValueError(f"service {name} must contain slot and intent lists")
        slot_names = [slot.get("name") for slot in slots if isinstance(slot, dict)]
        intent_names = [intent.get("name") for intent in intents if isinstance(intent, dict)]
        if len(slot_names) != len(slots) or any(not isinstance(item, str) or not item for item in slot_names):
            raise ValueError(f"service {name} has an invalid slot declaration")
        if len(intent_names) != len(intents) or any(not isinstance(item, str) or not item for item in intent_names):
            raise ValueError(f"service {name} has an invalid intent declaration")
        if len(set(slot_names)) != len(slot_names) or len(set(intent_names)) != len(intent_names):
            raise ValueError(f"service {name} contains duplicate typed identifiers")
        index[name] = {
            "slot_names": frozenset(slot_names),
            "intent_names": frozenset(intent_names),
        }
    return index


def build_structural_inventory(
    schema_payload: Any,
    dialogue_payload: Any,
    *,
    excluded_service_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    schema_index = compile_schema_index(schema_payload)
    if not isinstance(dialogue_payload, list):
        raise ValueError("dialogue payload must be a list")

    counters: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    service_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    record_index: list[dict[str, Any]] = []
    dialogue_ids: set[str] = set()

    for dialogue in dialogue_payload:
        counters["dialogue_count"] += 1
        if not isinstance(dialogue, dict):
            raise ValueError("every dialogue must be an object")
        dialogue_id = dialogue.get("dialogue_id")
        turns = dialogue.get("turns")
        if not isinstance(dialogue_id, str) or not dialogue_id or dialogue_id in dialogue_ids:
            raise ValueError("dialogue identifiers must be unique non-empty strings")
        if not isinstance(turns, list):
            raise ValueError(f"dialogue {dialogue_id} has no turn list")
        dialogue_ids.add(dialogue_id)
        counters["turn_count"] += len(turns)

        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"dialogue {dialogue_id} contains a non-object turn")
            if turn.get("speaker") != "USER":
                counters["non_user_turn_count"] += 1
                continue
            counters["user_turn_count"] += 1
            frames = turn.get("frames")
            if not isinstance(frames, list) or len(frames) != 1:
                reason_counts["not_exactly_one_frame"] += 1
                continue
            frame = frames[0]
            if not isinstance(frame, dict):
                reason_counts["frame_not_object"] += 1
                continue
            service = frame.get("service")
            if not isinstance(service, str) or service not in schema_index:
                reason_counts["service_not_in_schema"] += 1
                continue
            if any(service.startswith(prefix) for prefix in excluded_service_prefixes):
                reason_counts["excluded_service_prefix"] += 1
                continue
            state = frame.get("state")
            if not isinstance(state, dict):
                reason_counts["state_missing"] += 1
                continue
            active_intent = state.get("active_intent")
            requested_slots = state.get("requested_slots")
            slot_values = state.get("slot_values")
            if not isinstance(active_intent, str):
                reason_counts["active_intent_invalid"] += 1
                continue
            if active_intent != "NONE" and active_intent not in schema_index[service]["intent_names"]:
                reason_counts["active_intent_not_in_schema"] += 1
                continue
            if not isinstance(requested_slots, list) or any(not isinstance(item, str) for item in requested_slots):
                reason_counts["requested_slots_invalid"] += 1
                continue
            if not isinstance(slot_values, dict):
                reason_counts["slot_values_invalid"] += 1
                continue
            if any(slot not in schema_index[service]["slot_names"] for slot in slot_values):
                reason_counts["slot_not_in_schema"] += 1
                continue
            if any(
                not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values)
                for values in slot_values.values()
            ):
                reason_counts["slot_value_list_invalid"] += 1
                continue

            label_kind = "none" if active_intent == "NONE" else "active"
            record_id = f"{dialogue_id}::turn-{turn_index:03d}::{service}"
            row = {
                "record_id": record_id,
                "dialogue_id": dialogue_id,
                "turn_index": turn_index,
                "service": service,
                "active_intent": active_intent,
                "label_kind": label_kind,
                "requested_slot_count": len(requested_slots),
                "state_slot_keys": sorted(slot_values),
            }
            record_index.append(row)
            counters["eligible_record_count"] += 1
            counters[f"eligible_{label_kind}_record_count"] += 1
            service_counts[service] += 1
            intent_counts[f"{service}::{active_intent}"] += 1

    record_index.sort(key=lambda row: row["record_id"])
    if counters["eligible_record_count"] != len(record_index):
        raise AssertionError("eligible record count drift")
    if any("utterance" in row or "text" in row for row in record_index):
        raise AssertionError("language text leaked into structural inventory")
    return {
        "schema_service_count": len(schema_index),
        "counts": dict(sorted(counters.items())),
        "ineligibility_reason_counts": dict(sorted(reason_counts.items())),
        "eligible_service_counts": dict(sorted(service_counts.items())),
        "eligible_intent_counts": dict(sorted(intent_counts.items())),
        "record_index_sha256": canonical_sha256(record_index),
        "record_index": record_index,
        "contains_utterance_or_text_fields": False,
    }
