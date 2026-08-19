from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from typing import Any


STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
        "from", "get", "have", "i", "in", "is", "it", "me", "my", "of", "on",
        "or", "please", "service", "that", "the", "this", "to", "want", "with",
        "would", "you",
    }
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity


def normalized_tokens(value: str) -> frozenset[str]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value).replace("_", "-")
    tokens = re.findall(r"[a-z0-9]+", expanded.lower())
    return frozenset(token for token in tokens if len(token) > 1 and token not in STOP_WORDS)


def compile_schema(schema_payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(schema_payload, list) or not schema_payload:
        raise ValueError("schema must be a non-empty list")
    compiled: dict[str, dict[str, Any]] = {}
    for service in schema_payload:
        if not isinstance(service, dict):
            raise ValueError("service schema must be an object")
        service_name = service.get("service_name")
        intents = service.get("intents")
        slots = service.get("slots")
        if not isinstance(service_name, str) or not service_name or service_name in compiled:
            raise ValueError("service names must be unique non-empty strings")
        if not isinstance(intents, list) or not isinstance(slots, list):
            raise ValueError(f"service {service_name} requires intent and slot lists")
        intent_index: dict[str, dict[str, Any]] = {}
        for intent in intents:
            if not isinstance(intent, dict):
                raise ValueError(f"service {service_name} has a non-object intent")
            name = intent.get("name")
            description = intent.get("description")
            if not isinstance(name, str) or not name or name in intent_index:
                raise ValueError(f"service {service_name} has invalid intent names")
            if not isinstance(description, str):
                raise ValueError(f"intent {service_name}::{name} has no description")
            intent_index[name] = {
                "surface_tokens": normalized_tokens(f"{name} {description}"),
            }
        slot_names = [slot.get("name") for slot in slots if isinstance(slot, dict)]
        if len(slot_names) != len(slots) or len(slot_names) != len(set(slot_names)):
            raise ValueError(f"service {service_name} has invalid slot names")
        compiled[service_name] = {
            "intent_index": intent_index,
            "intent_names": tuple(intent_index),
            "slot_names": frozenset(slot_names),
        }
    return compiled


def hash_order(salt: str, *parts: str) -> str:
    return hashlib.sha256("::".join((salt, *parts)).encode()).hexdigest()


def _source_records(
    schema: dict[str, dict[str, Any]],
    dialogue_payload: Any,
    excluded_services: frozenset[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if not isinstance(dialogue_payload, list):
        raise ValueError("dialogues must be a list")
    records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen_dialogues: set[str] = set()
    for dialogue in dialogue_payload:
        if not isinstance(dialogue, dict):
            raise ValueError("dialogue must be an object")
        dialogue_id = dialogue.get("dialogue_id")
        turns = dialogue.get("turns")
        if not isinstance(dialogue_id, str) or not dialogue_id or dialogue_id in seen_dialogues:
            raise ValueError("dialogue ids must be unique non-empty strings")
        if not isinstance(turns, list):
            raise ValueError(f"dialogue {dialogue_id} has no turn list")
        seen_dialogues.add(dialogue_id)
        accumulated_user_tokens: set[str] = set()
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"dialogue {dialogue_id} contains a non-object turn")
            if turn.get("speaker") != "USER":
                continue
            utterance = turn.get("utterance")
            if not isinstance(utterance, str):
                reasons["user_utterance_not_string"] += 1
                continue
            accumulated_user_tokens.update(normalized_tokens(utterance))
            frames = turn.get("frames")
            if not isinstance(frames, list) or len(frames) != 1:
                reasons["not_exactly_one_frame"] += 1
                continue
            frame = frames[0]
            if not isinstance(frame, dict):
                reasons["frame_not_object"] += 1
                continue
            service = frame.get("service")
            if not isinstance(service, str) or service not in schema:
                reasons["service_not_in_schema"] += 1
                continue
            if service in excluded_services:
                reasons["excluded_service"] += 1
                continue
            state = frame.get("state")
            if not isinstance(state, dict):
                reasons["state_missing"] += 1
                continue
            active_intent = state.get("active_intent")
            slot_values = state.get("slot_values")
            requested_slots = state.get("requested_slots")
            if not isinstance(active_intent, str):
                reasons["active_intent_invalid"] += 1
                continue
            if active_intent != "NONE" and active_intent not in schema[service]["intent_index"]:
                reasons["active_intent_not_in_schema"] += 1
                continue
            if not isinstance(slot_values, dict) or any(
                slot not in schema[service]["slot_names"] for slot in slot_values
            ):
                reasons["state_slots_invalid"] += 1
                continue
            if not isinstance(requested_slots, list) or any(
                not isinstance(slot, str) for slot in requested_slots
            ):
                reasons["requested_slots_invalid"] += 1
                continue
            records.append(
                {
                    "source_record_id": f"{dialogue_id}::turn-{turn_index:03d}::{service}",
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "source_service": service,
                    "active_intent": active_intent,
                    "history_tokens": frozenset(accumulated_user_tokens),
                }
            )
    return records, reasons


def build_open_set_inventory(
    schema_payload: Any,
    dialogue_payload: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    schema = compile_schema(schema_payload)
    rules = config["classConstruction"]
    excluded = frozenset(config["excludedServices"])
    source_records, reasons = _source_records(schema, dialogue_payload, excluded)
    intent_counts: Counter[tuple[str, str]] = Counter(
        (row["source_service"], row["active_intent"])
        for row in source_records
        if row["active_intent"] != "NONE"
    )

    splits: dict[str, dict[str, Any]] = {}
    for service, declaration in schema.items():
        if service in excluded:
            continue
        intent_names = declaration["intent_names"]
        if len(intent_names) < rules["minimumTypedIntentCountPerService"]:
            continue
        hide_candidates = [
            intent
            for intent in intent_names
            if intent_counts[(service, intent)]
            >= rules["minimumSourceIntentRecordCountForHiding"]
        ]
        if not hide_candidates:
            continue
        hidden = min(
            hide_candidates,
            key=lambda intent: hash_order(rules["hiddenIntentSalt"], service, intent),
        )
        declared = tuple(intent for intent in intent_names if intent != hidden)
        if len(declared) < rules["minimumDeclaredIntentCountPerService"]:
            continue
        splits[service] = {
            "hidden_intent": hidden,
            "declared_intents": list(declared),
            "complete_intents": list(intent_names),
            "hide_candidate_count": len(hide_candidates),
        }

    rows: list[dict[str, Any]] = []
    eligible_source_records = [
        row for row in source_records if row["source_service"] in splits
    ]
    for source in eligible_source_records:
        service = source["source_service"]
        active_intent = source["active_intent"]
        split = splits[service]
        overlap_count = 0
        if active_intent == "NONE":
            class_label = "insufficient_evidence"
        elif active_intent == split["hidden_intent"]:
            class_label = "novel_valid"
            overlap_count = len(
                source["history_tokens"]
                & schema[service]["intent_index"][active_intent]["surface_tokens"]
            )
        else:
            overlap_count = len(
                source["history_tokens"]
                & schema[service]["intent_index"][active_intent]["surface_tokens"]
            )
            class_label = "known_familiar" if overlap_count >= 1 else "known_unfamiliar"
        rows.append(
            {
                "candidate_id": f"{source['source_record_id']}::target::{service}",
                "source_record_id": source["source_record_id"],
                "dialogue_id": source["dialogue_id"],
                "turn_index": source["turn_index"],
                "source_service": service,
                "target_service": service,
                "gold_source_intent": active_intent,
                "class_label": class_label,
                "schema_overlap_count": overlap_count,
            }
        )

        if active_intent == "NONE":
            continue
        target_candidates = [
            target
            for target, target_split in splits.items()
            if target != service and active_intent not in target_split["complete_intents"]
        ]
        if target_candidates:
            target = min(
                target_candidates,
                key=lambda candidate: hash_order(
                    rules["unsupportedTargetSalt"],
                    source["source_record_id"],
                    candidate,
                ),
            )
            rows.append(
                {
                    "candidate_id": f"{source['source_record_id']}::unsupported-target::{target}",
                    "source_record_id": source["source_record_id"],
                    "dialogue_id": source["dialogue_id"],
                    "turn_index": source["turn_index"],
                    "source_service": service,
                    "target_service": target,
                    "gold_source_intent": active_intent,
                    "class_label": "unsupported",
                    "schema_overlap_count": 0,
                }
            )

    rows.sort(key=lambda row: row["candidate_id"])
    if len(rows) != len({row["candidate_id"] for row in rows}):
        raise AssertionError("candidate identifiers are not unique")
    forbidden = {"utterance", "text", "tokens", "history", "slot_values", "values"}
    emitted_keys = set().union(*(row.keys() for row in rows)) if rows else set()
    if emitted_keys & forbidden:
        raise AssertionError("language or state values leaked into inventory")

    class_counts = Counter(row["class_label"] for row in rows)
    class_services: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        class_services[row["class_label"]].add(row["target_service"])
    return {
        "eligible_service_count": len(splits),
        "service_splits": dict(sorted(splits.items())),
        "source_record_count": len(source_records),
        "eligible_source_record_count": len(eligible_source_records),
        "candidate_count": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "class_service_counts": {
            label: len(services) for label, services in sorted(class_services.items())
        },
        "ineligibility_reason_counts": dict(sorted(reasons.items())),
        "candidate_index_sha256": canonical_sha256(rows),
        "candidate_index": rows,
        "contains_language_tokens_slot_values_or_histories": False,
    }
