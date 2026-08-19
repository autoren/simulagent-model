from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from v93_open_set_source import (
    canonical_sha256,
    compile_schema,
    git_blob_sha1,
    hash_order,
    normalized_tokens,
)


def _activation_source_records(
    schema: dict[str, dict[str, Any]],
    dialogue_payload: Any,
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
        previous_intent: dict[str, str] = defaultdict(lambda: "NONE")
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"dialogue {dialogue_id} contains a non-object turn")
            if turn.get("speaker") != "USER":
                continue
            utterance = turn.get("utterance")
            if not isinstance(utterance, str):
                reasons["user_utterance_not_string"] += 1
                continue
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
            prior = previous_intent[service]
            is_activation = active_intent != "NONE" and active_intent != prior
            previous_intent[service] = active_intent
            records.append(
                {
                    "source_record_id": f"{dialogue_id}::turn-{turn_index:03d}::{service}",
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "source_service": service,
                    "active_intent": active_intent,
                    "is_intent_activation": is_activation,
                    "current_turn_tokens": normalized_tokens(utterance),
                }
            )
    return records, reasons


def build_activation_open_set_inventory(
    schema_payload: Any,
    dialogue_payload: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    schema = compile_schema(schema_payload)
    records, reasons = _activation_source_records(schema, dialogue_payload)
    exposed = frozenset(config["previouslyExposedServices"])
    partition = config["catalogPartition"]
    activation_records = [row for row in records if row["is_intent_activation"]]
    service_activation_counts = Counter(row["source_service"] for row in activation_records)
    pair_activation_counts = Counter(
        (row["source_service"], row["active_intent"]) for row in activation_records
    )
    eligible_pairs_by_service: dict[str, tuple[str, ...]] = {}
    for service in schema:
        pairs = tuple(
            intent
            for intent in schema[service]["intent_names"]
            if pair_activation_counts[(service, intent)]
            >= partition["eligiblePairMinimumActivationCount"]
        )
        if pairs:
            eligible_pairs_by_service[service] = pairs
    eligible_services = sorted(
        (
            service
            for service in schema
            if service not in exposed
            and service_activation_counts[service]
            >= partition["eligibleServiceMinimumActivationCount"]
            and service in eligible_pairs_by_service
        ),
        key=lambda service: hash_order(partition["unsupportedServiceSalt"], service),
    )
    unsupported_count = min(
        partition["unsupportedServiceCount"],
        max(0, len(eligible_services) - partition["minimumCatalogServiceCount"]),
    )
    unsupported_services = tuple(eligible_services[:unsupported_count])
    catalog_services = tuple(eligible_services[unsupported_count:])

    hidden_service_candidates = sorted(
        catalog_services,
        key=lambda service: hash_order(partition["hiddenServiceSalt"], service),
    )
    hidden_services = tuple(hidden_service_candidates[: partition["hiddenServiceCount"]])
    hidden_pairs: set[tuple[str, str]] = set()
    for service in hidden_services:
        pair_candidates = sorted(
            eligible_pairs_by_service[service],
            key=lambda intent: hash_order(partition["hiddenIntentPairSalt"], service, intent),
        )
        for intent in pair_candidates[: partition["hiddenPairCountPerSelectedService"]]:
            hidden_pairs.add((service, intent))
    supported_pairs = frozenset(
        (service, intent)
        for service in catalog_services
        for intent in eligible_pairs_by_service.get(service, ())
    )
    declared_supported_pairs = supported_pairs - hidden_pairs
    complete_catalog_pairs = frozenset(
        (service, intent)
        for service in catalog_services
        for intent in schema[service]["intent_names"]
    )

    candidates: list[dict[str, Any]] = []
    for row in records:
        service = row["source_service"]
        intent = row["active_intent"]
        overlap_count = 0
        if service in unsupported_services:
            if not row["is_intent_activation"]:
                continue
            class_label = "unsupported"
        elif service in catalog_services:
            if intent == "NONE":
                class_label = "insufficient_evidence"
            elif not row["is_intent_activation"]:
                continue
            elif (service, intent) in hidden_pairs:
                class_label = "novel_valid"
                overlap_count = len(
                    row["current_turn_tokens"]
                    & schema[service]["intent_index"][intent]["surface_tokens"]
                )
            elif (service, intent) in declared_supported_pairs:
                overlap_count = len(
                    row["current_turn_tokens"]
                    & schema[service]["intent_index"][intent]["surface_tokens"]
                )
                class_label = "known_familiar" if overlap_count >= 1 else "known_unfamiliar"
            else:
                continue
        else:
            continue
        candidates.append(
            {
                "candidate_id": f"{row['source_record_id']}::activation-catalog",
                "source_record_id": row["source_record_id"],
                "dialogue_id": row["dialogue_id"],
                "turn_index": row["turn_index"],
                "source_service": service,
                "gold_source_intent": intent,
                "source_intent_activation": row["is_intent_activation"],
                "class_label": class_label,
                "current_turn_schema_overlap_count": overlap_count,
            }
        )
    candidates.sort(key=lambda row: row["candidate_id"])
    if len(candidates) != len({row["candidate_id"] for row in candidates}):
        raise AssertionError("activation candidate identifiers are not unique")
    forbidden = {"utterance", "text", "tokens", "history", "slot_values", "values"}
    keys = set().union(*(row.keys() for row in candidates)) if candidates else set()
    if keys & forbidden:
        raise AssertionError("language or values leaked into activation inventory")
    if any(
        not row["source_intent_activation"]
        for row in candidates
        if row["class_label"] != "insufficient_evidence"
    ):
        raise AssertionError("non-NONE class contains a continuation turn")

    class_counts = Counter(row["class_label"] for row in candidates)
    class_services: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        class_services[row["class_label"]].add(row["source_service"])
    return {
        "source_record_count": len(records),
        "source_intent_activation_count": len(activation_records),
        "eligible_fresh_services": list(eligible_services),
        "eligible_fresh_service_count": len(eligible_services),
        "catalog_services": list(catalog_services),
        "catalog_service_count": len(catalog_services),
        "unsupported_services": list(unsupported_services),
        "unsupported_service_count": len(unsupported_services),
        "supported_pairs": [f"{s}::{i}" for s, i in sorted(supported_pairs)],
        "supported_pair_count": len(supported_pairs),
        "hidden_services": list(hidden_services),
        "hidden_pairs": [f"{s}::{i}" for s, i in sorted(hidden_pairs)],
        "hidden_pair_count": len(hidden_pairs),
        "declared_supported_pairs": [
            f"{s}::{i}" for s, i in sorted(declared_supported_pairs)
        ],
        "declared_supported_pair_count": len(declared_supported_pairs),
        "complete_catalog_pair_count": len(complete_catalog_pairs),
        "candidate_count": len(candidates),
        "class_counts": dict(sorted(class_counts.items())),
        "class_service_counts": {
            label: len(services) for label, services in sorted(class_services.items())
        },
        "ineligibility_reason_counts": dict(sorted(reasons.items())),
        "candidate_index_sha256": canonical_sha256(candidates),
        "candidate_index": candidates,
        "all_non_none_candidates_are_source_intent_activations": True,
        "lexical_separation_uses_current_turn_only": True,
        "contains_language_tokens_slot_values_or_histories": False,
    }


__all__ = ["build_activation_open_set_inventory", "git_blob_sha1"]
