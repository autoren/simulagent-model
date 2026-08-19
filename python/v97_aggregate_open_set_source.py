from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from v93_open_set_source import canonical_sha256, compile_schema, git_blob_sha1, hash_order
from v95_activation_open_set_source import _activation_source_records


def _pooled_records(
    schema: dict[str, dict[str, Any]],
    shard_payloads: list[tuple[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, int]]:
    records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    shard_record_counts: dict[str, int] = {}
    seen_shards: set[str] = set()
    for shard_path, payload in shard_payloads:
        if not isinstance(shard_path, str) or not shard_path or shard_path in seen_shards:
            raise ValueError("shard paths must be unique non-empty strings")
        seen_shards.add(shard_path)
        shard_records, shard_reasons = _activation_source_records(schema, payload)
        shard_record_counts[shard_path] = len(shard_records)
        reasons.update(shard_reasons)
        for row in shard_records:
            records.append({**row, "source_shard": shard_path})
    return records, reasons, shard_record_counts


def build_aggregate_open_set_inventory(
    schema_payload: Any,
    shard_payloads: list[tuple[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    schema = compile_schema(schema_payload)
    records, reasons, shard_record_counts = _pooled_records(schema, shard_payloads)
    exposed = frozenset(config["previouslyExposedServices"])
    partition = config["servicePartition"]
    activation_records = [row for row in records if row["is_intent_activation"]]
    service_activation_counts = Counter(row["source_service"] for row in activation_records)
    pair_activation_counts = Counter(
        (row["source_service"], row["active_intent"]) for row in activation_records
    )
    eligible_pairs_by_service: dict[str, tuple[str, ...]] = {}
    for service in schema:
        pairs = tuple(
            intent for intent in schema[service]["intent_names"]
            if pair_activation_counts[(service, intent)] >= partition["eligiblePairMinimumActivationCount"]
        )
        if pairs:
            eligible_pairs_by_service[service] = pairs
    eligible_services = [
        service for service in schema
        if service not in exposed
        and service_activation_counts[service] >= partition["eligibleServiceMinimumActivationCount"]
        and service in eligible_pairs_by_service
    ]
    unsupported_order = sorted(
        eligible_services,
        key=lambda service: hash_order(partition["unsupportedServiceSalt"], service),
    )
    unsupported_services = tuple(unsupported_order[: partition["unsupportedServiceCount"]])
    catalog_order = sorted(
        (service for service in eligible_services if service not in unsupported_services),
        key=lambda service: hash_order(partition["catalogServiceSalt"], service),
    )
    catalog_services = tuple(catalog_order[: partition["catalogServiceCount"]])

    hidden_service_order = sorted(
        catalog_services,
        key=lambda service: hash_order(partition["hiddenServiceSalt"], service),
    )
    hidden_services = tuple(hidden_service_order[: partition["hiddenServiceCount"]])
    hidden_pairs: set[tuple[str, str]] = set()
    for service in hidden_services:
        pair_order = sorted(
            eligible_pairs_by_service[service],
            key=lambda intent: hash_order(partition["hiddenIntentPairSalt"], service, intent),
        )
        for intent in pair_order[: partition["hiddenPairCountPerSelectedService"]]:
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
                "candidate_id": f"{row['source_shard']}::{row['source_record_id']}::aggregate-catalog",
                "source_shard": row["source_shard"],
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
        raise AssertionError("aggregate candidate identifiers are not unique")
    forbidden = {"utterance", "text", "tokens", "history", "slot_values", "values"}
    keys = set().union(*(row.keys() for row in candidates)) if candidates else set()
    if keys & forbidden:
        raise AssertionError("language or values leaked into aggregate inventory")
    if set(catalog_services) & set(unsupported_services):
        raise AssertionError("catalog and unsupported services overlap")
    if any(
        not row["source_intent_activation"]
        for row in candidates if row["class_label"] != "insufficient_evidence"
    ):
        raise AssertionError("non-NONE class contains a continuation turn")

    class_counts = Counter(row["class_label"] for row in candidates)
    class_services: dict[str, set[str]] = defaultdict(set)
    class_dialogues: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in candidates:
        class_services[row["class_label"]].add(row["source_service"])
        class_dialogues[row["class_label"]].add((row["source_shard"], row["dialogue_id"]))
    return {
        "aggregate_shard_count": len(shard_payloads),
        "aggregate_source_record_count": len(records),
        "aggregate_source_intent_activation_count": len(activation_records),
        "shard_source_record_counts": dict(sorted(shard_record_counts.items())),
        "eligible_fresh_services": sorted(eligible_services),
        "eligible_fresh_service_count": len(eligible_services),
        "catalog_services": list(catalog_services),
        "catalog_service_count": len(catalog_services),
        "unsupported_services": list(unsupported_services),
        "unsupported_service_count": len(unsupported_services),
        "supported_pairs": [f"{service}::{intent}" for service, intent in sorted(supported_pairs)],
        "supported_pair_count": len(supported_pairs),
        "hidden_services": list(hidden_services),
        "hidden_pairs": [f"{service}::{intent}" for service, intent in sorted(hidden_pairs)],
        "hidden_pair_count": len(hidden_pairs),
        "declared_supported_pairs": [
            f"{service}::{intent}" for service, intent in sorted(declared_supported_pairs)
        ],
        "declared_supported_pair_count": len(declared_supported_pairs),
        "complete_catalog_pair_count": len(complete_catalog_pairs),
        "candidate_count": len(candidates),
        "class_counts": dict(sorted(class_counts.items())),
        "class_service_counts": {
            label: len(services) for label, services in sorted(class_services.items())
        },
        "class_dialogue_counts": {
            label: len(dialogues) for label, dialogues in sorted(class_dialogues.items())
        },
        "ineligibility_reason_counts": dict(sorted(reasons.items())),
        "candidate_index_sha256": canonical_sha256(candidates),
        "candidate_index": candidates,
        "service_roles_are_disjoint": True,
        "all_non_none_candidates_are_source_intent_activations": True,
        "lexical_separation_uses_current_turn_only": True,
        "contains_language_tokens_slot_values_or_histories": False,
    }


__all__ = ["build_aggregate_open_set_inventory", "git_blob_sha1"]
