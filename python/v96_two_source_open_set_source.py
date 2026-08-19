from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from v93_open_set_source import canonical_sha256, compile_schema, git_blob_sha1, hash_order
from v95_activation_open_set_source import _activation_source_records


def _eligible_pairs(
    schema: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    minimum_service_count: int,
    minimum_pair_count: int,
    excluded: frozenset[str],
) -> tuple[list[str], dict[str, tuple[str, ...]], Counter[str], Counter[tuple[str, str]]]:
    activations = [row for row in records if row["is_intent_activation"]]
    service_counts = Counter(row["source_service"] for row in activations)
    pair_counts = Counter((row["source_service"], row["active_intent"]) for row in activations)
    pairs_by_service: dict[str, tuple[str, ...]] = {}
    for service in schema:
        pairs = tuple(
            intent for intent in schema[service]["intent_names"]
            if pair_counts[(service, intent)] >= minimum_pair_count
        )
        if pairs:
            pairs_by_service[service] = pairs
    services = [
        service for service in schema
        if service not in excluded
        and service_counts[service] >= minimum_service_count
        and service in pairs_by_service
    ]
    return services, pairs_by_service, service_counts, pair_counts


def build_two_source_open_set_inventory(
    schema_payload: Any,
    catalog_dialogue_payload: Any,
    unsupported_dialogue_payload: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    schema = compile_schema(schema_payload)
    catalog_records, catalog_reasons = _activation_source_records(schema, catalog_dialogue_payload)
    unsupported_records, unsupported_reasons = _activation_source_records(schema, unsupported_dialogue_payload)
    exposed = frozenset(config["previouslyExposedServices"])
    catalog_partition = config["catalogPartition"]
    unsupported_partition = config["unsupportedPartition"]

    catalog_eligible, catalog_pairs_by_service, _, _ = _eligible_pairs(
        schema,
        catalog_records,
        catalog_partition["eligibleServiceMinimumActivationCount"],
        catalog_partition["eligiblePairMinimumActivationCount"],
        exposed,
    )
    catalog_eligible.sort(key=lambda service: hash_order(catalog_partition["catalogServiceSalt"], service))
    catalog_services = tuple(catalog_eligible[: catalog_partition["catalogServiceCount"]])

    hidden_service_candidates = sorted(
        catalog_services,
        key=lambda service: hash_order(catalog_partition["hiddenServiceSalt"], service),
    )
    hidden_services = tuple(hidden_service_candidates[: catalog_partition["hiddenServiceCount"]])
    hidden_pairs: set[tuple[str, str]] = set()
    for service in hidden_services:
        pair_candidates = sorted(
            catalog_pairs_by_service[service],
            key=lambda intent: hash_order(catalog_partition["hiddenIntentPairSalt"], service, intent),
        )
        for intent in pair_candidates[: catalog_partition["hiddenPairCountPerSelectedService"]]:
            hidden_pairs.add((service, intent))
    supported_pairs = frozenset(
        (service, intent)
        for service in catalog_services
        for intent in catalog_pairs_by_service.get(service, ())
    )
    declared_supported_pairs = supported_pairs - hidden_pairs
    complete_catalog_pairs = frozenset(
        (service, intent)
        for service in catalog_services
        for intent in schema[service]["intent_names"]
    )

    unsupported_excluded = exposed | frozenset(catalog_services)
    unsupported_eligible, unsupported_pairs_by_service, _, _ = _eligible_pairs(
        schema,
        unsupported_records,
        unsupported_partition["eligibleServiceMinimumActivationCount"],
        unsupported_partition["eligiblePairMinimumActivationCount"],
        unsupported_excluded,
    )
    unsupported_eligible.sort(
        key=lambda service: hash_order(unsupported_partition["unsupportedServiceSalt"], service)
    )
    unsupported_services = tuple(
        unsupported_eligible[: unsupported_partition["unsupportedServiceCount"]]
    )

    candidates: list[dict[str, Any]] = []
    for row in catalog_records:
        service = row["source_service"]
        intent = row["active_intent"]
        if service not in catalog_services:
            continue
        overlap_count = 0
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
        candidates.append(
            {
                "candidate_id": f"catalog::{row['source_record_id']}",
                "source_role": "catalog",
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
    for row in unsupported_records:
        if row["source_service"] not in unsupported_services or not row["is_intent_activation"]:
            continue
        candidates.append(
            {
                "candidate_id": f"unsupported::{row['source_record_id']}",
                "source_role": "unsupported",
                "source_record_id": row["source_record_id"],
                "dialogue_id": row["dialogue_id"],
                "turn_index": row["turn_index"],
                "source_service": row["source_service"],
                "gold_source_intent": row["active_intent"],
                "source_intent_activation": True,
                "class_label": "unsupported",
                "current_turn_schema_overlap_count": 0,
            }
        )
    candidates.sort(key=lambda row: row["candidate_id"])
    if len(candidates) != len({row["candidate_id"] for row in candidates}):
        raise AssertionError("two-source candidate identifiers are not unique")
    forbidden = {"utterance", "text", "tokens", "history", "slot_values", "values"}
    keys = set().union(*(row.keys() for row in candidates)) if candidates else set()
    if keys & forbidden:
        raise AssertionError("language or values leaked into two-source inventory")
    if set(catalog_services) & set(unsupported_services):
        raise AssertionError("catalog and unsupported services overlap")
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
        "catalog_source_record_count": len(catalog_records),
        "catalog_source_intent_activation_count": sum(row["is_intent_activation"] for row in catalog_records),
        "unsupported_source_record_count": len(unsupported_records),
        "unsupported_source_intent_activation_count": sum(row["is_intent_activation"] for row in unsupported_records),
        "eligible_catalog_services": list(catalog_eligible),
        "eligible_catalog_service_count": len(catalog_eligible),
        "catalog_services": list(catalog_services),
        "catalog_service_count": len(catalog_services),
        "eligible_unsupported_services": list(unsupported_eligible),
        "eligible_unsupported_service_count": len(unsupported_eligible),
        "unsupported_services": list(unsupported_services),
        "unsupported_service_count": len(unsupported_services),
        "unsupported_supported_pairs": [
            f"{service}::{intent}"
            for service in unsupported_services
            for intent in unsupported_pairs_by_service.get(service, ())
        ],
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
        "catalog_ineligibility_reason_counts": dict(sorted(catalog_reasons.items())),
        "unsupported_ineligibility_reason_counts": dict(sorted(unsupported_reasons.items())),
        "candidate_index_sha256": canonical_sha256(candidates),
        "candidate_index": candidates,
        "source_roles_are_disjoint": True,
        "all_non_none_candidates_are_source_intent_activations": True,
        "lexical_separation_uses_current_turn_only": True,
        "contains_language_tokens_slot_values_or_histories": False,
    }


__all__ = ["build_two_source_open_set_inventory", "git_blob_sha1"]
