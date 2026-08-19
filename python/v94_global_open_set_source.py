from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from v93_open_set_source import (
    _source_records,
    canonical_sha256,
    compile_schema,
    git_blob_sha1,
    hash_order,
)


def build_global_catalog_inventory(
    schema_payload: Any,
    dialogue_payload: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    schema = compile_schema(schema_payload)
    records, reasons = _source_records(schema, dialogue_payload, frozenset())
    exposed = frozenset(config["previouslyExposedServices"])
    partition = config["catalogPartition"]
    service_active_counts = Counter(
        row["source_service"] for row in records if row["active_intent"] != "NONE"
    )
    pair_active_counts = Counter(
        (row["source_service"], row["active_intent"])
        for row in records if row["active_intent"] != "NONE"
    )
    eligible_services = sorted(
        (
            service
            for service in schema
            if service not in exposed
            and service_active_counts[service] >= partition["eligibleServiceMinimumActiveRecordCount"]
            and any(
                pair_active_counts[(service, intent)] >= partition["eligiblePairMinimumActiveRecordCount"]
                for intent in schema[service]["intent_names"]
            )
        ),
        key=lambda service: hash_order(partition["unsupportedServiceSalt"], service),
    )
    unsupported_count = max(
        partition["minimumUnsupportedServiceCount"],
        len(eligible_services)
        * partition["unsupportedServiceFractionNumerator"]
        // partition["unsupportedServiceFractionDenominator"],
    )
    maximum_unsupported = max(0, len(eligible_services) - partition["minimumCatalogServiceCount"])
    unsupported_count = min(unsupported_count, maximum_unsupported)
    unsupported_services = tuple(eligible_services[:unsupported_count])
    catalog_services = tuple(eligible_services[unsupported_count:])

    supported_pairs = [
        (service, intent)
        for service in catalog_services
        for intent in schema[service]["intent_names"]
        if pair_active_counts[(service, intent)] >= partition["eligiblePairMinimumActiveRecordCount"]
    ]
    supported_pairs.sort(
        key=lambda pair: hash_order(partition["hiddenIntentPairSalt"], pair[0], pair[1])
    )
    hidden_count = max(
        partition["minimumHiddenPairCount"],
        len(supported_pairs)
        * partition["hiddenPairFractionNumerator"]
        // partition["hiddenPairFractionDenominator"],
    )
    maximum_hidden = max(0, len(supported_pairs) - partition["minimumDeclaredPairCount"])
    hidden_count = min(hidden_count, maximum_hidden)
    hidden_pairs = frozenset(supported_pairs[:hidden_count])
    declared_supported_pairs = frozenset(supported_pairs[hidden_count:])
    complete_catalog_pairs = frozenset(
        (service, intent)
        for service in catalog_services
        for intent in schema[service]["intent_names"]
    )
    declared_catalog_pairs = complete_catalog_pairs - hidden_pairs

    candidates: list[dict[str, Any]] = []
    for row in records:
        service = row["source_service"]
        intent = row["active_intent"]
        if service in unsupported_services:
            if intent == "NONE":
                continue
            class_label = "unsupported"
            overlap_count = 0
        elif service in catalog_services:
            if intent == "NONE":
                class_label = "insufficient_evidence"
                overlap_count = 0
            else:
                pair = (service, intent)
                overlap_count = len(
                    row["history_tokens"]
                    & schema[service]["intent_index"][intent]["surface_tokens"]
                )
                if pair in hidden_pairs:
                    class_label = "novel_valid"
                else:
                    class_label = "known_familiar" if overlap_count >= 1 else "known_unfamiliar"
        else:
            continue
        candidates.append(
            {
                "candidate_id": f"{row['source_record_id']}::global-catalog",
                "source_record_id": row["source_record_id"],
                "dialogue_id": row["dialogue_id"],
                "turn_index": row["turn_index"],
                "source_service": service,
                "gold_source_intent": intent,
                "class_label": class_label,
                "schema_overlap_count": overlap_count,
            }
        )
    candidates.sort(key=lambda row: row["candidate_id"])
    if len(candidates) != len({row["candidate_id"] for row in candidates}):
        raise AssertionError("global candidate identifiers are not unique")
    forbidden = {"utterance", "text", "tokens", "history", "slot_values", "values"}
    keys = set().union(*(row.keys() for row in candidates)) if candidates else set()
    if keys & forbidden:
        raise AssertionError("language or values leaked into global inventory")

    class_counts = Counter(row["class_label"] for row in candidates)
    class_services: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        class_services[row["class_label"]].add(row["source_service"])
    return {
        "source_record_count": len(records),
        "eligible_fresh_services": list(eligible_services),
        "eligible_fresh_service_count": len(eligible_services),
        "catalog_services": list(catalog_services),
        "catalog_service_count": len(catalog_services),
        "unsupported_services": list(unsupported_services),
        "unsupported_service_count": len(unsupported_services),
        "supported_pair_count": len(supported_pairs),
        "hidden_pairs": [f"{service}::{intent}" for service, intent in sorted(hidden_pairs)],
        "hidden_pair_count": len(hidden_pairs),
        "declared_supported_pairs": [
            f"{service}::{intent}" for service, intent in sorted(declared_supported_pairs)
        ],
        "declared_supported_pair_count": len(declared_supported_pairs),
        "declared_catalog_pairs": [
            f"{service}::{intent}" for service, intent in sorted(declared_catalog_pairs)
        ],
        "complete_catalog_pair_count": len(complete_catalog_pairs),
        "candidate_count": len(candidates),
        "class_counts": dict(sorted(class_counts.items())),
        "class_service_counts": {
            label: len(services) for label, services in sorted(class_services.items())
        },
        "ineligibility_reason_counts": dict(sorted(reasons.items())),
        "candidate_index_sha256": canonical_sha256(candidates),
        "candidate_index": candidates,
        "contains_language_tokens_slot_values_or_histories": False,
    }


__all__ = ["build_global_catalog_inventory", "git_blob_sha1"]
