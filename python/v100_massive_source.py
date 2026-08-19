from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import re
import tarfile
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order, normalized_tokens


SLOT_PATTERN = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*[^\[\]]*?\]")


def parse_massive_archive(data: bytes, expected_suffix: str) -> tuple[list[dict[str, Any]], str]:
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as archive:
        members = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.endswith(expected_suffix)
        ]
        if len(members) != 1:
            raise ValueError(f"expected one {expected_suffix} member, found {len(members)}")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise ValueError("MASSIVE locale member is unreadable")
        records = []
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid MASSIVE JSONL line {line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"MASSIVE JSONL line {line_number} is not an object")
            records.append(value)
    return records, members[0].name


def slot_types(annotated_utterance: str) -> frozenset[str]:
    if not isinstance(annotated_utterance, str):
        raise ValueError("annotated utterance must be a string")
    return frozenset(match.group(1).strip() for match in SLOT_PATTERN.finditer(annotated_utterance))


def _validated_records(records: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("MASSIVE records must be a non-empty list")
    required = frozenset(config["requiredRecordFields"])
    allowed_partitions = frozenset(config["allowedSourcePartitions"])
    partition_map = config["canonicalPartitionMap"]
    locale = config["locale"]
    validated = []
    seen: set[str] = set()
    for row in records:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError("MASSIVE record lacks required fields")
        identifier = str(row["id"])
        if not identifier or identifier in seen:
            raise ValueError("MASSIVE record identifiers must be unique")
        if row["locale"] != locale or row["partition"] not in allowed_partitions:
            raise ValueError("MASSIVE locale or partition mismatch")
        if not all(isinstance(row[key], str) and row[key] for key in ("scenario", "intent", "utt", "annot_utt")):
            raise ValueError("MASSIVE structural or utterance field is invalid")
        seen.add(identifier)
        validated.append({
            "id": identifier,
            "partition": partition_map[row["partition"]],
            "scenario": row["scenario"],
            "intent": row["intent"],
            "utterance_tokens": normalized_tokens(row["utt"]),
            "slot_types": slot_types(row["annot_utt"]),
        })
    return validated


def build_massive_source_inventory(records: Any, config: dict[str, Any]) -> dict[str, Any]:
    validated = _validated_records(records, config)
    partition = config["servicePartition"]
    scenario_counts = Counter(row["scenario"] for row in validated)
    intent_counts = Counter((row["scenario"], row["intent"]) for row in validated)
    intents_by_scenario: dict[str, set[str]] = defaultdict(set)
    global_intents: set[str] = set()
    global_slots: set[str] = set()
    for row in validated:
        intents_by_scenario[row["scenario"]].add(row["intent"])
        global_intents.add(row["intent"])
        global_slots.update(row["slot_types"])
    eligible_intents_by_scenario = {
        scenario: tuple(sorted(
            intent for intent in intents
            if intent_counts[(scenario, intent)] >= partition["eligibleIntentMinimumRecordCount"]
        ))
        for scenario, intents in intents_by_scenario.items()
    }
    eligible_scenarios = [
        scenario for scenario in intents_by_scenario
        if scenario_counts[scenario] >= partition["eligibleScenarioMinimumRecordCount"]
        and len(eligible_intents_by_scenario[scenario]) >= 2
    ]
    unsupported_order = sorted(
        eligible_scenarios,
        key=lambda scenario: hash_order(partition["unsupportedScenarioSalt"], scenario),
    )
    unsupported_scenarios = tuple(
        unsupported_order[: partition["unsupportedScenarioCount"]]
    )
    catalog_order = sorted(
        (scenario for scenario in eligible_scenarios if scenario not in unsupported_scenarios),
        key=lambda scenario: hash_order(partition["catalogScenarioSalt"], scenario),
    )
    catalog_scenarios = tuple(catalog_order[: partition["catalogScenarioCount"]])
    hidden_scenario_order = sorted(
        catalog_scenarios,
        key=lambda scenario: hash_order(partition["hiddenScenarioSalt"], scenario),
    )
    hidden_scenarios = tuple(hidden_scenario_order[: partition["hiddenScenarioCount"]])
    hidden_intents: set[tuple[str, str]] = set()
    for scenario in hidden_scenarios:
        intent_order = sorted(
            eligible_intents_by_scenario[scenario],
            key=lambda intent: hash_order(partition["hiddenIntentSalt"], scenario, intent),
        )
        for intent in intent_order[: partition["hiddenIntentCountPerSelectedScenario"]]:
            hidden_intents.add((scenario, intent))
    supported_intents = frozenset(
        (scenario, intent)
        for scenario in catalog_scenarios
        for intent in eligible_intents_by_scenario.get(scenario, ())
    )
    declared_intents = supported_intents - hidden_intents

    candidates = []
    for row in validated:
        scenario = row["scenario"]
        intent = row["intent"]
        overlap_count = 0
        if scenario in unsupported_scenarios:
            class_label = "unsupported"
        elif scenario in catalog_scenarios and (scenario, intent) in hidden_intents:
            class_label = "novel_valid"
            overlap_count = len(row["utterance_tokens"] & normalized_tokens(intent))
        elif scenario in catalog_scenarios and (scenario, intent) in declared_intents:
            overlap_count = len(row["utterance_tokens"] & normalized_tokens(intent))
            class_label = "known_familiar" if overlap_count >= 1 else "known_unfamiliar"
        else:
            continue
        candidates.append({
            "candidate_id": f"massive::{row['id']}",
            "source_id": row["id"],
            "partition": row["partition"],
            "scenario": scenario,
            "intent": intent,
            "class_label": class_label,
            "current_utterance_intent_overlap_count": overlap_count,
            "slot_type_count": len(row["slot_types"]),
        })
    candidates.sort(key=lambda row: row["candidate_id"])
    forbidden = {"utt", "utterance", "annot_utt", "tokens", "slot_values", "values", "text"}
    keys = set().union(*(row.keys() for row in candidates)) if candidates else set()
    if keys & forbidden:
        raise AssertionError("MASSIVE language or values leaked into inventory")
    class_counts = Counter(row["class_label"] for row in candidates)
    class_scenarios: dict[str, set[str]] = defaultdict(set)
    class_partitions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in candidates:
        class_scenarios[row["class_label"]].add(row["scenario"])
        class_partitions[row["class_label"]][row["partition"]] += 1
    return {
        "source_record_count": len(validated),
        "partition_counts": dict(sorted(Counter(row["partition"] for row in validated).items())),
        "scenario_count": len(intents_by_scenario),
        "intent_count": len(global_intents),
        "slot_type_count": len(global_slots),
        "eligible_scenarios": sorted(eligible_scenarios),
        "eligible_scenario_count": len(eligible_scenarios),
        "catalog_scenarios": list(catalog_scenarios),
        "catalog_scenario_count": len(catalog_scenarios),
        "unsupported_scenarios": list(unsupported_scenarios),
        "unsupported_scenario_count": len(unsupported_scenarios),
        "hidden_scenarios": list(hidden_scenarios),
        "hidden_intents": [f"{scenario}::{intent}" for scenario, intent in sorted(hidden_intents)],
        "hidden_intent_count": len(hidden_intents),
        "declared_intents": [f"{scenario}::{intent}" for scenario, intent in sorted(declared_intents)],
        "declared_intent_count": len(declared_intents),
        "candidate_count": len(candidates),
        "class_counts": dict(sorted(class_counts.items())),
        "class_scenario_counts": {
            label: len(scenarios) for label, scenarios in sorted(class_scenarios.items())
        },
        "class_partition_counts": {
            label: dict(sorted(counts.items())) for label, counts in sorted(class_partitions.items())
        },
        "candidate_index_sha256": canonical_sha256(candidates),
        "candidate_index": candidates,
        "roles_selected_before_utterance_features": True,
        "contains_raw_or_annotated_utterances_tokens_or_slot_values": False,
    }


def evaluate_massive_source_gates(
    inventory: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    gates = config["sourceGates"]
    counts = inventory["class_counts"]
    coverage = inventory["class_scenario_counts"]
    partitions = inventory["class_partition_counts"]
    required_classes = (
        "known_familiar", "known_unfamiliar", "novel_valid", "unsupported",
    )
    checks: dict[str, bool] = {
        "scenario_count": inventory["scenario_count"] >= gates["minimumScenarioCount"],
        "intent_count": inventory["intent_count"] >= gates["minimumIntentCount"],
        "slot_type_count": inventory["slot_type_count"] >= gates["minimumSlotTypeCount"],
        "eligible_scenario_count": (
            inventory["eligible_scenario_count"] >= gates["minimumEligibleScenarioCount"]
        ),
        "catalog_scenario_count": (
            inventory["catalog_scenario_count"] == gates["requiredCatalogScenarioCount"]
        ),
        "unsupported_scenario_count": (
            inventory["unsupported_scenario_count"] == gates["requiredUnsupportedScenarioCount"]
        ),
        "hidden_intent_count": (
            inventory["hidden_intent_count"] == gates["requiredHiddenIntentCount"]
        ),
        "declared_intent_count": (
            inventory["declared_intent_count"] >= gates["minimumDeclaredIntentCount"]
        ),
        "known_familiar_scenario_coverage": (
            coverage.get("known_familiar", 0) >= gates["minimumKnownClassScenarioCoverage"]
        ),
        "known_unfamiliar_scenario_coverage": (
            coverage.get("known_unfamiliar", 0) >= gates["minimumKnownClassScenarioCoverage"]
        ),
        "novel_scenario_coverage": (
            coverage.get("novel_valid", 0) == gates["requiredNovelScenarioCoverage"]
        ),
        "unsupported_scenario_coverage": (
            coverage.get("unsupported", 0) == gates["requiredUnsupportedScenarioCoverage"]
        ),
        "roles_selected_before_utterance_features": inventory["roles_selected_before_utterance_features"],
        "text_free_inventory": not inventory["contains_raw_or_annotated_utterances_tokens_or_slot_values"],
    }
    for label in required_classes:
        checks[f"{label}_candidate_count"] = (
            counts.get(label, 0) >= gates["minimumClassCandidateCount"]
        )
        checks[f"{label}_validation_count"] = (
            partitions.get(label, {}).get("validation", 0)
            >= gates["minimumValidationCandidateCountPerClass"]
        )
        checks[f"{label}_test_count"] = (
            partitions.get(label, {}).get("test", 0)
            >= gates["minimumTestCandidateCountPerClass"]
        )
    return checks


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "build_massive_source_inventory", "evaluate_massive_source_gates",
    "parse_massive_archive", "sha256_bytes", "slot_types",
]
