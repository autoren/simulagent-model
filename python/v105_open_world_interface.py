from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

from v104_massive_language_extraction import parse_annotated_slots
from v93_open_set_source import canonical_sha256, hash_order


def compile_visible_catalog(
    source_records: Any, source_inventory: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("MASSIVE source records are missing")
    declared = frozenset(config["visibleDeclaredIntents"])
    expected_declared = frozenset(source_inventory["declared_intents"])
    if declared != expected_declared:
        raise ValueError("visible declared intent set differs from frozen inventory")
    slot_types: dict[str, set[str]] = defaultdict(set)
    train_counts: dict[str, int] = defaultdict(int)
    for row in source_records:
        pair = f"{row['scenario']}::{row['intent']}"
        if row["partition"] != "train" or pair not in declared:
            continue
        train_counts[pair] += 1
        slot_types[pair].update(
            slot["slot_type"] for slot in parse_annotated_slots(row["annot_utt"])
        )
    intents = []
    for pair in sorted(declared):
        scenario, intent = pair.split("::", 1)
        if train_counts[pair] == 0:
            raise ValueError(f"declared intent lacks training annotations: {pair}")
        intents.append({
            "intent_id": pair,
            "scenario": scenario,
            "intent": intent,
            "slot_types": sorted(slot_types[pair]),
            "training_annotation_count": train_counts[pair],
        })
    visible_slots = sorted({slot for row in intents for slot in row["slot_types"]})
    catalog = {
        "scenarios": sorted(config["visibleScenarios"]),
        "intents": intents,
        "visible_unique_slot_types": visible_slots,
        "visible_unique_slot_type_count": len(visible_slots),
    }
    serialized = json.dumps(catalog, sort_keys=True)
    leak_count = sum(serialized.count(value) for value in config["hiddenGroundTruthMustNotAppearInVisibleCatalog"])
    return {
        "catalog": catalog,
        "catalog_sha256": canonical_sha256(catalog),
        "hidden_or_unsupported_schema_leak_count": leak_count,
    }


def complete_hypothesis_universe(catalog: dict[str, Any]) -> list[str]:
    hypotheses = [f"KNOWN::{row['intent_id']}" for row in catalog["intents"]]
    hypotheses.extend(f"NOVEL::{scenario}" for scenario in catalog["scenarios"])
    hypotheses.extend(["UNSUPPORTED", "INSUFFICIENT_EVIDENCE"])
    return sorted(hypotheses)


def select_controlled_insufficient_identifiers(
    population: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    rows = population["selected_population"]
    count = config["controlledInsufficientEvidence"]["recordsPerRole"]
    salt = config["controlledInsufficientEvidence"]["selectionSalt"]
    selected = {}
    for role in sorted(config.get("populationRoles", ["development", "protected_test"])):
        pool = [row for row in rows if row["role"] == role]
        ordered = sorted(pool, key=lambda row: hash_order(salt, role, row["candidate_id"]))
        if len(ordered) < count:
            raise ValueError("insufficient source identifiers for controlled ablation")
        selected[role] = [{
            "controlled_record_id": f"v105::missing::{role}::{row['candidate_id']}",
            "source_candidate_id": row["candidate_id"],
            "role": role,
            "observation_available": False,
            "expected_status": "ABSTAIN",
            "deterministic_runtime_action": "ABSTAIN_AND_ASK",
        } for row in ordered[:count]]
    return {
        "role_records": selected,
        "role_counts": {role: len(values) for role, values in selected.items()},
        "payload_sha256": canonical_sha256(selected),
        "contains_source_language": False,
    }


def response_fallback(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config["responseContract"]["invalidResponseFallback"])


def validate_response(value: Any, catalog: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
    contract = config["responseContract"]
    fallback = response_fallback(config)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return fallback, False, "invalid_json"
    if not isinstance(value, dict):
        return fallback, False, "not_object"
    required = set(contract["requiredKeys"])
    if set(value) != required:
        return fallback, False, "key_set"
    status = value["status"]
    known = value["known_intent"]
    scenario = value["novel_scenario"]
    confidence = value["confidence"]
    if status not in contract["allowedStatuses"]:
        return fallback, False, "status"
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return fallback, False, "confidence_type"
    if not contract["confidenceMinimum"] <= confidence <= contract["confidenceMaximum"]:
        return fallback, False, "confidence_range"
    declared = {row["intent_id"] for row in catalog["intents"]}
    scenarios = set(catalog["scenarios"])
    valid = bool(
        (status == "KNOWN" and known in declared and scenario is None)
        or (status == "NOVEL" and known is None and scenario in scenarios)
        or (status in {"UNSUPPORTED", "ABSTAIN"} and known is None and scenario is None)
    )
    if not valid:
        return fallback, False, "status_invariant"
    return {
        "status": status, "known_intent": known, "novel_scenario": scenario,
        "confidence": float(confidence),
    }, True, "valid"


def render_prompt(
    catalog: dict[str, Any], utterance: str | None, observation_available: bool,
    config: dict[str, Any],
) -> str:
    if observation_available and not isinstance(utterance, str):
        raise ValueError("available observation requires an utterance")
    if not observation_available:
        utterance = config["promptContract"]["missingObservationSentinel"]
    payload = {
        "instruction": config["promptContract"]["instruction"],
        "visible_catalog": catalog,
        "observation_available": observation_available,
        "user_utterance": utterance,
        "response_schema": config["responseContract"],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def evaluate_interface_gates(
    compiled: dict[str, Any], hypothesis_count: int,
    controlled: dict[str, Any], config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["interfaceGates"]
    catalog = compiled["catalog"]
    checks = {
        "visible_scenario_count": len(catalog["scenarios"]) == gates["requiredVisibleScenarioCount"],
        "visible_intent_count": len(catalog["intents"]) == gates["requiredVisibleIntentCount"],
        "visible_unique_slot_type_count": (
            catalog["visible_unique_slot_type_count"] >= gates["minimumVisibleUniqueSlotTypeCount"]
        ),
        "safe_hypothesis_count": hypothesis_count == gates["requiredSafeHypothesisCount"],
        "zero_hidden_or_unsupported_schema_leaks": (
            compiled["hidden_or_unsupported_schema_leak_count"]
            <= gates["maximumHiddenOrUnsupportedSchemaLeakCount"]
        ),
        "controlled_intervention_is_language_free": not controlled["contains_source_language"],
    }
    for role in ("development", "protected_test"):
        checks[f"{role}_controlled_insufficient_count"] = (
            controlled["role_counts"].get(role, 0)
            == gates["requiredControlledInsufficientRecordsPerRole"]
        )
    return checks


__all__ = [
    "compile_visible_catalog", "complete_hypothesis_universe", "evaluate_interface_gates",
    "render_prompt", "select_controlled_insufficient_identifiers", "validate_response",
]
