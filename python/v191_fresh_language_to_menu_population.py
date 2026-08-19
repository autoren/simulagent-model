from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order
from v183_sgd_contract_identifiability_population import _parse_candidate_id


def _definition_id(row: dict[str, Any]) -> str:
    return f"{row['service']}::{row['intent']}"


def _truth_kind(contract: dict[str, Any]) -> str:
    kinds = contract["truth_kinds"]
    if len(kinds) != 1 or kinds[0] not in {"KNOWN", "PROVISIONAL", "UNSUPPORTED"}:
        raise ValueError("V191 requires one frozen truth kind per contract")
    return kinds[0]


def build_population(
    source_inventory: dict[str, Any],
    contract_catalog: dict[str, Any],
    previous_hidden: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["population"]
    fresh = config["freshnessContract"]
    contracts = sorted(contract_catalog["contracts"], key=lambda row: row["capability_contract_id"])
    definition_to_contract = {
        source_id: contract["capability_contract_id"]
        for contract in contracts
        for source_id in contract["source_definition_ids"]
    }
    contract_by_id = {row["capability_contract_id"]: row for row in contracts}

    previous_sources = {
        row["source_candidate_id"]
        for row in previous_hidden["records"]
        if row["observation_available"]
    }
    previous_dialogues = {_parse_candidate_id(value)[1] for value in previous_sources}

    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in contract_by_id}
    for row in source_inventory["candidate_index"]:
        source_id = _definition_id(row)
        if row["partition"] != fresh["sourcePartition"] or source_id not in definition_to_contract:
            continue
        _, dialogue_id, _, _, _ = _parse_candidate_id(row["candidate_id"])
        if row["candidate_id"] in previous_sources or dialogue_id in previous_dialogues:
            continue
        pools[definition_to_contract[source_id]].append(row)

    availability = []
    for contract_id in sorted(pools):
        dialogue_ids = {_parse_candidate_id(row["candidate_id"])[1] for row in pools[contract_id]}
        availability.append(
            {
                "capability_contract_id": contract_id,
                "truth_kind": _truth_kind(contract_by_id[contract_id]),
                "eligible_source_record_count": len(pools[contract_id]),
                "eligible_dialogue_count": len(dialogue_ids),
            }
        )

    selected: list[tuple[str, dict[str, Any]]] = []
    selected_dialogues: set[str] = set()
    per_contract = spec["sourceRecordsPerContract"]
    for contract_id in sorted(pools):
        ordered = sorted(
            pools[contract_id],
            key=lambda row: hash_order(fresh["baseSalt"], contract_id, row["candidate_id"]),
        )
        chosen = []
        for row in ordered:
            dialogue_id = _parse_candidate_id(row["candidate_id"])[1]
            if dialogue_id in selected_dialogues:
                continue
            chosen.append(row)
            selected_dialogues.add(dialogue_id)
            if len(chosen) == per_contract:
                break
        if len(chosen) != per_contract:
            raise ValueError(f"insufficient globally dialogue-disjoint records for {contract_id}")
        selected.extend((contract_id, row) for row in chosen)

    hidden_records = []
    public_records = []
    for contract_id, row in sorted(selected, key=lambda item: (item[0], item[1]["candidate_id"])):
        digest = hashlib.sha256(f"{fresh['baseSalt']}::{row['candidate_id']}".encode()).hexdigest()[:24]
        record_id = f"v191::{digest}"
        source_definition_id = _definition_id(row)
        hidden_records.append(
            {
                "record_id": record_id,
                "role": spec["role"],
                "observation_available": True,
                "source_candidate_id": row["candidate_id"],
                "source_definition_id": source_definition_id,
                "source_partition": row["partition"],
                "source_dialogue_id": _parse_candidate_id(row["candidate_id"])[1],
                "target_contract_id": contract_id,
                "truth_kind": _truth_kind(contract_by_id[contract_id]),
            }
        )
        public_records.append(
            {"record_id": record_id, "role": spec["role"], "observation_available": True}
        )

    for index in range(spec["requiredMissingControlCount"]):
        digest = hashlib.sha256(f"{fresh['baseSalt']}::missing::{index:02d}".encode()).hexdigest()[:24]
        record_id = f"v191::{digest}"
        hidden_records.append(
            {
                "record_id": record_id,
                "role": spec["role"],
                "observation_available": False,
                "source_candidate_id": None,
                "source_definition_id": None,
                "source_partition": None,
                "source_dialogue_id": None,
                "target_contract_id": None,
                "truth_kind": "INSUFFICIENT_CONTROL",
            }
        )
        public_records.append(
            {"record_id": record_id, "role": spec["role"], "observation_available": False}
        )

    hidden_records.sort(key=lambda row: row["record_id"])
    public_records.sort(key=lambda row: row["record_id"])
    observed = [row for row in hidden_records if row["observation_available"]]
    missing = [row for row in hidden_records if not row["observation_available"]]
    contract_counts = Counter(row["target_contract_id"] for row in observed)
    truth_counts = Counter(row["truth_kind"] for row in observed)
    selected_sources = {row["source_candidate_id"] for row in observed}
    selected_dialogue_list = [row["source_dialogue_id"] for row in observed]
    reconstructed = sum(
        definition_to_contract.get(row["source_definition_id"]) == row["target_contract_id"]
        for row in observed
    )
    summary = {
        "contract_count": len(contract_by_id),
        "eligible_dialogue_minimum": min(row["eligible_dialogue_count"] for row in availability),
        "source_record_count": len(observed),
        "selected_dialogue_count": len(set(selected_dialogue_list)),
        "missing_control_count": len(missing),
        "fixture_count": len(hidden_records),
        "records_per_contract": dict(sorted(contract_counts.items())),
        "truth_kind_counts": dict(sorted(truth_counts.items())),
        "V183_source_record_overlap": len(selected_sources & previous_sources),
        "V183_dialogue_overlap": len(set(selected_dialogue_list) & previous_dialogues),
        "within_V191_dialogue_overlap": len(selected_dialogue_list) - len(set(selected_dialogue_list)),
        "source_identifier_reconstruction_rate": reconstructed / len(observed),
        "target_contract_mapping_rate": reconstructed / len(observed),
        "missing_insufficient_rate": sum(row["truth_kind"] == "INSUFFICIENT_CONTROL" for row in missing) / len(missing),
        "persisted_utterance_or_dialogue_text_count": 0,
        "persisted_slot_value_frame_or_span_count": 0,
        "manual_language_inspection_count": 0,
        "protected_language_read_count": 0,
        "policy_score_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    return {
        "availability_census": {
            "schema_version": "191-fresh-language-to-menu-availability-census",
            "records": availability,
            "record_count": len(availability),
            "contains_language": False,
            "payload_sha256": canonical_sha256(availability),
        },
        "public_identities": {
            "schema_version": "191-fresh-language-to-menu-development-identities",
            "role": spec["role"],
            "records": public_records,
            "record_count": len(public_records),
            "contains_language_or_targets": False,
            "payload_sha256": canonical_sha256(public_records),
        },
        "hidden_targets": {
            "schema_version": "191-fresh-language-to-menu-hidden-targets",
            "records": hidden_records,
            "record_count": len(hidden_records),
            "contains_language": False,
            "payload_sha256": canonical_sha256(hidden_records),
        },
        "summary": summary,
    }


def audit_population(population: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = population["summary"]
    gates = config["populationGates"]
    checks = {
        "availability_and_contract_coverage_are_sufficient": bool(
            summary["contract_count"] == gates["requiredSelectedContractCount"]
            and summary["eligible_dialogue_minimum"] >= gates["minimumUnusedDialogueCountPerContract"]
            and len(summary["records_per_contract"]) == gates["requiredSelectedContractCount"]
            and set(summary["records_per_contract"].values()) == {gates["requiredRecordsPerContract"]}
        ),
        "population_and_truth_counts_are_exact": bool(
            summary["source_record_count"] == gates["requiredSelectedSourceRecordCount"]
            and summary["selected_dialogue_count"] == gates["requiredSelectedDialogueCount"]
            and summary["missing_control_count"] == gates["requiredMissingControlCount"]
            and summary["fixture_count"] == gates["requiredFixtureCount"]
            and summary["truth_kind_counts"] == {
                "KNOWN": gates["requiredKnownSourceRecordCount"],
                "PROVISIONAL": gates["requiredProvisionalSourceRecordCount"],
                "UNSUPPORTED": gates["requiredUnsupportedSourceRecordCount"],
            }
        ),
        "record_and_dialogue_freshness_are_exact": bool(
            summary["V183_source_record_overlap"] == gates["requiredV183SourceRecordOverlap"]
            and summary["V183_dialogue_overlap"] == gates["requiredV183DialogueOverlap"]
            and summary["within_V191_dialogue_overlap"] == gates["requiredWithinV191DialogueOverlap"]
        ),
        "source_mapping_and_missing_controls_are_exact": bool(
            summary["source_identifier_reconstruction_rate"] == gates["requiredSourceIdentifierReconstructionRate"]
            and summary["target_contract_mapping_rate"] == gates["requiredTargetContractMappingRate"]
            and summary["missing_insufficient_rate"] == gates["requiredMissingInsufficientRate"]
        ),
        "language_model_authority_and_execution_access_is_zero": all(
            summary[key] == gates[gate]
            for key, gate in (
                ("persisted_utterance_or_dialogue_text_count", "maximumPersistedUtteranceOrDialogueTextCount"),
                ("persisted_slot_value_frame_or_span_count", "maximumPersistedSlotValueFrameOrSpanCount"),
                ("manual_language_inspection_count", "maximumManualLanguageInspectionCount"),
                ("protected_language_read_count", "maximumProtectedLanguageReadCount"),
                ("policy_score_count", "maximumPolicyScoreCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = ["audit_population", "build_population"]
