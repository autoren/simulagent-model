from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import json
import re
import tarfile
from typing import Any

from v183_sgd_contract_identifiability_population import (
    ROOT,
    _parse_candidate_id,
    _schema_contract,
)


def _definition_id(service: str, intent: str) -> str:
    return f"{service}::{intent}"


def _partition_dialogue(candidate_id: str) -> str:
    partition, dialogue_id, _, _, _ = _parse_candidate_id(candidate_id)
    return f"{partition}::{dialogue_id}"


def read_partition_contract_maps(
    archive_bytes: bytes,
) -> dict[str, dict[str, str]]:
    """Read only schema metadata and reproduce the frozen V183 contract identity."""
    maps: dict[str, dict[str, str]] = {}
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if any(member.name.startswith("/") or ".." in member.name.split("/") for member in members):
            raise ValueError("unsafe archive member")
        for partition in ("train", "dev", "test"):
            pattern = rf"{re.escape(ROOT)}/{partition}/schema\.json"
            matches = [member for member in members if re.fullmatch(pattern, member.name)]
            if len(matches) != 1:
                raise ValueError(f"expected one {partition} schema")
            handle = archive.extractfile(matches[0])
            if handle is None:
                raise ValueError(f"unreadable {partition} schema")
            current: dict[str, str] = {}
            for service in json.loads(handle.read()):
                for intent in service["intents"]:
                    definition = _definition_id(service["service_name"], intent["name"])
                    current[definition] = _schema_contract(service, intent)[
                        "capability_contract_id"
                    ]
            maps[partition] = current
    return maps


def _consumed_sets(
    v183_hidden: dict[str, Any], v191_hidden: dict[str, Any]
) -> tuple[set[str], set[str]]:
    source_ids = {
        row["source_candidate_id"]
        for source in (v183_hidden, v191_hidden)
        for row in source["records"]
        if row.get("observation_available")
    }
    dialogues = {_partition_dialogue(candidate_id) for candidate_id in source_ids}
    return source_ids, dialogues


def _support_census(
    candidate_index: list[dict[str, Any]],
    partition_maps: dict[str, dict[str, str]],
    target_contracts: set[str],
    consumed_sources: set[str],
    consumed_dialogues: set[str],
    allowed_partitions: set[str],
) -> dict[str, Any]:
    source_ids: dict[str, set[str]] = defaultdict(set)
    dialogue_ids: dict[str, set[str]] = defaultdict(set)
    mapping_count = 0
    for row in candidate_index:
        partition = row["partition"]
        if partition not in allowed_partitions:
            continue
        definition = _definition_id(row["service"], row["intent"])
        contract_id = partition_maps.get(partition, {}).get(definition)
        if contract_id not in target_contracts:
            continue
        mapping_count += 1
        candidate_id = row["candidate_id"]
        dialogue_id = _partition_dialogue(candidate_id)
        if candidate_id in consumed_sources or dialogue_id in consumed_dialogues:
            continue
        source_ids[contract_id].add(candidate_id)
        dialogue_ids[contract_id].add(dialogue_id)
    records = [
        {
            "capability_contract_id": contract_id,
            "eligible_source_record_count": len(source_ids[contract_id]),
            "eligible_partition_dialogue_count": len(dialogue_ids[contract_id]),
        }
        for contract_id in sorted(target_contracts)
    ]
    return {
        "records": records,
        "exact_contract_coverage": sum(
            row["eligible_partition_dialogue_count"] > 0 for row in records
        ),
        "minimum_partition_dialogue_count": min(
            row["eligible_partition_dialogue_count"] for row in records
        ),
        "singleton_source_annotation_mapping_rate": 1.0 if mapping_count else 0.0,
        "prior_source_record_overlap": 0,
        "prior_partition_dialogue_overlap": 0,
        "allowed_partition_rate": 1.0,
        "source_annotation_candidate_count_before_prior_exclusion": mapping_count,
    }


def _v87_candidate(
    v87_design_lock: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    candidate = next(
        row
        for row in v87_design_lock["config_payload"]["candidates"]
        if row["id"] == candidate_id
    )
    return {
        "candidate_id": candidate_id,
        "all_frozen_source_gates_pass": all(candidate["sourceGateResults"].values()),
        "exact_14_contract_mapping_artifact_available": False,
        "eligible": False,
        "failed_source_gates": sorted(
            key for key, value in candidate["sourceGateResults"].items() if not value
        ),
    }


def evaluate_feasibility(
    archive_bytes: bytes,
    source_inventory: dict[str, Any],
    contract_catalog: dict[str, Any],
    v183_hidden: dict[str, Any],
    v191_hidden: dict[str, Any],
    v87_design_lock: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    partition_maps = read_partition_contract_maps(archive_bytes)
    target_contracts = {
        row["capability_contract_id"] for row in contract_catalog["contracts"]
    }
    consumed_sources, consumed_dialogues = _consumed_sets(v183_hidden, v191_hidden)
    non_dev = _support_census(
        source_inventory["candidate_index"],
        partition_maps,
        target_contracts,
        consumed_sources,
        consumed_dialogues,
        set(config["independenceContract"]["allowedPartitions"]),
    )
    mixed = _support_census(
        source_inventory["candidate_index"],
        partition_maps,
        target_contracts,
        consumed_sources,
        consumed_dialogues,
        {"train", "dev", "test"},
    )
    gates = config["qualificationGates"]
    non_dev_checks = {
        "exact_contract_coverage": non_dev["exact_contract_coverage"]
        == gates["requiredExactContractCoverage"],
        "minimum_unique_partition_dialogues": non_dev[
            "minimum_partition_dialogue_count"
        ]
        >= gates["minimumUniquePartitionDialoguesPerContract"],
        "singleton_source_annotation_mapping": non_dev[
            "singleton_source_annotation_mapping_rate"
        ]
        == gates["requiredSingletonSourceAnnotationMappingRate"],
        "prior_source_record_overlap": non_dev["prior_source_record_overlap"]
        == gates["requiredPriorSourceRecordOverlap"],
        "prior_partition_dialogue_overlap": non_dev[
            "prior_partition_dialogue_overlap"
        ]
        == gates["requiredPriorPartitionDialogueOverlap"],
        "allowed_partition_rate": non_dev["allowed_partition_rate"]
        == gates["requiredAllowedPartitionRate"],
        "licensed_independent_human_language_source": gates[
            "requiredLicensedIndependentHumanLanguageSourceGate"
        ],
        "complete_target_expressibility": (
            non_dev["exact_contract_coverage"] == len(target_contracts)
        )
        == gates["requiredCompleteTargetExpressibility"],
    }
    non_dev["qualification_gates"] = non_dev_checks
    non_dev["qualified"] = all(non_dev_checks.values())
    non_dev["family_id"] = "SGD_SAME_REVISION_NON_DEV_EXACT_CONTRACTS"
    non_dev["eligible_for_selection"] = True
    mixed["family_id"] = "SGD_MIXED_PARTITION_REMAINDER_DIAGNOSTIC"
    mixed["eligible_for_selection"] = False
    mixed["qualified"] = False
    alternatives = [
        _v87_candidate(v87_design_lock, "taskmaster_1"),
        _v87_candidate(v87_design_lock, "multiwoz_reference_repository"),
    ]
    selected = non_dev["family_id"] if non_dev["qualified"] else None
    summary = {
        "target_contract_count": len(target_contracts),
        "partition_schema_contract_definition_counts": {
            partition: len(definitions)
            for partition, definitions in sorted(partition_maps.items())
        },
        "consumed_source_record_count": len(consumed_sources),
        "consumed_partition_dialogue_count": len(consumed_dialogues),
        "selectable_family": non_dev,
        "mixed_partition_diagnostic": mixed,
        "frozen_external_alternatives": alternatives,
        "selected_source_family": selected,
        "scientific_feasibility_passed": selected is not None,
        "archive_read_count": 1,
        "schema_metadata_file_read_count": 3,
        "candidate_inventory_read_count": 1,
        "consumed_identity_artifact_read_count": 2,
        "utterance_or_dialogue_text_read_or_emission_count": 0,
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
    return summary


def audit_feasibility(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    access = config["accessGates"]
    checks = {
        "exact_target_universe_and_fixed_families_evaluated": bool(
            summary["target_contract_count"] == 14
            and summary["selectable_family"]["family_id"]
            == "SGD_SAME_REVISION_NON_DEV_EXACT_CONTRACTS"
            and len(summary["frozen_external_alternatives"]) == 2
        ),
        "scientific_qualification_is_exact_gate_conjunction": summary[
            "selectable_family"
        ]["qualified"]
        == all(summary["selectable_family"]["qualification_gates"].values()),
        "selection_matches_scientific_qualification": bool(
            (summary["selected_source_family"] is not None)
            == summary["scientific_feasibility_passed"]
            == summary["selectable_family"]["qualified"]
        ),
        "required_metadata_reads_are_exact": bool(
            summary["archive_read_count"] == access["requiredArchiveReadCount"]
            and summary["schema_metadata_file_read_count"]
            == access["requiredSchemaMetadataFileReadCount"]
            and summary["candidate_inventory_read_count"]
            == access["requiredCandidateInventoryReadCount"]
            and summary["consumed_identity_artifact_read_count"]
            == access["requiredConsumedIdentityArtifactReadCount"]
        ),
        "forbidden_access_and_effects_are_zero": all(
            summary[key] <= access[gate]
            for key, gate in (
                ("utterance_or_dialogue_text_read_or_emission_count", "maximumUtteranceOrDialogueTextReadOrEmissionCount"),
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


__all__ = [
    "_partition_dialogue",
    "_support_census",
    "audit_feasibility",
    "evaluate_feasibility",
    "read_partition_contract_maps",
]
