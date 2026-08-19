from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any

from v93_open_set_source import canonical_sha256
from v183_sgd_contract_identifiability_population import _parse_candidate_id


def _definition_id(row: dict[str, Any]) -> str:
    return f"{row['service']}::{row['intent']}"


def _dialogue_id(candidate_id: str) -> str:
    return _parse_candidate_id(candidate_id)[1]


def build_binding(
    source_inventory: dict[str, Any],
    contract_catalog: dict[str, Any],
    v183_hidden: dict[str, Any],
    v191_hidden: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    decision = config["sourceDecision"]
    contracts = {row["capability_contract_id"]: row for row in contract_catalog["contracts"]}
    definition_to_contract = {
        source_id: contract_id
        for contract_id, contract in contracts.items()
        for source_id in contract["source_definition_ids"]
    }
    v183_observed = [row for row in v183_hidden["records"] if row["observation_available"]]
    v183_development = [row for row in v183_observed if row["role"] == "development"]
    v183_protected = [row for row in v183_observed if row["role"] == "protected"]
    v183_missing = [
        row for row in v183_hidden["records"]
        if row["role"] == "protected" and not row["observation_available"]
    ]
    v191_observed = [row for row in v191_hidden["records"] if row["observation_available"]]
    all_used_dialogues = {_dialogue_id(row["source_candidate_id"]) for row in v183_observed + v191_observed}
    remaining_by_contract: dict[str, dict[str, set[str]]] = {
        key: defaultdict(set) for key in contracts
    }
    for row in source_inventory["candidate_index"]:
        definition_id = _definition_id(row)
        if definition_id not in definition_to_contract:
            continue
        dialogue_id = _dialogue_id(row["candidate_id"])
        if dialogue_id in all_used_dialogues:
            continue
        remaining_by_contract[definition_to_contract[definition_id]][row["partition"]].add(dialogue_id)
    remaining_census = [
        {
            "capability_contract_id": contract_id,
            "truth_kind": contracts[contract_id]["truth_kinds"][0],
            "remaining_dev_dialogue_count": len(remaining_by_contract[contract_id]["dev"]),
            "remaining_test_dialogue_count": len(remaining_by_contract[contract_id]["test"]),
            "remaining_train_dialogue_count": len(remaining_by_contract[contract_id]["train"]),
        }
        for contract_id in sorted(contracts)
    ]

    development_dialogues = {_dialogue_id(row["source_candidate_id"]) for row in v183_development}
    v191_dialogues = {_dialogue_id(row["source_candidate_id"]) for row in v191_observed}
    eligible_by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in v183_protected:
        dialogue_id = _dialogue_id(row["source_candidate_id"])
        if dialogue_id in development_dialogues or dialogue_id in v191_dialogues:
            continue
        eligible_by_dialogue[dialogue_id].append(row)
    selected = []
    for dialogue_id, rows in eligible_by_dialogue.items():
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{decision['selectionSalt']}::{row['source_candidate_id']}".encode()
            ).hexdigest(),
        )
        selected.append(ordered[0])
    selected.sort(key=lambda row: row["record_id"])

    hidden_records = []
    public_records = []
    for row in selected:
        hidden_records.append({
            "record_id": row["record_id"],
            "role": config["population"]["role"],
            "source_role": "protected",
            "observation_available": True,
            "source_candidate_id": row["source_candidate_id"],
            "source_definition_id": row["source_definition_id"],
            "source_dialogue_id": _dialogue_id(row["source_candidate_id"]),
            "target_contract_id": row["truth_contract_id"],
            "truth_kind": row["truth_kind"],
        })
        public_records.append({
            "record_id": row["record_id"],
            "role": config["population"]["role"],
            "observation_available": True,
        })
    for row in sorted(v183_missing, key=lambda item: item["record_id"]):
        hidden_records.append({
            "record_id": row["record_id"],
            "role": config["population"]["role"],
            "source_role": "protected",
            "observation_available": False,
            "source_candidate_id": None,
            "source_definition_id": None,
            "source_dialogue_id": None,
            "target_contract_id": None,
            "truth_kind": "INSUFFICIENT_CONTROL",
        })
        public_records.append({
            "record_id": row["record_id"],
            "role": config["population"]["role"],
            "observation_available": False,
        })
    hidden_records.sort(key=lambda row: row["record_id"])
    public_records.sort(key=lambda row: row["record_id"])

    observed = [row for row in hidden_records if row["observation_available"]]
    missing = [row for row in hidden_records if not row["observation_available"]]
    dialogues = [row["source_dialogue_id"] for row in observed]
    contract_counts = Counter(row["target_contract_id"] for row in observed)
    truth_counts = Counter(row["truth_kind"] for row in observed)
    mapping_correct = sum(
        definition_to_contract[row["source_definition_id"]] == row["target_contract_id"]
        for row in observed
    )
    summary = {
        "remaining_dev_minimum_across_contracts": min(row["remaining_dev_dialogue_count"] for row in remaining_census),
        "limiting_contract_remaining_dev_count": next(
            row["remaining_dev_dialogue_count"] for row in remaining_census
            if row["capability_contract_id"] == decision["limitingContractId"]
        ),
        "all_dev_confirmation_feasible": all(row["remaining_dev_dialogue_count"] > 0 for row in remaining_census),
        "V183_protected_observed_record_count": len(v183_protected),
        "excluded_V183_development_overlap_record_count": sum(
            _dialogue_id(row["source_candidate_id"]) in development_dialogues for row in v183_protected
        ),
        "removed_within_protected_duplicate_dialogue_record_count": sum(len(rows) - 1 for rows in eligible_by_dialogue.values()),
        "source_record_count": len(observed),
        "selected_dialogue_count": len(set(dialogues)),
        "missing_control_count": len(missing),
        "fixture_count": len(hidden_records),
        "contract_count": len(contract_counts),
        "records_per_contract": dict(sorted(contract_counts.items())),
        "truth_kind_counts": dict(sorted(truth_counts.items())),
        "V183_development_dialogue_overlap": len(set(dialogues) & development_dialogues),
        "V191_dialogue_overlap": len(set(dialogues) & v191_dialogues),
        "within_confirmation_dialogue_overlap": len(dialogues) - len(set(dialogues)),
        "target_contract_mapping_rate": mapping_correct / len(observed),
        "missing_insufficient_rate": sum(row["truth_kind"] == "INSUFFICIENT_CONTROL" for row in missing) / len(missing),
        "protected_utterance_read_or_emission_count": 0,
        "manual_language_inspection_count": 0,
        "policy_score_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    return {
        "remaining_source_census": {
            "schema_version": "196-remaining-SGD-source-metadata-census",
            "contains_language": False,
            "records": remaining_census,
            "record_count": len(remaining_census),
            "payload_sha256": canonical_sha256(remaining_census),
        },
        "public_identities": {
            "schema_version": "196-dialogue-isolated-confirmation-identities",
            "role": config["population"]["role"],
            "contains_language_or_targets": False,
            "records": public_records,
            "record_count": len(public_records),
            "payload_sha256": canonical_sha256(public_records),
        },
        "hidden_targets": {
            "schema_version": "196-dialogue-isolated-confirmation-hidden-targets",
            "contains_language": False,
            "records": hidden_records,
            "record_count": len(hidden_records),
            "payload_sha256": canonical_sha256(hidden_records),
        },
        "summary": summary,
    }


def audit_binding(binding: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = binding["summary"]
    gates = config["populationGates"]
    checks = {
        "new_all_dev_population_is_rejected_by_exact_availability_census": bool(
            summary["remaining_dev_minimum_across_contracts"] == gates["requiredRemainingDevMinimumAcrossContracts"]
            and summary["limiting_contract_remaining_dev_count"] == gates["requiredLimitingContractRemainingDevCount"]
            and not summary["all_dev_confirmation_feasible"]
        ),
        "selected_population_counts_and_contract_coverage_are_exact": bool(
            summary["source_record_count"] == gates["requiredObservedRecordCount"]
            and summary["selected_dialogue_count"] == gates["requiredSelectedDialogueCount"]
            and summary["missing_control_count"] == gates["requiredMissingControlCount"]
            and summary["fixture_count"] == gates["requiredFixtureCount"]
            and summary["contract_count"] == gates["requiredContractCount"]
            and min(summary["records_per_contract"].values()) >= gates["minimumObservedRecordsPerContract"]
            and summary["truth_kind_counts"] == {
                "KNOWN": gates["requiredKnownRecordCount"],
                "PROVISIONAL": gates["requiredProvisionalRecordCount"],
                "UNSUPPORTED": gates["requiredUnsupportedRecordCount"],
            }
        ),
        "dialogue_isolation_is_exact": bool(
            summary["V183_development_dialogue_overlap"] == gates["requiredV183DevelopmentDialogueOverlap"]
            and summary["V191_dialogue_overlap"] == gates["requiredV191DialogueOverlap"]
            and summary["within_confirmation_dialogue_overlap"] == gates["requiredWithinConfirmationDialogueOverlap"]
        ),
        "target_mapping_and_missing_controls_are_exact": bool(
            summary["target_contract_mapping_rate"] == gates["requiredTargetMappingRate"]
            and summary["missing_insufficient_rate"] == gates["requiredMissingInsufficientRate"]
        ),
        "language_model_authority_and_execution_access_is_zero": all(
            summary[key] == gates[gate]
            for key, gate in (
                ("protected_utterance_read_or_emission_count", "maximumProtectedUtteranceReadOrEmissionCount"),
                ("manual_language_inspection_count", "maximumManualLanguageInspectionCount"),
                ("policy_score_count", "maximumPolicyScoreCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("real_service_call_count", "maximumRealServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = ["audit_binding", "build_binding"]
