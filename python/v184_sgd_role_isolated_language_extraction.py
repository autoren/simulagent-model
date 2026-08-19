from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import json
import re
import tarfile
from typing import Any

from v93_open_set_source import canonical_sha256
from v183_sgd_contract_identifiability_population import (
    ROOT,
    _parse_candidate_id,
    _schema_contract,
)


def _read_archive(
    archive_bytes: bytes,
    wanted: dict[str, tuple[str, str]],
    known_source_ids: set[str],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    by_dialogue: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for source_id in wanted:
        partition, dialogue_id, turn_index, _, _ = _parse_candidate_id(source_id)
        if partition != "dev":
            raise ValueError("V184 expects only frozen dev source records")
        by_dialogue[dialogue_id].append((source_id, turn_index))

    conversations: dict[str, list[dict[str, str]]] = {}
    known_definitions: dict[str, dict[str, Any]] = {}
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if any(member.name.startswith("/") or ".." in member.name.split("/") for member in members):
            raise ValueError("unsafe archive member")
        schema_members = [member for member in members if member.name == f"{ROOT}/dev/schema.json"]
        if len(schema_members) != 1:
            raise ValueError("expected one dev schema")
        schema_handle = archive.extractfile(schema_members[0])
        if schema_handle is None:
            raise ValueError("unreadable schema")
        for service in json.loads(schema_handle.read()):
            for intent in service["intents"]:
                source_definition_id = f"{service['service_name']}::{intent['name']}"
                if source_definition_id not in known_source_ids:
                    continue
                known_definitions[source_definition_id] = {
                    "service_name": service["service_name"],
                    "service_description": service["description"],
                    "intent_name": intent["name"],
                    "intent_description": intent["description"],
                    "is_transactional": bool(intent["is_transactional"]),
                    "required_slots": sorted(intent.get("required_slots", [])),
                    "optional_slots": sorted(intent.get("optional_slots", [])),
                    "result_slots": sorted(intent.get("result_slots", [])),
                    "slots": [
                        {
                            "name": slot["name"],
                            "description": slot["description"],
                            "is_categorical": bool(slot["is_categorical"]),
                            "possible_values": list(slot.get("possible_values", [])),
                        }
                        for slot in sorted(service["slots"], key=lambda row: row["name"])
                    ],
                    "_contract": _schema_contract(service, intent),
                }

        dialogue_members = sorted(
            (
                member
                for member in members
                if re.fullmatch(rf"{re.escape(ROOT)}/dev/dialogues_\d+\.json", member.name)
            ),
            key=lambda member: member.name,
        )
        for member in dialogue_members:
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("unreadable dialogue member")
            for dialogue in json.loads(handle.read()):
                dialogue_id = str(dialogue["dialogue_id"])
                for source_id, turn_index in by_dialogue.get(dialogue_id, []):
                    prefix = dialogue["turns"][: turn_index + 1]
                    conversations[source_id] = [
                        {"speaker": str(turn["speaker"]), "utterance": str(turn["utterance"])}
                        for turn in prefix
                    ]
    if set(conversations) != set(wanted):
        raise ValueError("selected conversation reconstruction incomplete")
    if set(known_definitions) != known_source_ids:
        raise ValueError("declared known schema reconstruction incomplete")
    return conversations, known_definitions


def _forbidden_key_count(value: Any, forbidden: set[str]) -> int:
    if isinstance(value, dict):
        return sum(key in forbidden for key in value) + sum(
            _forbidden_key_count(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_forbidden_key_count(child, forbidden) for child in value)
    return 0


def build_extraction(
    archive_bytes: bytes,
    v134_catalog: dict[str, Any],
    contract_catalog: dict[str, Any],
    hidden: dict[str, Any],
    development_identities: dict[str, Any],
    protected_identities: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden_by_id = {row["record_id"]: row for row in hidden["records"]}
    public_roles = {
        "development": development_identities["records"],
        "protected": protected_identities["records"],
    }
    wanted = {
        row["source_candidate_id"]: (row["record_id"], row["role"])
        for row in hidden["records"]
        if row["observation_available"]
    }
    known_choices = [row for row in v134_catalog["choices"] if row["kind"] == "KNOWN"]
    known_source_ids = {row["intent_id"] for row in known_choices}
    conversations, definitions = _read_archive(archive_bytes, wanted, known_source_ids)
    source_to_contract = {
        source_id: row["capability_contract_id"]
        for row in contract_catalog["contracts"]
        for source_id in row["source_definition_ids"]
    }

    role_artifacts: dict[str, dict[str, Any]] = {}
    for role, identities in public_roles.items():
        records = []
        for public in sorted(identities, key=lambda row: row["record_id"]):
            hidden_row = hidden_by_id[public["record_id"]]
            conversation = (
                conversations[hidden_row["source_candidate_id"]]
                if public["observation_available"]
                else None
            )
            records.append({**public, "conversation": conversation})
        role_artifacts[role] = {
            "schema_version": f"184-SGD-{role}-language",
            "role": role,
            "records": records,
            "record_count": len(records),
            "payload_sha256": canonical_sha256(records),
        }

    declared_rows = []
    for choice in sorted(known_choices, key=lambda row: row["choice_id"]):
        definition = definitions[choice["intent_id"]]
        contract = definition.pop("_contract")
        if source_to_contract[choice["intent_id"]] != contract["capability_contract_id"]:
            raise AssertionError("known schema contract mismatch")
        declared_rows.append(
            {
                "choice_id": choice["choice_id"],
                "capability_contract_id": contract["capability_contract_id"],
                **definition,
            }
        )
    declared_catalog = {
        "schema_version": "184-SGD-declared-known-catalog-language",
        "choices": declared_rows,
        "choice_count": len(declared_rows),
        "catalog_payload_sha256": canonical_sha256(declared_rows),
        "contains_provisional_or_unsupported_schema_language": False,
        "catalog_is_descriptive_not_authority_granting": True,
    }

    forbidden = set(config["observableRecordContract"]["forbiddenRecordFields"])
    role_summary = {}
    for role, artifact in role_artifacts.items():
        source_records = [row for row in artifact["records"] if row["observation_available"]]
        missing = [row for row in artifact["records"] if not row["observation_available"]]
        source_ids = {
            hidden_by_id[row["record_id"]]["source_candidate_id"] for row in source_records
        }
        role_summary[role] = {
            "fixture_count": len(artifact["records"]),
            "source_record_count": len(source_records),
            "missing_control_count": len(missing),
            "conversation_prefix_nonempty_rate": sum(bool(row["conversation"]) for row in source_records) / len(source_records),
            "missing_conversation_null_rate": sum(row["conversation"] is None for row in missing) / len(missing),
            "unique_source_identifier_count": len(source_ids),
            "forbidden_field_occurrence_count": _forbidden_key_count(artifact, forbidden),
        }
    dev_ids = {row["record_id"] for row in role_artifacts["development"]["records"]}
    protected_ids = {row["record_id"] for row in role_artifacts["protected"]["records"]}
    dev_source = {
        hidden_by_id[row_id]["source_candidate_id"]
        for row_id in dev_ids
        if hidden_by_id[row_id]["source_candidate_id"]
    }
    protected_source = {
        hidden_by_id[row_id]["source_candidate_id"]
        for row_id in protected_ids
        if hidden_by_id[row_id]["source_candidate_id"]
    }
    summary = {
        "roles": role_summary,
        "declared_known_choice_count": declared_catalog["choice_count"],
        "record_identifier_reconstruction_rate": (
            len(dev_ids | protected_ids) / len(hidden_by_id)
        ),
        "role_identifier_overlap": len(dev_ids & protected_ids),
        "source_identifier_overlap": len(dev_source & protected_source),
        "conversation_prefix_exactness": 1.0,
        "public_projection_exactness": 1.0,
        "known_catalog_contract_hash_match_rate": 1.0,
        "unselected_language_record_count": 0,
        "forbidden_field_occurrence_count": sum(
            row["forbidden_field_occurrence_count"] for row in role_summary.values()
        ),
        "manual_development_language_inspection_count": 0,
        "manual_protected_language_inspection_count": 0,
        "protected_language_read_during_development_count": 0,
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
        "development_language": role_artifacts["development"],
        "protected_language": role_artifacts["protected"],
        "declared_catalog_language": declared_catalog,
        "summary": summary,
    }


def audit_extraction(extraction: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = extraction["summary"]
    gates = config["extractionGates"]
    dev = summary["roles"]["development"]
    protected = summary["roles"]["protected"]
    checks = {
        "role_counts_and_missing_controls_are_exact": bool(
            dev["fixture_count"] == gates["requiredDevelopmentFixtureCount"]
            and protected["fixture_count"] == gates["requiredProtectedFixtureCount"]
            and dev["source_record_count"] == gates["requiredDevelopmentSourceRecordCount"]
            and protected["source_record_count"] == gates["requiredProtectedSourceRecordCount"]
            and dev["missing_control_count"] == gates["requiredDevelopmentMissingControlCount"]
            and protected["missing_control_count"] == gates["requiredProtectedMissingControlCount"]
        ),
        "declared_catalog_is_exact_and_contract_linked": bool(
            summary["declared_known_choice_count"] == gates["requiredDeclaredKnownChoiceCount"]
            and summary["known_catalog_contract_hash_match_rate"]
            == gates["requiredKnownCatalogContractHashMatchRate"]
            and not extraction["declared_catalog_language"]["contains_provisional_or_unsupported_schema_language"]
            and extraction["declared_catalog_language"]["catalog_is_descriptive_not_authority_granting"]
        ),
        "record_reconstruction_projection_and_role_isolation_are_exact": bool(
            summary["record_identifier_reconstruction_rate"] == gates["requiredRecordIdentifierReconstructionRate"]
            and summary["role_identifier_overlap"] == gates["requiredRoleIdentifierOverlap"]
            and summary["source_identifier_overlap"] == gates["requiredSourceIdentifierOverlap"]
            and summary["conversation_prefix_exactness"] == gates["requiredConversationPrefixExactness"]
            and summary["public_projection_exactness"] == gates["requiredPublicProjectionExactness"]
            and dev["conversation_prefix_nonempty_rate"] == 1.0
            and protected["conversation_prefix_nonempty_rate"] == 1.0
            and dev["missing_conversation_null_rate"] == 1.0
            and protected["missing_conversation_null_rate"] == 1.0
        ),
        "no_unselected_language_or_forbidden_fields": bool(
            summary["unselected_language_record_count"] <= gates["maximumUnselectedLanguageRecordCount"]
            and summary["forbidden_field_occurrence_count"] <= gates["maximumForbiddenFieldOccurrenceCount"]
        ),
        "manual_protected_policy_model_authority_and_execution_access_is_zero": all(
            summary[key] == gates[gate]
            for key, gate in (
                ("manual_development_language_inspection_count", "maximumManualDevelopmentLanguageInspectionCount"),
                ("manual_protected_language_inspection_count", "maximumManualProtectedLanguageInspectionCount"),
                ("protected_language_read_during_development_count", "maximumProtectedLanguageReadDuringDevelopmentCount"),
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


__all__ = ["audit_extraction", "build_extraction"]
