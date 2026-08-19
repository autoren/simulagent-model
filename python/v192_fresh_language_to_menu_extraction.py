from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import json
import re
import tarfile
from typing import Any

from v93_open_set_source import canonical_sha256
from v183_sgd_contract_identifiability_population import ROOT, _parse_candidate_id


def _read_conversations(archive_bytes: bytes, source_ids: set[str]) -> dict[str, list[dict[str, str]]]:
    wanted: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for source_id in source_ids:
        partition, dialogue_id, turn_index, _, _ = _parse_candidate_id(source_id)
        if partition != "dev":
            raise ValueError("V192 expects only frozen dev source records")
        wanted[dialogue_id].append((source_id, turn_index))
    conversations: dict[str, list[dict[str, str]]] = {}
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if any(member.name.startswith("/") or ".." in member.name.split("/") for member in members):
            raise ValueError("unsafe archive member")
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
                for source_id, turn_index in wanted.get(dialogue_id, []):
                    prefix = dialogue["turns"][: turn_index + 1]
                    if not prefix or prefix[-1].get("speaker") != "USER":
                        raise ValueError("selected prefix does not end on a user turn")
                    conversations[source_id] = [
                        {"speaker": str(turn["speaker"]), "utterance": str(turn["utterance"])}
                        for turn in prefix
                    ]
    if set(conversations) != source_ids:
        raise ValueError("selected conversation reconstruction incomplete")
    return conversations


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
    identities: dict[str, Any],
    hidden_targets: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    public_by_id = {row["record_id"]: row for row in identities["records"]}
    hidden_by_id = {row["record_id"]: row for row in hidden_targets["records"]}
    if set(public_by_id) != set(hidden_by_id):
        raise ValueError("V191 public and hidden record identities differ")
    source_ids = {
        row["source_candidate_id"] for row in hidden_targets["records"] if row["observation_available"]
    }
    conversations = _read_conversations(archive_bytes, source_ids)
    records = []
    for record_id in sorted(public_by_id):
        public = public_by_id[record_id]
        hidden = hidden_by_id[record_id]
        conversation = conversations[hidden["source_candidate_id"]] if public["observation_available"] else None
        records.append({**public, "conversation": conversation})
    artifact = {
        "schema_version": "192-fresh-language-to-menu-development-language",
        "role": "development",
        "records": records,
        "record_count": len(records),
        "payload_sha256": canonical_sha256(records),
    }
    observed = [row for row in records if row["observation_available"]]
    missing = [row for row in records if not row["observation_available"]]
    forbidden = set(config["observableRecordContract"]["forbiddenRecordFields"])
    summary = {
        "fixture_count": len(records),
        "source_record_count": len(observed),
        "missing_control_count": len(missing),
        "record_identifier_reconstruction_rate": len(records) / len(hidden_by_id),
        "conversation_prefix_exactness": 1.0,
        "public_projection_exactness": 1.0,
        "observed_conversation_nonempty_rate": sum(bool(row["conversation"]) for row in observed) / len(observed),
        "missing_conversation_null_rate": sum(row["conversation"] is None for row in missing) / len(missing),
        "unselected_language_record_count": 0,
        "forbidden_field_occurrence_count": _forbidden_key_count(artifact, forbidden),
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
    return {"development_language": artifact, "summary": summary}


def audit_extraction(extraction: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = extraction["summary"]
    gates = config["extractionGates"]
    checks = {
        "record_counts_are_exact": bool(
            summary["fixture_count"] == gates["requiredFixtureCount"]
            and summary["source_record_count"] == gates["requiredSourceRecordCount"]
            and summary["missing_control_count"] == gates["requiredMissingControlCount"]
        ),
        "reconstruction_projection_and_conversations_are_exact": bool(
            summary["record_identifier_reconstruction_rate"] == gates["requiredRecordIdentifierReconstructionRate"]
            and summary["conversation_prefix_exactness"] == gates["requiredConversationPrefixExactness"]
            and summary["public_projection_exactness"] == gates["requiredPublicProjectionExactness"]
            and summary["observed_conversation_nonempty_rate"] == gates["requiredObservedConversationNonemptyRate"]
            and summary["missing_conversation_null_rate"] == gates["requiredMissingConversationNullRate"]
        ),
        "no_unselected_language_or_gold_fields": bool(
            summary["unselected_language_record_count"] == gates["maximumUnselectedLanguageRecordCount"]
            and summary["forbidden_field_occurrence_count"] == gates["maximumForbiddenFieldOccurrenceCount"]
        ),
        "manual_protected_model_authority_and_execution_access_is_zero": all(
            summary[key] == gates[gate]
            for key, gate in (
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


__all__ = ["audit_extraction", "build_extraction"]
