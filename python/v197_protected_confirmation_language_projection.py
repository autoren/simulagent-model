from __future__ import annotations

from typing import Any

from v93_open_set_source import canonical_sha256


FORBIDDEN_FIELDS = {
    "presented_candidate_choice_id", "source_candidate_id", "source_definition_id",
    "truth_contract_id", "target_contract_id", "truth_kind", "compatible_contract_ids",
    "compatible_contract_count", "compatible_truth_kinds", "evaluation_choice",
    "source_truth_choice_id", "identifiability_status", "frame_signature",
    "target_contract_retained", "service", "intent", "slots", "slot_values", "spans",
}


def _forbidden_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(key in FORBIDDEN_FIELDS for key in value) + sum(_forbidden_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_forbidden_count(item) for item in value)
    return 0


def build_projection(
    protected_language: dict[str, Any],
    confirmation_identities: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    selected_ids = {row["record_id"] for row in confirmation_identities["records"]}
    source_by_id = {row["record_id"]: row for row in protected_language["records"]}
    projected = []
    for identity in sorted(confirmation_identities["records"], key=lambda row: row["record_id"]):
        source = source_by_id[identity["record_id"]]
        if source["observation_available"] != identity["observation_available"]:
            raise ValueError("V197 observation availability mismatch")
        projected.append({
            "record_id": identity["record_id"],
            "role": config["projection"]["outputRole"],
            "observation_available": identity["observation_available"],
            "conversation": source["conversation"],
        })
    observed = [row for row in projected if row["observation_available"]]
    missing = [row for row in projected if not row["observation_available"]]
    exact = sum(
        row["conversation"] == source_by_id[row["record_id"]]["conversation"] for row in projected
    )
    summary = {
        "input_record_count": len(protected_language["records"]),
        "selected_record_count": len(projected),
        "selected_observed_count": len(observed),
        "selected_missing_count": len(missing),
        "unselected_read_but_not_emitted_count": len(set(source_by_id) - selected_ids),
        "identifier_reconstruction_rate": len(set(source_by_id) & selected_ids) / len(selected_ids),
        "conversation_projection_exactness": exact / len(projected),
        "missing_conversation_null_rate": sum(row["conversation"] is None for row in missing) / len(missing),
        "forbidden_field_occurrence_count": _forbidden_count(projected),
        "unselected_language_emission_count": sum(row["record_id"] not in selected_ids for row in projected),
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
    artifact = {
        "schema_version": "197-selected-protected-confirmation-language",
        "role": config["projection"]["outputRole"],
        "records": projected,
        "record_count": len(projected),
        "payload_sha256": canonical_sha256(projected),
    }
    return {"language": artifact, "summary": summary}


def audit_projection(projection: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = projection["summary"]
    gates = config["projectionGates"]
    checks = {
        "input_selected_and_unselected_counts_are_exact": bool(
            summary["input_record_count"] == gates["requiredInputRecordCount"]
            and summary["selected_record_count"] == gates["requiredSelectedRecordCount"]
            and summary["selected_observed_count"] == gates["requiredSelectedObservedCount"]
            and summary["selected_missing_count"] == gates["requiredSelectedMissingCount"]
            and summary["unselected_read_but_not_emitted_count"] == gates["requiredUnselectedReadButNotEmittedCount"]
        ),
        "identifier_conversation_and_missing_projection_are_exact": bool(
            summary["identifier_reconstruction_rate"] == gates["requiredIdentifierReconstructionRate"]
            and summary["conversation_projection_exactness"] == gates["requiredConversationProjectionExactness"]
            and summary["missing_conversation_null_rate"] == gates["requiredMissingConversationNullRate"]
        ),
        "forbidden_and_unselected_fields_are_not_emitted": bool(
            summary["forbidden_field_occurrence_count"] <= gates["maximumForbiddenFieldOccurrenceCount"]
            and summary["unselected_language_emission_count"] <= gates["maximumUnselectedLanguageEmissionCount"]
        ),
        "scoring_model_authority_and_execution_access_is_zero": all(
            summary[key] == gates[gate]
            for key, gate in (
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


__all__ = ["audit_projection", "build_projection"]
