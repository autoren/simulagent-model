from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def score_census(
    manifest: dict[str, Any], evidence: dict[str, Any], config: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    frozen_units = {unit["unitId"]: unit for unit in config["sourceUnits"]}
    evidence_units = {unit["unit_id"]: unit for unit in evidence["source_units"]}
    attempts = manifest["attempts"]
    expected_attempts = {
        (unit["unitId"], url) for unit in config["sourceUnits"] for url in unit["urls"]
    }
    actual_attempts = {(row["unit_id"], row["url"]) for row in attempts}
    successful = [row for row in attempts if row["success"]]
    snapshot_checks = []
    for row in successful:
        path = project_root / row["snapshot_path"]
        snapshot_checks.append(
            path.is_file()
            and file_sha256(path) == row["sha256"]
            and path.stat().st_size == row["byte_count"]
        )
    dimensions = set(config["dimensions"])
    assessment_coverage = []
    true_evidence_coverage = []
    unit_metrics: dict[str, Any] = {}
    eligible_payload: list[str] = []
    eligible_controls: list[str] = []
    unresolved_selected = 0
    successful_by_unit = {
        unit_id: {row["url"] for row in successful if row["unit_id"] == unit_id}
        for unit_id in frozen_units
    }
    for unit_id, frozen in frozen_units.items():
        record = evidence_units.get(unit_id, {})
        assessments = record.get("assessments", {})
        links = record.get("assessment_evidence_urls", {})
        assessment_coverage.append(set(assessments) == dimensions and set(links) == dimensions)
        per_true = []
        for dimension in dimensions:
            if assessments.get(dimension) is True:
                cited = links.get(dimension, [])
                per_true.append(
                    bool(cited)
                    and all(url in frozen["urls"] for url in cited)
                    and any(url in successful_by_unit[unit_id] for url in cited)
                )
        true_evidence_coverage.extend(per_true)
        role = frozen["intendedRole"]
        mandatory = config["roleRequirements"][role]
        missing = [dimension for dimension in mandatory if assessments.get(dimension) is not True]
        eligible = not missing
        if role == "PAYLOAD_BENCHMARK_CANDIDATE" and eligible:
            eligible_payload.append(unit_id)
        if role == "VALIDATION_CONTROL" and eligible:
            eligible_controls.append(unit_id)
        if unit_id in evidence.get("recommended_payload_candidate_ids", []):
            unresolved_selected += len(missing)
        unit_metrics[unit_id] = {
            "intended_role": role,
            "successful_url_count": len(successful_by_unit[unit_id]),
            "mandatory_dimension_count": len(mandatory),
            "missing_mandatory_dimensions": missing,
            "eligible_for_intended_role": eligible,
            "true_dimension_count": sum(value is True for value in assessments.values()),
            "false_dimension_count": sum(value is False for value in assessments.values()),
        }
    metrics = {
        "source_unit_count": len(frozen_units),
        "frozen_url_attempt_count": len(expected_attempts),
        "recorded_url_attempt_count": len(attempts),
        "url_accounting_rate": len(expected_attempts & actual_attempts) / len(expected_attempts),
        "unexpected_url_attempt_count": len(actual_attempts - expected_attempts),
        "successful_url_count": len(successful),
        "failed_url_count": len(attempts) - len(successful),
        "successful_snapshot_hash_coverage": _rate(snapshot_checks),
        "assessment_dimension_coverage": _rate(assessment_coverage),
        "true_assessment_evidence_coverage": _rate(true_evidence_coverage),
        "eligible_payload_benchmark_candidate_ids": sorted(eligible_payload),
        "eligible_payload_benchmark_candidate_count": len(eligible_payload),
        "eligible_validation_control_ids": sorted(eligible_controls),
        "eligible_validation_control_count": len(eligible_controls),
        "selected_payload_candidate_unresolved_mandatory_dimension_count": unresolved_selected,
        "recommendation_role_separation": bool(evidence["recommendation"]["role_separation_explicit"]),
        "retrospective_not_speaker_intent_boundary": bool(
            evidence["recommendation"]["retrospective_artifact_not_new_speaker_intent"]
        ),
        "unit_metrics": unit_metrics,
    }
    metrics["finite_metrics"] = _finite(metrics)
    return metrics


def audit_census(
    metrics: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["censusGates"]
    checks = {
        "source_units_and_URL_attempts_fully_accounted": bool(
            metrics["source_unit_count"] == gates["requiredSourceUnitCount"]
            and metrics["frozen_url_attempt_count"] == gates["requiredFrozenURLAttemptCount"]
            and metrics["recorded_url_attempt_count"] == gates["requiredFrozenURLAttemptCount"]
            and metrics["url_accounting_rate"] == gates["requiredURLAccountingRate"]
            and metrics["unexpected_url_attempt_count"] == 0
        ),
        "successful_metadata_snapshots_content_hashed": metrics["successful_snapshot_hash_coverage"] == gates["requiredSuccessfulSnapshotHashCoverage"],
        "assessments_complete_and_every_true_value_evidenced": bool(
            metrics["assessment_dimension_coverage"] == gates["requiredAssessmentDimensionCoverage"]
            and metrics["true_assessment_evidence_coverage"] == gates["requiredTrueAssessmentEvidenceCoverage"]
        ),
        "payload_and_validation_roles_have_eligible_sources": bool(
            metrics["eligible_payload_benchmark_candidate_count"] >= gates["minimumEligiblePayloadBenchmarkCandidateCount"]
            and metrics["eligible_validation_control_count"] >= gates["minimumEligibleValidationControlCount"]
            and metrics["selected_payload_candidate_unresolved_mandatory_dimension_count"] <= gates["maximumSelectedPayloadCandidateUnresolvedMandatoryDimensionCount"]
        ),
        "recommendation_preserves_roles_and_claim_boundary": bool(
            metrics["recommendation_role_separation"] == gates["requiredRecommendationRoleSeparation"]
            and metrics["retrospective_not_speaker_intent_boundary"] == gates["requiredRetrospectiveNotSpeakerIntentBoundary"]
        ),
        "metrics_are_finite": metrics["finite_metrics"] == gates["requiredFiniteMetrics"],
    }
    access_gates = config["accessGates"]
    access_checks = {
        "one_metadata_census_run": access["metadata_census_run_count"] == access_gates["requiredMetadataCensusRunCount"],
        "payload_protected_model_and_effect_boundaries_zero": bool(
            access["bulk_ontology_payload_download_count"] <= access_gates["maximumBulkOntologyPayloadDownloadCount"]
            and access["alignment_payload_download_count"] <= access_gates["maximumAlignmentPayloadDownloadCount"]
            and access["test_suite_payload_download_count"] <= access_gates["maximumTestSuitePayloadDownloadCount"]
            and access["v213_protected_access_count"] <= access_gates["maximumV213ProtectedAccessCount"]
            and access["model_load_count"] <= access_gates["maximumModelLoadCount"]
            and access["model_generation_count"] <= access_gates["maximumModelGenerationCount"]
            and access["api_call_count"] <= access_gates["maximumAPICallCount"]
            and access["training_run_count"] <= access_gates["maximumTrainingRunCount"]
            and access["ontology_registration_count"] <= access_gates["maximumOntologyRegistrationCount"]
            and access["trusted_state_mutation_count"] <= access_gates["maximumTrustedStateMutationCount"]
            and access["service_action_count"] <= access_gates["maximumServiceActionCount"]
            and access["external_side_effect_count"] <= access_gates["maximumExternalSideEffectCount"]
            and access["actual_execution_count"] <= access_gates["maximumActualExecutionCount"]
        ),
    }
    passed = all(checks.values()) and all(access_checks.values())
    return {
        "passed": passed,
        "branch": "BOUNDED_EXTERNAL_PAYLOAD_DESIGN_ELIGIBLE" if passed else "NEGATIVE_EXTERNAL_METADATA_FEASIBILITY",
        "decision": config["decisionRule"]["ifEveryIntegrityCensusAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "checks": checks,
        "access_checks": access_checks,
    }
