from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from v10_protocol import file_sha256


def rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def metadata_only_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False
    if parsed.netloc not in {"api.github.com", "raw.githubusercontent.com", "www.wikidata.org"}:
        return False
    lowered_path = parsed.path.lower()
    if re.search(r"/(issues|pull|pulls|comments|discussions)/\d+", lowered_path):
        return False
    if "property_proposal/archive" in lowered_path:
        return False
    query = parse_qs(parsed.query)
    if parsed.netloc == "api.github.com" and any(key in query for key in ("q", "search")):
        return False
    if parsed.netloc == "www.wikidata.org" and parsed.path.endswith("/w/api.php"):
        return query.get("prop") == ["categoryinfo"]
    return True


def score_census(
    manifest: dict[str, Any], evidence: dict[str, Any], config: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    frozen_units = {unit["unitId"]: unit for unit in config["sourceUnits"]}
    evidence_units = {unit["unit_id"]: unit for unit in evidence["source_units"]}
    dimensions = set(config["dimensions"])
    expected_attempts = {
        (unit["unitId"], url) for unit in config["sourceUnits"] for url in unit["urls"]
    }
    attempts = manifest["attempts"]
    actual_attempts = {(row["unit_id"], row["url"]) for row in attempts}
    successful = [row for row in attempts if row["success"]]
    snapshot_checks = []
    for row in successful:
        snapshot = project_root / row["snapshot_path"]
        snapshot_checks.append(
            snapshot.is_file()
            and file_sha256(snapshot) == row["sha256"]
            and snapshot.stat().st_size == row["byte_count"]
        )
    successful_by_unit = {
        unit_id: {row["url"] for row in successful if row["unit_id"] == unit_id}
        for unit_id in frozen_units
    }
    assessment_coverage: list[bool] = []
    true_evidence_coverage: list[bool] = []
    unit_metrics: dict[str, Any] = {}
    eligible: list[str] = []
    for unit_id, frozen in frozen_units.items():
        record = evidence_units.get(unit_id, {})
        assessments = record.get("assessments", {})
        citations = record.get("assessment_evidence_urls", {})
        reasons = record.get("false_or_qualified_reasons", {})
        assessment_coverage.append(
            set(assessments) == dimensions
            and set(citations) == dimensions
            and all(
                assessments.get(dimension) is True or bool(reasons.get(dimension))
                for dimension in dimensions
            )
        )
        for dimension in dimensions:
            if assessments.get(dimension) is True:
                cited = citations.get(dimension, [])
                true_evidence_coverage.append(
                    bool(cited)
                    and all(url in frozen["urls"] for url in cited)
                    and all(url in successful_by_unit[unit_id] for url in cited)
                )
        missing = sorted(
            dimension for dimension in dimensions if assessments.get(dimension) is not True
        )
        if not missing:
            eligible.append(unit_id)
        unit_metrics[unit_id] = {
            "successful_url_count": len(successful_by_unit[unit_id]),
            "true_dimension_count": sum(value is True for value in assessments.values()),
            "false_dimension_count": sum(value is False for value in assessments.values()),
            "missing_mandatory_dimensions": missing,
            "eligible_for_source_specific_acquisition_design": not missing,
        }
    selected = evidence.get("recommended_source_specific_candidate_ids", [])
    selected_missing = sum(
        len(unit_metrics.get(unit_id, {}).get("missing_mandatory_dimensions", dimensions))
        for unit_id in selected
    )
    metrics = {
        "source_unit_count": len(frozen_units),
        "frozen_url_attempt_count": len(expected_attempts),
        "recorded_url_attempt_count": len(attempts),
        "url_accounting_rate": len(expected_attempts & actual_attempts) / len(expected_attempts),
        "unexpected_url_attempt_count": len(actual_attempts - expected_attempts),
        "metadata_only_url_rate": rate([metadata_only_url(url) for _, url in actual_attempts]),
        "successful_url_count": len(successful),
        "failed_url_count": len(attempts) - len(successful),
        "successful_snapshot_hash_coverage": rate(snapshot_checks),
        "successful_snapshot_total_bytes": sum(row["byte_count"] for row in successful),
        "assessment_dimension_coverage": rate(assessment_coverage),
        "true_assessment_evidence_coverage": rate(true_evidence_coverage),
        "eligible_source_specific_candidate_ids": sorted(eligible),
        "eligible_source_specific_candidate_count": len(eligible),
        "recommended_source_specific_candidate_ids": selected,
        "selected_candidate_missing_mandatory_dimension_count": selected_missing,
        "claim_narrowed_to_structured_semantic_change_requests": bool(
            evidence["recommendation"]["claim_narrowed_to_structured_semantic_change_requests"]
        ),
        "formal_task_record_body_read_count": manifest["formal_task_record_body_read_count"],
        "issue_proposal_comment_pull_or_archive_record_request_count": manifest[
            "issue_proposal_comment_pull_or_archive_record_request_count"
        ],
        "unit_metrics": unit_metrics,
    }
    metrics["finite_metrics"] = finite(metrics)
    return metrics


def audit_census(
    metrics: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["censusGates"]
    checks = {
        "source_units_and_URL_attempts_are_exact": bool(
            metrics["source_unit_count"] == gates["requiredSourceUnitCount"]
            and metrics["frozen_url_attempt_count"] == gates["requiredFrozenURLAttemptCount"]
            and metrics["recorded_url_attempt_count"] == gates["requiredFrozenURLAttemptCount"]
            and metrics["url_accounting_rate"] == gates["requiredURLAccountingRate"]
            and metrics["unexpected_url_attempt_count"] == 0
        ),
        "metadata_firewall_and_snapshot_integrity_pass": bool(
            metrics["metadata_only_url_rate"] == 1.0
            and metrics["successful_snapshot_hash_coverage"]
            == gates["requiredSuccessfulSnapshotHashCoverage"]
            and metrics["successful_snapshot_total_bytes"]
            <= config["evidenceContract"]["maximumTotalSuccessfulSnapshotBytes"]
            and metrics["formal_task_record_body_read_count"]
            == gates["requiredFormalTaskRecordBodyReadCount"]
            and metrics["issue_proposal_comment_pull_or_archive_record_request_count"] == 0
        ),
        "assessments_are_complete_and_true_values_are_evidenced": bool(
            metrics["assessment_dimension_coverage"]
            == gates["requiredAssessmentDimensionCoverage"]
            and metrics["true_assessment_evidence_coverage"]
            == gates["requiredTrueAssessmentEvidenceCoverage"]
        ),
        "recommendation_is_narrow_and_selects_only_fully_eligible_sources": bool(
            metrics["claim_narrowed_to_structured_semantic_change_requests"]
            == gates["requiredClaimNarrowingToStructuredSemanticChangeRequests"]
            and metrics["selected_candidate_missing_mandatory_dimension_count"]
            <= gates["maximumSelectedCandidateMissingMandatoryDimensionCount"]
        ),
        "metrics_are_finite": metrics["finite_metrics"] == gates["requiredFiniteMetrics"],
    }
    access_gates = config["accessGates"]
    access_checks = {
        "one_metadata_census_run": access["metadata_census_run_count"]
        == access_gates["requiredMetadataCensusRunCount"],
        "record_model_protected_and_effect_boundaries_are_zero": all(
            access[key] <= access_gates[gate]
            for key, gate in {
                "formal_task_record_body_read_count": "maximumFormalTaskRecordBodyReadCount",
                "issue_proposal_comment_pull_or_archive_record_request_count": "maximumIssueProposalCommentPullOrArchiveRecordRequestCount",
                "protected_research_record_read_count": "maximumProtectedResearchRecordReadCount",
                "model_load_count": "maximumModelLoadCount",
                "model_generation_count": "maximumModelGenerationCount",
                "model_api_call_count": "maximumModelAPICallCount",
                "training_run_count": "maximumTrainingRunCount",
                "ontology_registration_count": "maximumOntologyRegistrationCount",
                "trusted_state_mutation_count": "maximumTrustedStateMutationCount",
                "service_action_count": "maximumServiceActionCount",
                "external_side_effect_count": "maximumExternalSideEffectCount",
                "actual_execution_count": "maximumActualExecutionCount",
            }.items()
        ),
    }
    integrity_passed = all(checks.values()) and all(access_checks.values())
    eligible = (
        metrics["eligible_source_specific_candidate_count"]
        >= gates["minimumEligibleSourceSpecificAcquisitionCandidateCount"]
    )
    branch = (
        "SOURCE_SPECIFIC_ACQUISITION_DESIGN_ELIGIBLE"
        if eligible
        else "NO_COMPLETE_ARCHIVED_ADJUDICATION_SOURCE"
    )
    decision = (
        config["decisionRule"]["ifAtLeastOneSourcePassesEveryMandatoryDimension"]
        if eligible
        else config["decisionRule"]["otherwise"]
    )
    return {
        "passed": integrity_passed,
        "branch": branch,
        "decision": decision,
        "checks": checks,
        "access_checks": access_checks,
    }
