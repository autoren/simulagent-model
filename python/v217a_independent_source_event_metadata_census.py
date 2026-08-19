from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from v10_protocol import file_sha256


def _rate(values: list[bool], *, empty: float = 1.0) -> float:
    return sum(values) / len(values) if values else empty


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
    dimensions = set(config["dimensions"])
    event_categories = set(config["eventCategories"])
    ambiguity_indicators = set(config["ambiguityIndicators"])
    attempts = manifest["attempts"]
    expected_attempts = {
        (unit["unitId"], url) for unit in config["sourceUnits"] for url in unit["urls"]
    }
    actual_attempts = {(row["unit_id"], row["url"]) for row in attempts}
    successful = [row for row in attempts if row["success"]]
    successful_by_unit = {
        unit_id: {row["url"] for row in successful if row["unit_id"] == unit_id}
        for unit_id in frozen_units
    }
    snapshot_checks = []
    for row in successful:
        path = Path(row["snapshot_path"])
        snapshot_checks.append(
            path.is_file()
            and file_sha256(path) == row["sha256"]
            and path.stat().st_size == row["byte_count"]
        )
    assessment_coverage: list[bool] = []
    true_claim_coverage: list[bool] = []
    eligible: list[str] = []
    unit_metrics: dict[str, Any] = {}
    requirements = config["eligibilityRequirements"]
    for unit_id, frozen in frozen_units.items():
        record = evidence_units.get(unit_id, {})
        assessments = record.get("assessments", {})
        assessment_links = record.get("assessment_evidence_urls", {})
        assessment_coverage.append(set(assessments) == dimensions and set(assessment_links) == dimensions)
        for dimension in dimensions:
            if assessments.get(dimension) is True:
                cited = assessment_links.get(dimension, [])
                true_claim_coverage.append(
                    bool(cited)
                    and all(url in frozen["urls"] for url in cited)
                    and any(url in successful_by_unit[unit_id] for url in cited)
                )
        claimed_events = set(record.get("documented_event_categories", []))
        event_links = record.get("event_category_evidence_urls", {})
        valid_event_schema = claimed_events <= event_categories and set(event_links) == claimed_events
        for category in claimed_events:
            cited = event_links.get(category, [])
            true_claim_coverage.append(
                bool(cited)
                and all(url in frozen["urls"] for url in cited)
                and any(url in successful_by_unit[unit_id] for url in cited)
            )
        claimed_ambiguity = set(record.get("ambiguity_indicators", []))
        ambiguity_links = record.get("ambiguity_indicator_evidence_urls", {})
        valid_ambiguity_schema = claimed_ambiguity <= ambiguity_indicators and set(ambiguity_links) == claimed_ambiguity
        for indicator in claimed_ambiguity:
            cited = ambiguity_links.get(indicator, [])
            true_claim_coverage.append(
                bool(cited)
                and all(url in frozen["urls"] for url in cited)
                and any(url in successful_by_unit[unit_id] for url in cited)
            )
        releases = record.get("historical_releases", [])
        valid_releases = []
        asset_sizes = []
        for release in releases:
            metadata_url = release.get("metadata_url")
            assets = release.get("bounded_assets", [])
            release_valid = bool(
                release.get("release_id")
                and metadata_url in frozen["urls"]
                and metadata_url in successful_by_unit[unit_id]
                and assets
                and all(
                    isinstance(asset.get("byte_count"), int)
                    and 0 < asset["byte_count"] <= requirements["maximumSingleAssetBytes"]
                    and urlparse(asset.get("url", "")).scheme == "https"
                    and asset.get("format") in {"OBO", "OWL", "JSON"}
                    for asset in assets
                )
            )
            valid_releases.append(release_valid)
            if release_valid:
                asset_sizes.append(min(asset["byte_count"] for asset in assets))
        distinct_release_count = len({release.get("release_id") for release in releases if release.get("release_id")})
        two_release_bytes = sum(sorted(asset_sizes)[:2]) if len(asset_sizes) >= 2 else None
        missing_dimensions = [
            dimension for dimension in requirements["requiredDimensions"] if assessments.get(dimension) is not True
        ]
        text_event = bool(claimed_events & set(config["textEventCategories"]))
        lifecycle_event = bool(claimed_events & set(config["lifecycleOrMappingEventCategories"]))
        eligible_for_source = bool(
            not missing_dimensions
            and valid_event_schema
            and valid_ambiguity_schema
            and len(claimed_events) >= requirements["minimumDocumentedEventCategoryCount"]
            and (text_event or not requirements["requireTextEventCategory"])
            and (lifecycle_event or not requirements["requireLifecycleOrMappingEventCategory"])
            and len(claimed_ambiguity) >= requirements["minimumAmbiguityIndicatorCount"]
            and distinct_release_count >= requirements["minimumExactHistoricalReleaseCount"]
            and len(valid_releases) == len(releases)
            and two_release_bytes is not None
            and two_release_bytes <= requirements["maximumTwoReleasePayloadBytes"]
        )
        if eligible_for_source:
            eligible.append(unit_id)
        unit_metrics[unit_id] = {
            "selection_priority": frozen["selectionPriority"],
            "successful_url_count": len(successful_by_unit[unit_id]),
            "true_dimension_count": sum(value is True for value in assessments.values()),
            "missing_required_dimensions": missing_dimensions,
            "documented_event_categories": sorted(claimed_events),
            "event_category_count": len(claimed_events),
            "has_text_event": text_event,
            "has_lifecycle_or_mapping_event": lifecycle_event,
            "ambiguity_indicators": sorted(claimed_ambiguity),
            "ambiguity_indicator_count": len(claimed_ambiguity),
            "distinct_historical_release_count": distinct_release_count,
            "valid_historical_release_count": sum(valid_releases),
            "minimum_two_release_payload_bytes": two_release_bytes,
            "eligible": eligible_for_source,
        }
    priority = {unit["unitId"]: unit["selectionPriority"] for unit in config["sourceUnits"]}
    expected_selected = [min(eligible, key=lambda unit_id: priority[unit_id])] if eligible else []
    recommended = evidence.get("recommended_source_ids", [])
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
        "true_claim_evidence_coverage": _rate(true_claim_coverage),
        "eligible_source_ids": sorted(eligible),
        "eligible_source_count": len(eligible),
        "selected_source_ids": recommended,
        "selected_source_count": len(recommended),
        "expected_selected_source_ids": expected_selected,
        "selection_priority_correctness": 1.0 if recommended == expected_selected else 0.0,
        "retrospective_not_speaker_intent_boundary": bool(
            evidence["recommendation"]["retrospective_artifact_not_new_speaker_intent"]
        ),
        "total_term_count_used_for_eligibility": False,
        "unit_metrics": unit_metrics,
    }
    metrics["finite_metrics"] = _finite(metrics)
    return metrics


def audit_census(metrics: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["censusGates"]
    checks = {
        "source_units_and_URL_attempts_fully_accounted": bool(
            metrics["source_unit_count"] == gates["requiredSourceUnitCount"]
            and metrics["frozen_url_attempt_count"] == gates["requiredFrozenURLAttemptCount"]
            and metrics["recorded_url_attempt_count"] == gates["requiredFrozenURLAttemptCount"]
            and metrics["url_accounting_rate"] == gates["requiredURLAccountingRate"]
            and metrics["unexpected_url_attempt_count"] == 0
        ),
        "metadata_snapshots_and_claim_evidence_are_complete": bool(
            metrics["successful_snapshot_hash_coverage"] == gates["requiredSuccessfulSnapshotHashCoverage"]
            and metrics["assessment_dimension_coverage"] == gates["requiredAssessmentDimensionCoverage"]
            and metrics["true_claim_evidence_coverage"] == gates["requiredTrueClaimEvidenceCoverage"]
        ),
        "at_least_one_source_is_eligible_and_at_most_one_is_selected": bool(
            metrics["eligible_source_count"] >= gates["minimumEligibleSourceCount"]
            and metrics["selected_source_count"] <= gates["maximumSelectedSourceCount"]
            and metrics["selection_priority_correctness"] == gates["requiredSelectionPriorityCorrectness"]
        ),
        "claim_and_no_term_threshold_boundaries_hold": bool(
            metrics["retrospective_not_speaker_intent_boundary"] == gates["requiredRetrospectiveNotSpeakerIntentBoundary"]
            and not metrics["total_term_count_used_for_eligibility"]
        ),
        "metrics_are_finite": metrics["finite_metrics"] == gates["requiredFiniteMetrics"],
    }
    limits = config["accessGates"]
    access_checks = {
        "one_metadata_capture_and_census": bool(
            access["metadata_capture_run_count"] == limits["requiredMetadataCaptureRunCount"]
            and access["metadata_census_run_count"] == limits["requiredMetadataCensusRunCount"]
        ),
        "payload_protected_model_and_effect_boundaries_zero": bool(
            access["candidate_payload_download_count"] <= limits["maximumCandidatePayloadDownloadCount"]
            and access["v216_protected_access_count"] <= limits["maximumV216ProtectedAccessCount"]
            and access["v213_protected_access_count"] <= limits["maximumV213ProtectedAccessCount"]
            and access["model_load_count"] <= limits["maximumModelLoadCount"]
            and access["model_generation_count"] <= limits["maximumModelGenerationCount"]
            and access["model_api_call_count"] <= limits["maximumModelAPICallCount"]
            and access["training_run_count"] <= limits["maximumTrainingRunCount"]
            and access["ontology_registration_count"] <= limits["maximumOntologyRegistrationCount"]
            and access["trusted_state_mutation_count"] <= limits["maximumTrustedStateMutationCount"]
            and access["service_action_count"] <= limits["maximumServiceActionCount"]
            and access["external_side_effect_count_beyond_read_only_metadata"] <= limits["maximumExternalSideEffectCountBeyondReadOnlyMetadata"]
            and access["actual_execution_count"] <= limits["maximumActualExecutionCount"]
        ),
    }
    passed = all(checks.values()) and all(access_checks.values())
    return {
        "passed": passed,
        "branch": "FRESH_INDEPENDENT_SOURCE_PAYLOAD_DESIGN_ELIGIBLE" if passed else "STOP_EXTERNAL_RETROSPECTIVE_SOURCE_BRANCH",
        "decision": config["decisionRule"]["ifEveryIntegrityEligibilitySelectionAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "checks": checks,
        "access_checks": access_checks,
    }

