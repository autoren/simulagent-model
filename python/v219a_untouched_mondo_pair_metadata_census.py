from __future__ import annotations

import hashlib
import math
import re
from typing import Any
from urllib.parse import urlparse


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


def body_sha256(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def release_body_categories(body: str, config: dict[str, Any]) -> dict[str, bool]:
    patterns = config["releaseBodyControl"]["categoryPatterns"]
    return {category: re.search(pattern, body) is not None for category, pattern in patterns.items()}


def ordered_releases(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(releases, key=lambda release: release["tag_name"])
    return sorted(ordered, key=lambda release: release["published_at"], reverse=True)


def enumerate_untouched_adjacent_pairs(
    releases: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    ordered = ordered_releases(releases)
    excluded = set(config["pairEnumeration"]["excludedReleaseTags"])
    pairs: list[dict[str, Any]] = []
    for index in range(len(ordered) - 1):
        newer = ordered[index]
        older = ordered[index + 1]
        if newer["tag_name"] in excluded or older["tag_name"] in excluded:
            continue
        pairs.append(
            {
                "pair_id": f'{older["tag_name"]}__to__{newer["tag_name"]}',
                "full_order_newer_index": index,
                "full_order_older_index": index + 1,
                "older_release": older,
                "newer_release": newer,
            }
        )
    pairs.sort(
        key=lambda pair: (
            pair["older_release"]["tag_name"], pair["newer_release"]["tag_name"]
        )
    )
    pairs.sort(key=lambda pair: pair["newer_release"]["published_at"], reverse=True)
    return pairs, [release["tag_name"] for release in ordered]


def _valid_digest(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _asset_for_name(release: dict[str, Any], name: str) -> tuple[dict[str, Any] | None, int]:
    matches = [asset for asset in release.get("assets", []) if asset.get("name") == name]
    return (matches[0] if len(matches) == 1 else None), len(matches)


def assess_pair(pair: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    requirements = config["assetRequirements"]
    role_records: list[dict[str, Any]] = []
    for role in config["requiredAssetRoles"]:
        side = role["releaseSide"]
        release = pair["older_release"] if side == "OLDER" else pair["newer_release"]
        asset, match_count = _asset_for_name(release, role["assetName"])
        if asset is None:
            role_records.append(
                {
                    "role": role["role"],
                    "release_side": side,
                    "release_tag": release["tag_name"],
                    "asset_name": role["assetName"],
                    "format": role["format"],
                    "match_count": match_count,
                    "valid": False,
                }
            )
            continue
        digest = (asset.get("digest") or "").removeprefix("sha256:")
        url = asset.get("browser_download_url", "")
        size = asset.get("size")
        valid = bool(
            match_count == 1
            and urlparse(url).scheme == "https"
            and isinstance(size, int)
            and size > 0
            and size <= requirements["maximumSingleAssetBytes"]
            and _valid_digest(digest)
        )
        role_records.append(
            {
                "role": role["role"],
                "release_side": side,
                "release_tag": release["tag_name"],
                "asset_name": role["assetName"],
                "format": role["format"],
                "match_count": match_count,
                "url": url,
                "byte_count": size,
                "declared_sha256": digest,
                "valid": valid,
            }
        )
    body = pair["newer_release"].get("body")
    body_is_string = isinstance(body, str)
    categories = release_body_categories(body if body_is_string else "", config)
    byte_counts = [record.get("byte_count", 0) for record in role_records if record.get("valid")]
    total_bytes = sum(byte_counts) if len(byte_counts) == len(role_records) else None
    original_adjacency = pair["full_order_older_index"] == pair["full_order_newer_index"] + 1
    eligible = bool(
        original_adjacency
        and len(role_records) == requirements["requiredAssetRoleCount"]
        and all(record["valid"] for record in role_records)
        and total_bytes is not None
        and total_bytes <= requirements["maximumPairPayloadBytes"]
        and body_is_string
        and all(categories.values())
    )
    return {
        "pair_id": pair["pair_id"],
        "older_release_tag": pair["older_release"]["tag_name"],
        "newer_release_tag": pair["newer_release"]["tag_name"],
        "older_published_at": pair["older_release"]["published_at"],
        "newer_published_at": pair["newer_release"]["published_at"],
        "full_order_newer_index": pair["full_order_newer_index"],
        "full_order_older_index": pair["full_order_older_index"],
        "original_adjacency": original_adjacency,
        "asset_roles": role_records,
        "total_pair_payload_bytes": total_bytes,
        "release_body_control": {
            "source": "body_field_of_newer_release_object_in_frozen_API_snapshot",
            "newer_release_tag": pair["newer_release"]["tag_name"],
            "body_sha256": body_sha256(body) if body_is_string else None,
            "category_matches": categories,
            "repository_readme_used": False,
        },
        "eligible": eligible,
    }


def build_census(
    releases: list[dict[str, Any]], config: dict[str, Any], *, snapshot_hash_accurate: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs, full_order = enumerate_untouched_adjacent_pairs(releases, config)
    assessments = [assess_pair(pair, config) for pair in pairs]
    eligible = [assessment for assessment in assessments if assessment["eligible"]]
    selected = eligible[:1] if config["pairEnumeration"]["selectFirstEligiblePair"] else []
    expected_selection = eligible[:1]
    roles = [role for assessment in assessments for role in assessment["asset_roles"]]
    categories = [
        matched
        for assessment in assessments
        for matched in assessment["release_body_control"]["category_matches"].values()
    ]
    evidence = {
        "schema_version": "219a-untouched-mondo-pair-metadata-census-evidence",
        "experiment": config["experiment"],
        "evidence_source_id": config["evidenceSource"]["sourceId"],
        "evidence_snapshot_path": config["evidenceSource"]["path"],
        "evidence_snapshot_sha256": config["evidenceSource"]["sha256"],
        "full_release_order": full_order,
        "excluded_release_tags": config["pairEnumeration"]["excludedReleaseTags"],
        "adjacency_computed_before_exclusion": True,
        "pair_assessments": assessments,
        "eligible_pair_ids": [assessment["pair_id"] for assessment in eligible],
        "selected_pair_ids": [assessment["pair_id"] for assessment in selected],
        "selected_pair_assessments": selected,
        "claim_boundary": {
            "retrospective_artifact_not_new_speaker_intent": True,
            "payload_body_or_ontology_record_read": False,
        },
    }
    expected_pair_count = config["pairEnumeration"]["expectedUntouchedAdjacentPairCount"]
    metrics: dict[str, Any] = {
        "evidence_snapshot_hash_accuracy": 1.0 if snapshot_hash_accurate else 0.0,
        "release_count": len(releases),
        "excluded_release_count": len(config["pairEnumeration"]["excludedReleaseTags"]),
        "untouched_adjacent_pair_count": len(pairs),
        "original_adjacency_accuracy": _rate([assessment["original_adjacency"] for assessment in assessments]),
        "pair_assessment_coverage": len(assessments) / expected_pair_count if expected_pair_count else 1.0,
        "asset_role_coverage": _rate([record["match_count"] == 1 and record["valid"] for record in roles]),
        "asset_digest_coverage": _rate([_valid_digest(record.get("declared_sha256", "")) for record in roles]),
        "release_body_hash_coverage": _rate(
            [_valid_digest(assessment["release_body_control"].get("body_sha256") or "") for assessment in assessments]
        ),
        "release_body_category_coverage": _rate(categories),
        "eligible_pair_ids": [assessment["pair_id"] for assessment in eligible],
        "eligible_pair_count": len(eligible),
        "selected_pair_ids": [assessment["pair_id"] for assessment in selected],
        "selected_pair_count": len(selected),
        "expected_selected_pair_ids": [assessment["pair_id"] for assessment in expected_selection],
        "selection_priority_correctness": 1.0 if selected == expected_selection else 0.0,
        "retrospective_not_speaker_intent_boundary": True,
        "pair_metrics": {
            assessment["pair_id"]: {
                "eligible": assessment["eligible"],
                "total_pair_payload_bytes": assessment["total_pair_payload_bytes"],
                "release_body_categories": assessment["release_body_control"]["category_matches"],
            }
            for assessment in assessments
        },
    }
    metrics["finite_metrics"] = _finite(metrics)
    return evidence, metrics


def audit_census(metrics: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["censusGates"]
    checks = {
        "snapshot_release_exclusion_and_pair_accounting_are_exact": bool(
            metrics["evidence_snapshot_hash_accuracy"] == gates["requiredEvidenceSnapshotHashAccuracy"]
            and metrics["release_count"] == gates["requiredReleaseCount"]
            and metrics["excluded_release_count"] == gates["requiredExcludedReleaseCount"]
            and metrics["untouched_adjacent_pair_count"] == gates["requiredUntouchedAdjacentPairCount"]
        ),
        "original_adjacency_and_pair_assessment_are_complete": bool(
            metrics["original_adjacency_accuracy"] == gates["requiredOriginalAdjacencyAccuracy"]
            and metrics["pair_assessment_coverage"] == gates["requiredPairAssessmentCoverage"]
        ),
        "asset_roles_digests_and_official_release_body_controls_are_complete": bool(
            metrics["asset_role_coverage"] == gates["requiredAssetRoleCoverage"]
            and metrics["asset_digest_coverage"] == gates["requiredAssetDigestCoverage"]
            and metrics["release_body_hash_coverage"] == gates["requiredReleaseBodyHashCoverage"]
            and metrics["release_body_category_coverage"] == gates["requiredReleaseBodyCategoryCoverage"]
        ),
        "eligible_pair_and_selection_priority_gates_hold": bool(
            metrics["eligible_pair_count"] >= gates["minimumEligiblePairCount"]
            and metrics["selected_pair_count"] <= gates["maximumSelectedPairCount"]
            and metrics["selection_priority_correctness"] == gates["requiredSelectionPriorityCorrectness"]
        ),
        "claim_boundary_and_finite_metrics_hold": bool(
            metrics["retrospective_not_speaker_intent_boundary"]
            == gates["requiredRetrospectiveNotSpeakerIntentBoundary"]
            and metrics["finite_metrics"] == gates["requiredFiniteMetrics"]
        ),
    }
    limits = config["accessGates"]
    access_checks = {
        "exactly_one_local_metadata_census_and_snapshot_read": bool(
            access["metadata_census_run_count"] == limits["requiredMetadataCensusRunCount"]
            and access["evidence_snapshot_read_count"] == limits["requiredEvidenceSnapshotReadCount"]
        ),
        "network_payload_protected_method_model_and_effect_boundaries_are_zero": bool(
            access["network_request_count"] <= limits["maximumNetworkRequestCount"]
            and access["new_payload_body_read_count"] <= limits["maximumNewPayloadBodyReadCount"]
            and access["new_ontology_term_or_axiom_record_read_count"] <= limits["maximumNewOntologyTermOrAxiomRecordReadCount"]
            and access["v218_development_record_read_count"] <= limits["maximumV218DevelopmentRecordReadCount"]
            and access["v218_protected_record_read_count"] <= limits["maximumV218ProtectedRecordReadCount"]
            and access["v216_protected_access_count"] <= limits["maximumV216ProtectedAccessCount"]
            and access["v213_protected_access_count"] <= limits["maximumV213ProtectedAccessCount"]
            and access["deterministic_method_evaluation_count"] <= limits["maximumDeterministicMethodEvaluationCount"]
            and access["model_load_count"] <= limits["maximumModelLoadCount"]
            and access["model_generation_count"] <= limits["maximumModelGenerationCount"]
            and access["model_api_call_count"] <= limits["maximumModelAPICallCount"]
            and access["training_run_count"] <= limits["maximumTrainingRunCount"]
            and access["ontology_registration_count"] <= limits["maximumOntologyRegistrationCount"]
            and access["trusted_state_mutation_count"] <= limits["maximumTrustedStateMutationCount"]
            and access["service_action_count"] <= limits["maximumServiceActionCount"]
            and access["external_side_effect_count"] <= limits["maximumExternalSideEffectCount"]
            and access["actual_execution_count"] <= limits["maximumActualExecutionCount"]
        ),
    }
    passed = all(checks.values()) and all(access_checks.values())
    return {
        "passed": passed,
        "branch": "UNTOUCHED_MONDO_PAIR_PAYLOAD_DESIGN_ELIGIBLE" if passed else "STOP_FRESH_MONDO_PAYLOAD_BRANCH",
        "decision": config["decisionRule"]["ifEveryMetadataPairControlSelectionAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "checks": checks,
        "access_checks": access_checks,
    }
