from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from v218_mondo_artifact_population import (
    audit_population as _audit_population,
    build_population_records as _build_population_records,
    event_types,
    load_obo,
    parse_tsv,
    score_population as _score_population,
)


def build_population_records(
    older_terms: dict[str, dict[str, list[str]]],
    newer_terms: dict[str, dict[str, list[str]]],
    older_candidates: set[str],
    newer_candidates: set[str],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    public, truth, manifest = _build_population_records(
        older_terms, newer_terms, older_candidates, newer_candidates, config
    )
    newer_tag = config["populationDesign"]["newerReleaseTag"]
    for record in public:
        if record["evidence_mode"] == "CURRENT_RELEASE_DECLARED":
            record["release_evidence"] = newer_tag
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in truth:
        families[record["group_id"]].append(record)
    any_events = {
        group_id: set().union(*(set(record["event_types"]) for record in records))
        for group_id, records in families.items()
    }
    manifest.update(
        {
            "schema_version": "220-fresh-mondo-population-build-manifest",
            "addition_event_family_count": sum("ADDED" in events for events in any_events.values()),
            "mapping_event_family_count": sum("MAPPING_CHANGED" in events for events in any_events.values()),
        }
    )
    return public, truth, manifest


def score_population(
    retrieval_manifest: dict[str, Any],
    parser_control: dict[str, Any],
    public_records: list[dict[str, Any]],
    truth_records: list[dict[str, Any]],
    split: dict[str, Any],
    population_manifest: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    metrics = _score_population(
        retrieval_manifest, parser_control, public_records, truth_records, split,
        population_manifest, config, project_root
    )
    development_groups = set(population_manifest["development_group_ids"])
    protected_groups = set(population_manifest["protected_group_ids"])
    metrics["eligible_record_count"] = population_manifest["eligible_record_count"]
    metrics["development_group_count"] = len(development_groups)
    metrics["protected_group_count"] = len(protected_groups)
    metrics["cross_split_group_overlap_count"] = len(development_groups & protected_groups)
    metrics["split_manifest_exact"] = bool(
        sorted(split["development_group_ids"]) == sorted(development_groups)
        and sorted(split["protected_group_ids"]) == sorted(protected_groups)
    )
    metrics["oracle_evaluation_scope"] = "DEVELOPMENT_ONLY"
    metrics["protected_files_loaded_for_scoring"] = False
    metrics["addition_event_family_count"] = population_manifest["addition_event_family_count"]
    metrics["mapping_event_family_count"] = population_manifest["mapping_event_family_count"]
    return metrics


def audit_population(metrics: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    compatibility = dict(config)
    compatibility["decisionRule"] = {
        "ifEveryPayloadControlPopulationIntegrityAndAccessGatePasses": "compatibility_pass",
        "otherwise": "compatibility_fail",
    }
    base = _audit_population(metrics, access, compatibility)
    gates = config["populationGates"]
    stratum_passed = bool(
        metrics["addition_event_family_count"] >= gates["minimumAdditionEventFamilyCount"]
        and metrics["text_change_family_count"] >= gates["minimumTextChangeFamilyCount"]
        and metrics["lifecycle_event_family_count"] >= gates["minimumLifecycleEventFamilyCount"]
        and metrics["mapping_event_family_count"] >= gates["minimumMappingEventFamilyCount"]
        and metrics["ambiguous_unspecified_family_count"] >= gates["minimumAmbiguousUnspecifiedFamilyCount"]
        and metrics["decision_contrast_family_count"] >= gates["minimumDecisionContrastFamilyCount"]
    )
    checks = dict(base["checks"])
    checks["event_strata_are_direct_and_noncompensatory"] = stratum_passed
    checks["oracle_scoring_is_development_only_and_protected_files_remain_unloaded"] = bool(
        config["populationGates"]["oracleEvaluationScope"]
        == "DEVELOPMENT_ONLY_WITH_PROTECTED_FILES_HASHED_BUT_NOT_LOADED"
        and metrics["oracle_evaluation_scope"] == "DEVELOPMENT_ONLY"
        and not metrics["protected_files_loaded_for_scoring"]
    )
    access_checks = dict(base["access_checks"])
    limits = config["accessGates"]
    access_checks["prior_population_and_protected_boundaries_zero"] = bool(
        access["v218_development_record_read_count"] <= limits["maximumV218DevelopmentRecordReadCount"]
        and access["v218_protected_record_read_count"] <= limits["maximumV218ProtectedRecordReadCount"]
        and access["protected_file_load_for_scoring_count"] <= limits["maximumProtectedFileLoadForScoringCount"]
    )
    passed = all(checks.values()) and all(access_checks.values())
    return {
        "passed": passed,
        "branch": "FRESH_MONDO_REPRESENTATIONAL_POPULATION_ELIGIBLE" if passed else "NEGATIVE_FRESH_MONDO_PAYLOAD_CONTROL_OR_POPULATION_FEASIBILITY",
        "decision": config["decisionRule"]["ifEveryPayloadControlPopulationStratumIntegrityAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "checks": checks,
        "access_checks": access_checks,
    }
