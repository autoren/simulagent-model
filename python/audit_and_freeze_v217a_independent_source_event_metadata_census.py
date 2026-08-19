#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v217a-independent-source-event-metadata-census.json",
        "plan": PROJECT_ROOT / "docs/v217a-independent-source-event-metadata-census-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v216.md",
        "protocol": PROJECT_ROOT / "python/v217a_independent_source_event_metadata_census.py",
        "tests": PROJECT_ROOT / "python/test_v217a_independent_source_event_metadata_census.py",
        "capture": PROJECT_ROOT / "python/capture_v217a_independent_source_event_metadata.py",
        "runner": PROJECT_ROOT / "python/run_v217a_independent_source_event_metadata_census.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v217a_independent_source_event_metadata_census_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v217a_independent_source_event_metadata_census.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v217a-independent-source-event-metadata-census/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v217a-independent-source-event-metadata-census-lock.json"
    output_root = PROJECT_ROOT / "outputs/v217a-independent-source-event-metadata-census"
    outcome_path = PROJECT_ROOT / "configs/v217a-independent-source-event-metadata-census-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V217A is already audited, frozen, captured, run, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV216r1OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    units = config["sourceUnits"]
    urls = [(unit["unitId"], url) for unit in units for url in unit["urls"]]
    dimensions = set(config["dimensions"])
    requirements = config["eligibilityRequirements"]
    gates = config["censusGates"]
    exposure = config["priorExposure"]
    official_domains = {
        "obofoundry.org", "api.github.com", "mondo.monarchinitiative.org",
        "geneontology.org", "obophenotype.github.io",
    }
    checks = {
        "V216_negative_is_frozen_and_authorizes_only_non_V217_roadmap_selection": bool(
            valid_lock(parent)
            and parent["outcome"]["repair_passed"]
            and not parent["outcome"]["V216_scientific_passed"]
            and parent["outcome"]["V216_branch"] == "NEGATIVE_EXTERNAL_PAYLOAD_OR_POPULATION_FEASIBILITY"
            and parent["authorization"]["select_post_V216_negative_non_V217_roadmap"]
            and not parent["authorization"]["design_V217_deterministic_external_reconstruction_controls"]
        ),
        "three_independent_units_nine_official_metadata_URLs_and_priority_are_frozen": bool(
            len(units) == 3
            and len(urls) == 9
            and len(set(urls)) == 9
            and {unit["unitId"] for unit in units} == {"MONDO", "GENE_ONTOLOGY", "CELL_ONTOLOGY"}
            and sorted(unit["selectionPriority"] for unit in units) == [1, 2, 3]
            and all(urlparse(url).scheme == "https" and urlparse(url).netloc in official_domains for _, url in urls)
        ),
        "event_ambiguity_license_asset_and_no_term_threshold_requirements_are_complete": bool(
            len(dimensions) == 13
            and set(requirements["requiredDimensions"]) == dimensions
            and len(config["eventCategories"]) == 6
            and len(config["ambiguityIndicators"]) == 5
            and requirements["minimumExactHistoricalReleaseCount"] == 2
            and requirements["minimumDocumentedEventCategoryCount"] == 3
            and requirements["requireTextEventCategory"]
            and requirements["requireLifecycleOrMappingEventCategory"]
            and requirements["minimumAmbiguityIndicatorCount"] == 1
            and not requirements["totalOntologyTermCountIsEligibilityCriterion"]
        ),
        "evidence_and_selection_rules_are_noncompensatory_and_frozen": bool(
            config["evidenceContract"]["missingAmbiguousOrUnsupportedScoresFalse"]
            and config["evidenceContract"]["generalAvailabilityDoesNotImplyLicense"]
            and config["evidenceContract"]["assetURLAndByteCountMustAppearInFrozenMetadata"]
            and config["evidenceContract"]["doNotFollowAssetDownloadLinks"]
            and config["selectionRule"]["selectAtMostOne"]
            and config["selectionRule"]["selectLowestFrozenPriorityAmongEligible"]
            and config["selectionRule"]["priorityOrder"] == ["MONDO", "GENE_ONTOLOGY", "CELL_ONTOLOGY"]
            and config["selectionRule"]["doNotUseV216PayloadOrProtectedRecordsForSelection"]
        ),
        "census_gates_require_one_eligible_source_but_at_most_one_selection": bool(
            gates["requiredSourceUnitCount"] == 3
            and gates["requiredFrozenURLAttemptCount"] == 9
            and gates["requiredSuccessfulSnapshotHashCoverage"] == 1.0
            and gates["requiredTrueClaimEvidenceCoverage"] == 1.0
            and gates["minimumEligibleSourceCount"] == 1
            and gates["maximumSelectedSourceCount"] == 1
            and gates["requiredSelectionPriorityCorrectness"] == 1.0
        ),
        "prelock_source_specific_scores_payload_and_protected_access_are_zero": bool(
            not exposure["blindFamilyDiscoveryClaim"]
            and exposure["sourceSpecificReleaseMetadataReadCount"] == 0
            and exposure["candidatePayloadBodyByteReadCount"] == 0
            and exposure["formalSourceAssessmentCount"] == 0
            and exposure["V216ProtectedReadCount"] == 0
        ),
        "pass_authority_is_narrow_and_all_outputs_absent": bool(
            config["decisionRule"]["passAuthorizesPayloadDesignOnly"]
            and not config["decisionRule"]["passAuthorizesPayloadDownloadOrModelRun"]
            and not config["decisionRule"]["passAuthorizesV216V217ProtectedUseRegistrationMutationActionOrExecution"]
            and all(path.is_file() for path in (*paths.values(), parent_path))
            and not output_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "217a-independent-source-event-metadata-census-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_nine_URL_metadata_capture_and_census" if passed else "reject_V217A_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, "parent_V216r1_outcome": parent_path, "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "217a-independent-source-event-metadata-census-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "capture_nine_frozen_official_metadata_URLs_once": True,
            "score_one_frozen_source_event_census": True,
            "download_candidate_payload_or_open_V216_protected": False,
            "run_model_register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()

