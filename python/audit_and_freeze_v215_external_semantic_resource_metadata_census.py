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
        "config": PROJECT_ROOT / "configs/v215-external-semantic-resource-metadata-census.json",
        "plan": PROJECT_ROOT / "docs/v215-external-semantic-resource-metadata-census-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v214.md",
        "protocol": PROJECT_ROOT / "python/v215_external_semantic_resource_metadata_census.py",
        "tests": PROJECT_ROOT / "python/test_v215_external_semantic_resource_metadata_census.py",
        "capture": PROJECT_ROOT / "python/capture_v215_external_resource_metadata.py",
        "runner": PROJECT_ROOT / "python/run_v215_external_semantic_resource_metadata_census.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v215_external_semantic_resource_metadata_census_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v215_external_semantic_resource_metadata_census.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v215-external-resource-metadata-census/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v215-external-semantic-resource-metadata-census-lock.json"
    output_root = PROJECT_ROOT / "outputs/v215-external-resource-metadata-census/census"
    snapshot_root = PROJECT_ROOT / "outputs/v215-external-resource-metadata-census/metadata-snapshots"
    outcome_path = PROJECT_ROOT / "configs/v215-external-semantic-resource-metadata-census-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, snapshot_root, outcome_path)):
        raise RuntimeError("V215 is already audited, frozen, captured, run, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV214OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    units = config["sourceUnits"]
    urls = [(unit["unitId"], url) for unit in units for url in unit["urls"]]
    dimensions = set(config["dimensions"])
    requirements = config["roleRequirements"]
    gates = config["censusGates"]
    exposure = config["preLockExposure"]
    official_domains = {"obofoundry.org", "www.ebi.ac.uk", "oaei.ontologymatching.org", "www.w3.org"}
    checks = {
        "V214_is_frozen_deterministic_closure_and_authorizes_non_model_successor": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["branch"] == "DETERMINISTIC_CLOSURE_ZERO_MODEL_ELIGIBILITY"
            and not parent["outcome"]["model_eligible"]
            and parent["authorization"]["design_next_non_model_stage"]
            and not parent["authorization"]["open_V213_protected_or_run_model_without_separate_lock"]
        ),
        "prior_high_level_exposure_disclosed_without_formal_scores_or_payload": bool(
            not config["priorExposure"]["blindSourceDiscoveryClaim"]
            and len(config["priorExposure"]["highLevelSourceFamiliesPreviouslyKnown"]) == 4
            and config["priorExposure"]["formalMetadataRecordScoreCount"] == 0
            and config["priorExposure"]["bulkOntologyAlignmentOrTestPayloadDownloadCount"] == 0
        ),
        "four_role_distinct_units_and_twelve_official_URL_attempts_frozen": bool(
            len(units) == 4
            and len(urls) == 12
            and len(set(urls)) == 12
            and {unit["intendedRole"] for unit in units}
            == {"PAYLOAD_BENCHMARK_CANDIDATE", "VALIDATION_CONTROL", "INFRASTRUCTURE_ONLY"}
            and all(urlparse(url).scheme == "https" and urlparse(url).netloc in official_domains for _, url in urls)
        ),
        "dimension_requirements_and_missing_evidence_rule_are_complete": bool(
            len(dimensions) == 10
            and set(requirements) == {"PAYLOAD_BENCHMARK_CANDIDATE", "VALIDATION_CONTROL", "INFRASTRUCTURE_ONLY"}
            and all(set(values) <= dimensions for values in requirements.values())
            and config["evidenceContract"]["missingAmbiguousOrUnsupportedScoresFalse"]
            and config["evidenceContract"]["generalAvailabilityDoesNotImplyLicense"]
            and config["evidenceContract"]["sha256EverySuccessfulSnapshot"]
        ),
        "noncompensatory_feasibility_and_role_gates_frozen": bool(
            gates["requiredSourceUnitCount"] == 4
            and gates["requiredFrozenURLAttemptCount"] == 12
            and gates["requiredURLAccountingRate"] == 1.0
            and gates["requiredSuccessfulSnapshotHashCoverage"] == 1.0
            and gates["requiredTrueAssessmentEvidenceCoverage"] == 1.0
            and gates["minimumEligiblePayloadBenchmarkCandidateCount"] == 1
            and gates["minimumEligibleValidationControlCount"] == 1
            and gates["maximumSelectedPayloadCandidateUnresolvedMandatoryDimensionCount"] == 0
        ),
        "prelock_exposure_zero_and_pass_authority_narrow": bool(
            all(value == 0 for value in exposure.values())
            and config["decisionRule"]["passAuthorizesPayloadDesignOnly"]
            and not config["decisionRule"]["passAuthorizesPayloadDownloadOrModelRun"]
            and not config["decisionRule"]["passAuthorizesRegistrationMutationServiceActionOrExecution"]
        ),
        "all_required_files_exist_and_census_outputs_absent": bool(
            all(path.is_file() for path in (*paths.values(), parent_path))
            and not output_root.exists()
            and not snapshot_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "215-external-semantic-resource-metadata-census-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_frozen_URL_metadata_capture_and_census" if passed else "reject_V215_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, "parent_V214_outcome": parent_path, "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "215-external-semantic-resource-metadata-census-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "capture_frozen_official_metadata_URLs_once": True,
            "score_one_frozen_metadata_census": True,
            "download_bulk_ontology_alignment_or_test_payload": False,
            "open_protected_run_model_register_mutate_service_act_execute": False,
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
