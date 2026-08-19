#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v223_archived_semantic_adjudication_metadata_census import metadata_only_url


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v223-archived-semantic-adjudication-metadata-census.json",
        "plan": PROJECT_ROOT / "docs/v223-archived-semantic-adjudication-metadata-census-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v221r1.md",
        "protocol": PROJECT_ROOT / "python/v223_archived_semantic_adjudication_metadata_census.py",
        "tests": PROJECT_ROOT / "python/test_v223_archived_semantic_adjudication_metadata_census.py",
        "capture": PROJECT_ROOT / "python/capture_v223_archived_semantic_adjudication_metadata.py",
        "runner": PROJECT_ROOT / "python/run_v223_archived_semantic_adjudication_metadata_census.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v223_archived_semantic_adjudication_metadata_census_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v223_archived_semantic_adjudication_metadata_census.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v223-archived-semantic-adjudication-metadata-census/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v223-archived-semantic-adjudication-metadata-census-lock.json"
    output_root = PROJECT_ROOT / "outputs/v223-archived-semantic-adjudication-metadata-census/census"
    snapshot_root = PROJECT_ROOT / "outputs/v223-archived-semantic-adjudication-metadata-census/metadata-snapshots"
    outcome_path = PROJECT_ROOT / "configs/v223-archived-semantic-adjudication-metadata-census-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, snapshot_root, outcome_path)):
        raise RuntimeError("V223 is already audited, frozen, captured, run, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    parents = {key: PROJECT_ROOT / value for key, value in config["parents"].items()}
    parent_values = {key: json.loads(path.read_text()) for key, path in parents.items()}
    units = config["sourceUnits"]
    urls = [(unit["unitId"], url) for unit in units for url in unit["urls"]]
    dimensions = set(config["dimensions"])
    exposure = config["priorExposure"]
    firewall = config["languageFirewall"]
    gates = config["censusGates"]
    github_units = [unit for unit in units if unit["unitId"] != "WIKIDATA_PROPERTY_PROPOSALS"]
    checks = {
        "parents_are_exact_and_establish_nonduplicate_branch_context": bool(
            all(valid_lock(value) for value in parent_values.values())
            and parent_values["V221r1Outcome"]["outcome"]["verification_passed"]
            and parent_values["V221r1Outcome"]["outcome"]["branch"]
            == "DETERMINISTIC_SUFFICIENT_CLOSE_MODEL_ESCALATION"
            and parent_values["V215Outcome"]["outcome"]["passed"]
            and parent_values["V215Outcome"]["outcome"]["branch"]
            == "BOUNDED_EXTERNAL_PAYLOAD_DESIGN_ELIGIBLE"
            and parent_values["V208Outcome"]["outcome"]["passed"]
            and not parent_values["V208Outcome"]["outcome"]["scientific_feasibility_passed"]
        ),
        "four_units_twenty_one_unique_metadata_only_URLs_are_frozen": bool(
            len(units) == 4
            and len(urls) == 21
            and len(set(urls)) == 21
            and all(metadata_only_url(url) for _, url in urls)
            and all(re.fullmatch(r"[0-9a-f]{40}", unit["pinnedRevision"]) for unit in github_units)
        ),
        "seventeen_noncompensatory_dimensions_and_branch_gates_are_complete": bool(
            len(dimensions) == 17
            and gates["requiredSourceUnitCount"] == 4
            and gates["requiredFrozenURLAttemptCount"] == 21
            and gates["requiredURLAccountingRate"] == 1.0
            and gates["requiredSuccessfulSnapshotHashCoverage"] == 1.0
            and gates["requiredAssessmentDimensionCoverage"] == 1.0
            and gates["requiredTrueAssessmentEvidenceCoverage"] == 1.0
            and gates["minimumEligibleSourceSpecificAcquisitionCandidateCount"] == 1
            and gates["maximumSelectedCandidateMissingMandatoryDimensionCount"] == 0
        ),
        "language_firewall_is_closed_to_every_task_record_endpoint": bool(
            all(firewall[key] for key in (
                "allowRepositoryMetadata", "allowWorkflowDocumentation", "allowIssueTemplates",
                "allowLabelVocabularies", "allowReleaseAndTagMetadata",
                "allowPolicyTemplateModuleAndCategoryMetadata"
            ))
            and not any(firewall[key] for key in (
                "allowIssueOrProposalRecordEndpoint",
                "allowCommentPullRequestDiscussionOrArchiveRecordEndpoint",
                "allowRecordTitleEnumeration", "allowTaskRecordBody"
            ))
        ),
        "discovery_exposure_is_disclosed_and_formal_record_access_is_zero": bool(
            not exposure["blindSourceDiscoveryClaim"]
            and exposure["discoverySearchSnippetsWereSeen"]
            and exposure["formalTaskRecordBodyReadCount"] == 0
            and exposure["formalTaskRecordScoreCount"] == 0
            and exposure["formalIssueProposalCommentOrPullPayloadDownloadCount"] == 0
            and len(exposure["excludedFromEverySuccessorPopulation"]) >= 13
        ),
        "successor_authority_is_narrow_and_model_effect_boundaries_remain_closed": bool(
            config["successorContract"]["metadataFirstBeforeRequestLanguage"]
            and config["successorContract"]["deterministicControlsBeforeAnyModel"]
            and not config["decisionRule"]["passAuthorizesTaskRecordLanguageAccess"]
            and not config["decisionRule"]["passAuthorizesModelOrAPI"]
            and not config["decisionRule"]["passAuthorizesRegistrationMutationActionOrExecution"]
        ),
        "all_required_files_exist_and_formal_outputs_are_absent": bool(
            all(path.is_file() for path in (*paths.values(), *parents.values()))
            and not output_root.exists()
            and not snapshot_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "223-archived-semantic-adjudication-metadata-census-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_metadata_only_capture_and_census" if passed else "reject_V223_design",
        "checks": checks,
        "formal_task_record_body_read_count_before_lock": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, **{f"parent_{key}": path for key, path in parents.items()}, "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "223-archived-semantic-adjudication-metadata-census-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "capture_frozen_metadata_URLs_once": True,
            "score_one_metadata_census": True,
            "open_task_record_language_or_record_endpoint": False,
            "run_model_train_register_mutate_service_act_execute": False,
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

