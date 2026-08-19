#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v224_graphql_queries import NODE_QUERY, RECORD_QUERY, RELEASE_QUERY, forbidden_selected_fields


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v224-mondo-record-disposition-metadata-census.json",
        "plan": PROJECT_ROOT / "docs/v224-mondo-record-disposition-metadata-census-plan.md",
        "queries": PROJECT_ROOT / "python/v224_graphql_queries.py",
        "protocol": PROJECT_ROOT / "python/v224_mondo_record_disposition_metadata_census.py",
        "tests": PROJECT_ROOT / "python/test_v224_mondo_record_disposition_metadata_census.py",
        "capture": PROJECT_ROOT / "python/capture_v224_mondo_record_disposition_metadata.py",
        "runner": PROJECT_ROOT / "python/run_v224_mondo_record_disposition_metadata_census.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v224_mondo_record_disposition_metadata_census_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v224_mondo_record_disposition_metadata_census.py",
    }
    output_root = PROJECT_ROOT / "outputs/v224-mondo-record-disposition-metadata-census"
    audit_path = output_root / "design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v224-mondo-record-disposition-metadata-census-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v224-mondo-record-disposition-metadata-census-outcome-lock.json"
    if output_root.exists() or lock_path.exists() or outcome_path.exists():
        raise RuntimeError("V224 is already audited, run, or frozen")
    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV223r1Outcome"]
    parent = json.loads(parent_path.read_text())
    source = config["source"]
    prior = config["priorExposure"]
    strata = config["dispositionContract"]["substantiveStrata"]
    query_failures = {
        "record": forbidden_selected_fields(RECORD_QUERY),
        "node": forbidden_selected_fields(NODE_QUERY),
        "release": forbidden_selected_fields(RELEASE_QUERY),
    }
    start = datetime.fromisoformat(source["creationWindowStart"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(source["creationWindowEnd"].replace("Z", "+00:00"))
    checks = {
        "V223r1_parent_is_exact_positive_and_authorizes_V224_design_only": bool(
            valid_lock(parent)
            and parent["outcome"]["V223_scientific_passed"]
            and parent["outcome"]["V223_branch"] == "SOURCE_SPECIFIC_ACQUISITION_DESIGN_ELIGIBLE"
            and parent["authorization"]["design_V224_Mondo_metadata_first_record_disposition_census"]
            and not parent["authorization"]["open_task_record_language_or_run_model"]
        ),
        "source_window_revision_slices_and_exclusions_are_frozen": bool(
            source["repository"] == "monarch-initiative/mondo"
            and len(source["workflowRevision"]) == 40
            and start.year == 2021 and end.year == 2024
            and source["expectedSearchSliceCount"] == 48
            and prior["recordTitleOrBodyReadCount"] == 0
            and prior["formalRecordMetadataReadCount"] == 0
            and prior["excludedIssueNumbers"] == [503, 673, 10448]
        ),
        "GraphQL_queries_select_no_task_language": not any(query_failures.values()),
        "four_dispositions_are_noncompensatory_and_deep_gate_is_fixed": bool(
            strata == [
                "ACCEPTED_NEW", "EXISTING_OR_DUPLICATE", "INSUFFICIENT_OR_CLARIFY",
                "UNSUPPORTED_OR_OUT_OF_SCOPE",
            ]
            and config["samplingContract"]["minimumPreliminaryRecordsPerStratum"] == 24
            and config["samplingContract"]["minimumFinalRetainedRecordsPerStratum"] == 24
            and config["samplingContract"]["maximumDeepAuditRecordsPerStratum"] == 64
            and not config["samplingContract"]["poolAcrossStrata"]
            and config["dispositionContract"]["acceptedRequiresExactlyOneAddedMondoId"]
            and config["dispositionContract"]["duplicateRequiresCanonicalAcceptedTermReleasedBeforeDuplicateCreation"]
        ),
        "human_independence_AI_exclusions_and_language_firewall_are_exact": bool(
            config["humanActorContract"]["requesterAndAdjudicatorMustDiffer"]
            and config["humanActorContract"]["requesterAndMergerMustDiffer"]
            and len(config["labels"]["aiOrAutomationExclusions"]) == 7
            and not config["languageFirewall"]["allowIssueOrPullTitle"]
            and not config["languageFirewall"]["allowIssueOrPullBody"]
            and not config["languageFirewall"]["allowCommentOrReviewText"]
            and not config["languageFirewall"]["allowProtectedResearchRecord"]
            and not config["languageFirewall"]["allowModelOrAPIModel"]
        ),
        "decision_authorizes_only_later_design_and_all_files_exist": bool(
            not config["decisionRule"]["passAuthorizesTaskLanguageAccess"]
            and not config["decisionRule"]["passAuthorizesModelOrAPI"]
            and not config["decisionRule"]["passAuthorizesRegistrationMutationActionOrExecution"]
            and all(path.is_file() for path in paths.values())
            and parent_path.is_file()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "224-mondo-record-disposition-metadata-census-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_V224_and_authorize_one_metadata_only_census" if passed else "reject_V224_design",
        "checks": checks,
        "query_forbidden_selected_fields": query_failures,
        "formal_record_metadata_read_count_before_lock": 0,
        "task_record_title_or_body_read_count_before_lock": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, "parent_V223r1_outcome": parent_path, "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "224-mondo-record-disposition-metadata-census-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_one_metadata_only_record_disposition_census": True,
            "read_issue_or_pull_title_body_comment_or_review_text": False,
            "open_protected_or_run_model": False,
            "register_mutate_service_act_execute": False,
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

