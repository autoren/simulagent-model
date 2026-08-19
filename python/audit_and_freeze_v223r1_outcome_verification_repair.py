#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v223_archived_semantic_adjudication_metadata_census import dependency_hashes_exact
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v223r1_outcome_verification_repair import failed_checks, positive_outcome_matches


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v223r1-outcome-verification-repair.json",
        "plan": PROJECT_ROOT / "docs/v223r1-outcome-verification-repair-plan.md",
        "protocol": PROJECT_ROOT / "python/v223r1_outcome_verification_repair.py",
        "tests": PROJECT_ROOT / "python/test_v223r1_outcome_verification_repair.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v223r1_outcome_verification_repair.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v223r1_outcome_verification_repair.py",
    }
    output_root = PROJECT_ROOT / "outputs/v223r1-outcome-verification-repair"
    audit_path = output_root / "design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v223r1-outcome-verification-repair-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v223r1-outcome-verification-repair-outcome-lock.json"
    if output_root.exists() or lock_path.exists() or outcome_path.exists():
        raise RuntimeError("V223r1 is already audited or frozen")
    repair = json.loads(paths["config"].read_text())
    parent_lock_path = PROJECT_ROOT / repair["parentV223DesignLock"]
    failed_audit_path = PROJECT_ROOT / repair["parentV223FailedOutcomeAudit"]
    summary_path = PROJECT_ROOT / repair["parentV223Summary"]
    result_path = PROJECT_ROOT / repair["parentV223Result"]
    results_path = PROJECT_ROOT / repair["parentV223ResultsDocument"]
    parent = json.loads(parent_lock_path.read_text())
    failed_audit = json.loads(failed_audit_path.read_text())
    summary = json.loads(summary_path.read_text())
    result = json.loads(result_path.read_text())
    invariant = repair["repairInvariant"]
    exposure = repair["preLockExposure"]
    checks = {
        "parent_lock_is_valid_and_all_original_dependencies_are_restored": bool(
            valid_lock(parent)
            and file_sha256(parent_lock_path) == repair["parentV223DesignLockSha256"]
            and dependency_hashes_exact(parent)
        ),
        "original_failed_audit_is_exact_and_only_dependency_check_failed": bool(
            file_sha256(failed_audit_path) == repair["parentV223FailedOutcomeAuditSha256"]
            and failed_audit.get("passed") is False
            and failed_checks(failed_audit) == sorted(invariant["expectedOriginalFailedChecks"])
            and failed_audit.get("branch") == invariant["expectedV223Branch"]
            and failed_audit.get("formal_task_record_body_read_count") == 0
        ),
        "existing_scientific_summary_and_result_are_positive_and_exactly_bounded": bool(
            positive_outcome_matches(summary, result, invariant)
            and result["authorization"]["design_source_specific_metadata_first_acquisition_and_identifiability_stage"]
            and not result["authorization"]["open_task_record_language_or_run_model_without_separate_lock"]
        ),
        "repair_changes_no_science_and_authorizes_only_one_verification": bool(
            exposure["V223CensusRerunCount"] == 0
            and exposure["newMetadataRetrievalCount"] == 0
            and exposure["taskRecordBodyReadCount"] == 0
            and exposure["recordEndpointRequestCount"] == 0
            and exposure["sourceAssessmentChangeCount"] == 0
            and exposure["gateOrThresholdChangeCount"] == 0
            and exposure["modelLoadOrGenerationCount"] == 0
            and exposure["actualExecutionCount"] == 0
            and not repair["decisionRule"]["repairCanAlterScientificOutcome"]
            and not repair["decisionRule"]["repairCanOpenTaskLanguageOrRunModel"]
        ),
        "repair_files_and_results_document_exist": bool(
            all(path.is_file() for path in paths.values()) and results_path.is_file()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "223r1-outcome-verification-repair-design-audit",
        "experiment": repair["experiment"],
        "passed": passed,
        "decision": "freeze_V223r1_and_authorize_one_exact_outcome_verification" if passed else "reject_V223r1",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        **paths,
        "parent_V223_design_lock": parent_lock_path,
        "parent_V223_failed_outcome_audit": failed_audit_path,
        "parent_V223_summary": summary_path,
        "parent_V223_result": result_path,
        "parent_V223_results_document": results_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "223r1-outcome-verification-repair-lock",
        "experiment": repair["experiment"],
        "config_payload": repair,
        "authorization": {
            "verify_and_freeze_exact_existing_V223_positive_once": True,
            "retrieve_reassess_change_gate_open_language_or_run_model": False,
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

