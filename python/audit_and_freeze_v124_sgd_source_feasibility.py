#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v124-sgd-source-feasibility.json"
    metadata_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/source-metadata.json"
    plan_path = PROJECT_ROOT / "docs/v124-sgd-source-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v124_sgd_source_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v124_sgd_source_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v124_sgd_source_feasibility.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v124_sgd_source_feasibility_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v124_sgd_source_feasibility.py"
    audit_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v124-sgd-source-feasibility-lock.json"
    archive_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/source/sgd-pinned.tar.gz"
    inventory_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/source-inventory/sgd-open-set-inventory.json"
    if any(path.exists() for path in (audit_path, lock_path, archive_path, inventory_path)):
        raise RuntimeError("V124 already frozen, downloaded, or evaluated")

    config = json.loads(config_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV123OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    auth = parent["authorization"]
    checks = {
        "V123_is_valid_and_authorizes_only_external_source_audit": bool(
            valid_lock(parent) and valid_lock(parent_lock)
            and parent["outcome"]["passed"] and parent["outcome"]["audit_pass"]
            and auth["preregister_external_controlled_open_set_source_feasibility_audit"]
            and not auth["reuse_prior_or_protected_identifiers"]
            and not auth["evaluate_language_signal_trigger_or_model"]
            and not auth["run_API_training_action_or_execution"]
        ),
        "official_source_revision_archive_and_license_metadata_are_pinned": bool(
            metadata["repository"] == "google-research-datasets/dstc8-schema-guided-dialogue"
            and metadata["repository_head_revision"] == config["revision"]
            and metadata["archive_url"] == config["archiveUrl"]
            and metadata["head_status"] == 200
            and metadata["license_claim"] == config["license"] == "CC BY-SA 4.0"
            and metadata["repository_archived_read_only"]
        ),
        "prelock_exposure_is_metadata_only": bool(
            config["preLockExposure"]["metadataOnlyRequestCount"] == metadata["metadata_only_request_count"] == 2
            and config["preLockExposure"]["archivePayloadDownloadCount"] == metadata["archive_payload_download_count"] == 0
            and config["preLockExposure"]["languageRecordInspectionCount"] == 0
            and config["preLockExposure"]["modelLoadCount"] == 0
            and config["preLockExposure"]["modelGenerationCount"] == 0
        ),
        "source_run_is_text_free_non_model_and_non_executable": bool(
            not config["candidateDefinition"]["emitUtteranceOrSlotValues"]
            and not config["candidateDefinition"]["manualLanguageInspection"]
            and config["sourceGates"]["maximumEmittedLanguageRecordCount"] == 0
            and config["sourceGates"]["maximumManualLanguageInspectionCount"] == 0
            and config["sourceGates"]["maximumModelLoadCount"] == 0
            and config["sourceGates"]["maximumModelGenerationCount"] == 0
            and config["sourceGates"]["maximumActualExecutionCount"] == 0
        ),
        "success_authorizes_only_text_free_catalog_and_population_preregistration": bool(
            config["decisionRule"]["passAuthorizesOnlyTextFreeCatalogAndPopulationPreregistration"]
            and not config["decisionRule"]["passAuthorizesImmediateSelectedLanguageSignalTriggerOrModelEvaluation"]
            and not config["decisionRule"]["passAuthorizesProtectedInductionAPITrainingActionOrExecution"]
        ),
        "code_metadata_and_output_absence_hold": bool(
            metadata_path.is_file()
            and all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path))
            and not archive_path.exists() and not inventory_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "124-sgd-source-feasibility-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "decision": "freeze_and_authorize_one_pinned_source_download_and_inventory" if passed else "reject_V124_design",
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    deps = {
        "config": config_path,
        "source_metadata": metadata_path,
        "parent_outcome": parent_path,
        "parent_analysis_lock": parent_lock_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "124-sgd-source-feasibility-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "download_one_pinned_archive_and_build_one_text_free_inventory": True,
            "modify_source_revision_candidate_definition_gates_or_decision": False,
            "emit_or_manually_inspect_language_or_slot_values": False,
            "evaluate_retrieval_signal_trigger_or_model": False,
            "grant_protected_induction_authority_or_execution": False,
        },
    }
    for key, path in deps.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
