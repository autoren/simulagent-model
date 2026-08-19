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
from v224_graphql_queries import NODE_QUERY, RECORD_QUERY, forbidden_selected_fields, selected_field_tokens
from v224r2_graphql_queries import DEEP_NODE_QUERY, RELEASE_QUERY, THIN_RECORD_QUERY


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v224r2-two-stage-query-execution-repair.json",
        "plan": PROJECT_ROOT / "docs/v224r2-two-stage-query-execution-repair-plan.md",
        "V224_initial_failure": PROJECT_ROOT / "docs/v224-initial-capture-failure.md",
        "V224r1_failure": PROJECT_ROOT / "docs/v224r1-capture-failure.md",
        "queries": PROJECT_ROOT / "python/v224r2_graphql_queries.py",
        "protocol": PROJECT_ROOT / "python/v224r2_two_stage_query_execution_repair.py",
        "tests": PROJECT_ROOT / "python/test_v224r2_two_stage_query_execution_repair.py",
        "capture": PROJECT_ROOT / "python/capture_v224r2_two_stage_query_execution_repair.py",
        "runner": PROJECT_ROOT / "python/run_v224r2_two_stage_query_execution_repair.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v224r2_two_stage_query_execution_repair_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v224r2_two_stage_query_execution_repair.py",
    }
    output_root = PROJECT_ROOT / "outputs/v224r2-two-stage-query-execution-repair"
    audit_path = output_root / "design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v224r2-two-stage-query-execution-repair-lock.json"
    if output_root.exists() or lock_path.exists():
        raise RuntimeError("V224r2 is already audited or locked")
    repair = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / repair["parentV224DesignLock"]
    r1_path = PROJECT_ROOT / repair["parentV224r1RepairLock"]
    parent = json.loads(parent_path.read_text())
    r1 = json.loads(r1_path.read_text())
    failure = repair["failureBoundary"]
    artifacts = {key: PROJECT_ROOT / value for key, value in parent["config_payload"]["artifacts"].items()}
    allowed_shape = {"query", "string"}
    checks = {
        "parent_V224_and_V224r1_locks_are_valid_exact": bool(
            valid_lock(parent) and valid_lock(r1)
            and dependency_hashes_exact(parent) and dependency_hashes_exact(r1)
            and file_sha256(parent_path) == repair["parentV224DesignLockSha256"]
            and file_sha256(r1_path) == repair["parentV224r1RepairLockSha256"]
        ),
        "two_failed_attempts_are_bounded_and_no_scientific_artifact_exists": bool(
            failure["V224FailedAttemptCount"] == 1 and failure["V224r1FailedAttemptCount"] == 1
            and failure["V224r1Exception"] == "HTTP_502"
            and failure["V224r1PersistedRecordMetadataCount"] == 0
            and failure["V224r1PersistedScientificArtifactCount"] == 0
            and failure["V224r1TaskLanguagePersistenceOrResearchExposureCount"] == 0
            and not any(path.exists() for key, path in artifacts.items() if key != "scopePolicySnapshot")
        ),
        "scope_snapshot_remains_exact": bool(
            artifacts["scopePolicySnapshot"].is_file()
            and file_sha256(artifacts["scopePolicySnapshot"]) == failure["scopePolicySnapshotSha256"]
        ),
        "thin_query_is_safe_subset_and_deep_query_is_original_exact": bool(
            not forbidden_selected_fields(THIN_RECORD_QUERY)
            and not forbidden_selected_fields(DEEP_NODE_QUERY)
            and not forbidden_selected_fields(RELEASE_QUERY)
            and DEEP_NODE_QUERY == NODE_QUERY
            and selected_field_tokens(THIN_RECORD_QUERY)
            <= selected_field_tokens(RECORD_QUERY) | allowed_shape
            and "mergedBy" not in THIN_RECORD_QUERY
            and "reviews(first" not in THIN_RECORD_QUERY
            and "files(first" not in THIN_RECORD_QUERY
        ),
        "science_gates_and_decision_are_unchanged": bool(
            repair["repair"]["thinQueryIsStrictFieldSubsetOfOriginalSafeQuery"]
            and repair["repair"]["deepNodeQueryIsByteExactOriginalNodeQuery"]
            and repair["repair"]["maximumIdenticalTransientRetries"] == 3
            and all(
                repair["repair"][key] is False
                for key in ("sourceWindowChanged", "exclusionsChanged", "dispositionContractChanged",
                            "samplingChanged", "gatesChanged", "decisionRuleChanged")
            )
        ),
        "repair_authority_keeps_language_models_and_effects_closed": bool(
            failure["modelRunCount"] == 0 and failure["actualExecutionCount"] == 0
            and not repair["decisionRule"]["authorizesTaskLanguageOrModel"]
            and not repair["decisionRule"]["authorizesRegistrationMutationServiceActionOrExecution"]
            and all(path.is_file() for path in paths.values())
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "224r2-two-stage-query-execution-repair-design-audit",
        "experiment": repair["experiment"],
        "passed": passed,
        "decision": "freeze_V224r2_and_authorize_one_completed_two_stage_census" if passed else "reject_V224r2",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, "parent_V224_design_lock": parent_path, "parent_V224r1_repair_lock": r1_path,
                    "scope_policy_snapshot": artifacts["scopePolicySnapshot"], "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "224r2-two-stage-query-execution-repair-lock",
        "experiment": repair["experiment"],
        "config_payload": repair,
        "authorization": {
            "run_one_completed_two_stage_V224_metadata_census": True,
            "read_task_language_open_protected_or_run_model": False,
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

