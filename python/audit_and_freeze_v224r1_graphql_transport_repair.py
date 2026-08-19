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
from v224_graphql_queries import NODE_QUERY, RECORD_QUERY, RELEASE_QUERY, forbidden_selected_fields


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v224r1-graphql-transport-repair.json",
        "plan": PROJECT_ROOT / "docs/v224r1-graphql-transport-repair-plan.md",
        "failure_record": PROJECT_ROOT / "docs/v224-initial-capture-failure.md",
        "protocol": PROJECT_ROOT / "python/v224r1_graphql_transport_repair.py",
        "tests": PROJECT_ROOT / "python/test_v224r1_graphql_transport_repair.py",
        "capture": PROJECT_ROOT / "python/capture_v224r1_graphql_transport_repair.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v224r1_graphql_transport_repair.py",
    }
    output_root = PROJECT_ROOT / "outputs/v224r1-graphql-transport-repair"
    audit_path = output_root / "design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v224r1-graphql-transport-repair-lock.json"
    if output_root.exists() or lock_path.exists():
        raise RuntimeError("V224r1 is already audited or locked")
    repair = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / repair["parentV224DesignLock"]
    parent = json.loads(parent_path.read_text())
    failure = repair["failureBoundary"]
    scope_path = PROJECT_ROOT / failure["scopePolicySnapshot"]
    parent_artifacts = {
        key: PROJECT_ROOT / value for key, value in parent["config_payload"]["artifacts"].items()
    }
    checks = {
        "parent_V224_lock_is_valid_exact_and_science_stays_immutable": bool(
            valid_lock(parent)
            and dependency_hashes_exact(parent)
            and file_sha256(parent_path) == repair["parentV224DesignLockSha256"]
        ),
        "failure_preceded_every_record_response_and_scientific_artifact": bool(
            failure["exceptionType"] == "CalledProcessError"
            and failure["successfulRecordGraphQLResponseCount"] == 0
            and failure["formalRecordMetadataReadCount"] == 0
            and failure["completedSearchSliceCount"] == 0
            and failure["otherFormalArtifactCount"] == 0
            and not any(path.exists() for key, path in parent_artifacts.items() if key != "scopePolicySnapshot")
        ),
        "scope_policy_is_exact_and_only_existing_capture_artifact": bool(
            scope_path.is_file()
            and file_sha256(scope_path) == failure["scopePolicySnapshotSha256"]
            and failure["scopePolicyRetrievalCount"] == 1
        ),
        "repair_changes_transport_only_and_query_firewall_remains_exact": bool(
            repair["repair"]["operation"]
            == "send_exact_locked_GraphQL_document_and_variables_as_separate_JSON_members_to_official_endpoint"
            and not any(forbidden_selected_fields(query) for query in (RECORD_QUERY, NODE_QUERY, RELEASE_QUERY))
            and all(
                repair["repair"][key] is False
                for key in (
                    "GraphQLDocumentChanged", "GraphQLSelectedFieldsChanged", "sourceWindowChanged",
                    "exclusionsChanged", "dispositionContractChanged", "samplingChanged", "gatesChanged",
                    "decisionRuleChanged",
                )
            )
        ),
        "repair_authorizes_one_capture_only_with_language_model_and_effects_closed": bool(
            failure["taskRecordTitleOrBodyReadCount"] == 0
            and failure["commentOrReviewTextReadCount"] == 0
            and failure["modelRunCount"] == 0
            and failure["actualExecutionCount"] == 0
            and not repair["decisionRule"]["authorizesTaskLanguageOrModel"]
            and not repair["decisionRule"]["authorizesRegistrationMutationServiceActionOrExecution"]
            and all(path.is_file() for path in paths.values())
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "224r1-graphql-transport-repair-design-audit",
        "experiment": repair["experiment"],
        "passed": passed,
        "decision": "freeze_V224r1_and_authorize_one_repaired_capture" if passed else "reject_V224r1",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, "parent_V224_design_lock": parent_path, "scope_policy_snapshot": scope_path,
                    "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "224r1-graphql-transport-repair-lock",
        "experiment": repair["experiment"],
        "config_payload": repair,
        "authorization": {
            "run_one_repaired_V224_metadata_capture": True,
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

