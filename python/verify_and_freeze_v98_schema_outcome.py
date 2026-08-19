#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v98_test_schema_feasibility import build_test_schema_inventory, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v98-test-schema-feasibility-lock.json"
    result_path = PROJECT_ROOT / "outputs/v98-test-schema-feasibility/schema-inventory/result.json"
    inventory_path = PROJECT_ROOT / "outputs/v98-test-schema-feasibility/schema-inventory/test-schema-inventory.json"
    doc_path = PROJECT_ROOT / "docs/v98-test-schema-feasibility-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v98_schema_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v98-test-schema-feasibility/schema-outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v98-test-schema-feasibility-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V98 schema outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V98 schema result before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    config = lock["config_payload"]
    source_path = PROJECT_ROOT / result["source_integrity"]["local_path"]
    source_bytes = source_path.read_bytes()
    dev_path = PROJECT_ROOT / config["developmentSchemaDependency"]["localPath"]
    reconstructed = build_test_schema_inventory(
        json.loads(dev_path.read_text()), json.loads(source_bytes), config
    )
    dependency_keys = (
        "config", "parent_source_outcome", "source_authority_lock", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "schema_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "test_schema_identity_is_exact": bool(
            len(source_bytes) == config["testSchema"]["byteSize"]
            and git_blob_sha1(source_bytes) == config["testSchema"]["gitBlobSha1"]
            and file_sha256(source_path) == result["source_integrity"]["local_sha256"]
        ),
        "inventory_reconstructs_exactly": reconstructed == {key: value for key, value in inventory.items() if key != "provenance"},
        "result_and_gate_decision_are_consistent": bool(
            result["passed"] == all(result["gates"].values())
            and result["decision"] == (
                "freeze_schema_and_preregister_test_dialogue_source_pool"
                if result["passed"] else "stop_V98_before_test_dialogue_or_model_access"
            )
        ),
        "language_free_and_zero_dialogue_or_model_access_boundary_holds": bool(
            not inventory["contains_schema_language_or_surface_tokens"]
            and not inventory["provenance"]["contains_schema_language"]
            and all(result["access"][key] == 0 for key in (
                "test_dialogue_payload_access_count", "emitted_schema_language_record_count",
                "manual_schema_language_inspection_count", "model_load_count", "model_generation_count",
                "LLM_API_call_count", "adapter_training_run_count", "real_service_call_count",
                "external_side_effect_count",
            ))
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "98-test-schema-feasibility-outcome-audit",
        "experiment": "v98_test_schema_feasibility_outcome_audit",
        "passed": integrity_passed,
        "scientific_schema_feasibility_passed": result["passed"],
        "decision": (
            "freeze_positive_V98_test_schema_feasibility" if result["passed"]
            else "freeze_negative_V98_test_schema_feasibility"
        ) if integrity_passed else "reject_V98_schema_outcome",
        "checks": checks,
        "independent_summary": {
            "novel_service_families": reconstructed["novel_service_families"],
            "novel_service_family_count": reconstructed["novel_service_family_count"],
            "eligible_novel_service_count": reconstructed["eligible_novel_service_count"],
            "failed_gates": sorted(key for key, value in result["gates"].items() if not value),
        },
        "additional_access": {
            "test_dialogue_payload_access_count": 0,
            "manual_schema_language_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "schema_lock": lock_path, "result": result_path, "inventory": inventory_path,
        "source_schema": source_path, "verifier": verifier_path, "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "98-test-schema-feasibility-outcome-lock",
        "experiment": "v98_test_schema_feasibility_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_schema_feasibility_passed": result["passed"],
            "decision": audit["decision"],
            "inventory_summary": result["inventory_summary"],
        },
        "authorization": {
            "modify_or_rerun_V98_schema_stage": False,
            "preregister_test_dialogue_source_pool_metadata": result["passed"],
            "download_test_dialogue_payload_before_source_pool_lock": False,
            "select_population_or_extract_language": False,
            "manually_inspect_schema_or_dialogue_language": False,
            "load_local_or_API_model": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["source_schema_git_blob_sha1"] = git_blob_sha1(source_bytes)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
