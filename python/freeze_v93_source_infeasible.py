#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v93_open_set_source import build_open_set_inventory, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    source_lock_path = PROJECT_ROOT / "configs/v93-controlled-open-set-source-lock.json"
    result_path = PROJECT_ROOT / "outputs/v93-controlled-open-set/source-inventory/result.json"
    inventory_path = PROJECT_ROOT / "outputs/v93-controlled-open-set/source-inventory/open-set-inventory.json"
    doc_path = PROJECT_ROOT / "docs/v93-controlled-open-set-source-results.md"
    auditor_path = PROJECT_ROOT / "python/freeze_v93_source_infeasible.py"
    audit_path = PROJECT_ROOT / "outputs/v93-controlled-open-set/source-closure-audit.json"
    closure_path = PROJECT_ROOT / "configs/v93-controlled-open-set-source-closure-lock.json"
    if audit_path.exists() or closure_path.exists():
        raise RuntimeError("V93 negative source result is already closed")

    source_lock = json.loads(source_lock_path.read_text())
    lock_payload = {key: value for key, value in source_lock.items() if key != "lock_payload_sha256"}
    result = json.loads(result_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    config = source_lock["config_payload"]
    source_path = PROJECT_ROOT / result["source_integrity"]["local_path"]
    source_bytes = source_path.read_bytes()
    schema_path = PROJECT_ROOT / config["schemaDependency"]["localPath"]
    reconstructed = build_open_set_inventory(
        json.loads(schema_path.read_text()), json.loads(source_bytes), config
    )
    dependency_keys = (
        "config", "parent_architecture_lock", "source_authority_lock", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    failed_scientific_gates = sorted(
        key for key, value in result["gates"].items()
        if not value and key not in {"text_free_inventory", "zero_manual_model_API_training_service_or_side_effect_access"}
    )
    checks = {
        "source_lock_and_original_dependencies_are_exact": bool(
            payload_hash(lock_payload) == source_lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / source_lock[key]) == source_lock[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "source_bytes_match_registered_identity": bool(
            len(source_bytes) == config["newDialogueShard"]["byteSize"]
            and git_blob_sha1(source_bytes) == config["newDialogueShard"]["gitBlobSha1"]
            and file_sha256(source_path) == result["source_integrity"]["local_sha256"]
        ),
        "frozen_inventory_reconstructs_exactly": reconstructed == {
            key: value for key, value in inventory.items() if key != "provenance"
        },
        "registered_stop_decision_and_zero_population_reconstruct": bool(
            not result["passed"]
            and result["decision"] == "stop_V93_before_population_or_model_access"
            and inventory["source_record_count"] == 154
            and inventory["eligible_service_count"] == 0
            and inventory["candidate_count"] == 0
            and inventory["class_counts"] == {}
        ),
        "all_scientific_source_gates_failed_without_relaxation": len(failed_scientific_gates) == 11,
        "text_free_and_zero_access_gates_passed": bool(
            result["gates"]["text_free_inventory"]
            and result["gates"]["zero_manual_model_API_training_service_or_side_effect_access"]
            and all(
                result["access"][key] == 0
                for key in (
                    "emitted_language_record_count", "manual_utterance_inspection_count", "model_load_count",
                    "model_generation_count", "LLM_API_call_count", "adapter_training_run_count",
                    "real_service_call_count", "external_side_effect_count",
                )
            )
        ),
        "negative_result_document_exists": doc_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "93-controlled-open-set-source-closure-audit",
        "experiment": "v93_controlled_open_set_source_infeasibility_closure",
        "passed": passed,
        "decision": "freeze_V93_source_infeasible_without_population_or_model_access" if passed else "reject_V93_closure",
        "checks": checks,
        "failed_scientific_gates": failed_scientific_gates,
        "additional_access": {
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    closure = {
        "schema_version": "93-controlled-open-set-source-closure-lock",
        "experiment": "v93_controlled_open_set_source_infeasibility_closure_lock",
        "source_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
        "source_lock_sha256": file_sha256(source_lock_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "source_file": str(source_path.relative_to(PROJECT_ROOT)),
        "source_file_sha256": file_sha256(source_path),
        "source_file_git_blob_sha1": git_blob_sha1(source_bytes),
        "closure_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "closure_auditor_sha256": file_sha256(auditor_path),
        "closure_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "closure_audit_sha256": file_sha256(audit_path),
        "results_document": str(doc_path.relative_to(PROJECT_ROOT)),
        "results_document_sha256": file_sha256(doc_path),
        "outcome": {
            "passed": True,
            "scientific_source_feasibility_passed": False,
            "decision": audit["decision"],
            "source_record_count": 154,
            "eligible_service_count": 0,
            "candidate_count": 0,
        },
        "authorization": {
            "modify_relax_or_rerun_V93": False,
            "select_V93_population_or_extract_language": False,
            "load_model_for_V93": False,
            "claim_novelty_or_abstention_evidence": False,
            "preregister_fresh_global_capability_catalog_successor": True,
            "reuse_V93_source_for_successor_outcomes": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    closure["lock_payload_sha256"] = payload_hash(closure)
    closure_path.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(closure_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(closure_path)}, indent=2))


if __name__ == "__main__":
    main()
