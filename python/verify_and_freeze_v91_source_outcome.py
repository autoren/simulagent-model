#!/usr/bin/env python3
"""Independently verify and freeze the V91 source-extension outcome."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v87_external_source_inventory import git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    source_lock_path = PROJECT_ROOT / "configs/v91-rank-only-source-lock.json"
    result_path = PROJECT_ROOT / "outputs/v91-rank-only/source-inventory/result.json"
    inventory_path = (
        PROJECT_ROOT / "outputs/v91-rank-only/source-inventory/structural-inventory.json"
    )
    doc_path = PROJECT_ROOT / "docs/v91-rank-only-source-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v91_source_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v91-rank-only/source-outcome-audit.json"
    outcome_lock_path = PROJECT_ROOT / "configs/v91-rank-only-source-outcome-lock.json"
    if audit_path.exists() or outcome_lock_path.exists():
        raise RuntimeError("V91 source outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V91 source-only result before freezing the outcome")

    source_lock = json.loads(source_lock_path.read_text())
    source_lock_payload = {
        key: value
        for key, value in source_lock.items()
        if key != "lock_payload_sha256"
    }
    config = source_lock["config_payload"]
    result = json.loads(result_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    source_path = PROJECT_ROOT / result["source_integrity"]["local_path"]
    source_bytes = source_path.read_bytes()
    records = inventory["record_index"]
    counts = Counter(row["label_kind"] for row in records)
    service_counts = Counter(row["service"] for row in records)
    intent_counts = Counter(
        f"{row['service']}::{row['active_intent']}" for row in records
    )
    forbidden = {
        "utterance",
        "text",
        "slot_values",
        "values",
        "question",
        "prompt",
        "dialogue_history",
    }
    keys = set().union(*(set(row) for row in records)) if records else set()
    dependency_keys = (
        "config",
        "source_authority_lock",
        "parent_model_decision_lock",
        "previous_source_outcome_lock",
        "plan",
        "inventory_module",
        "runner",
        "verifier",
        "auditor",
    )
    checks = {
        "source_lock_and_all_frozen_dependencies_are_exact": bool(
            payload_hash(source_lock_payload) == source_lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / source_lock[key])
                == source_lock[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "download_matches_registered_size_blob_and_local_hash": bool(
            len(source_bytes)
            == config["newDialogueShard"]["byteSize"]
            == result["source_integrity"]["byte_size"]
            and git_blob_sha1(source_bytes)
            == config["newDialogueShard"]["gitBlobSha1"]
            == result["source_integrity"]["git_blob_sha1"]
            and file_sha256(source_path) == result["source_integrity"]["local_sha256"]
        ),
        "inventory_identity_uniqueness_and_text_exclusion_reconstruct": bool(
            canonical_sha256(records) == inventory["record_index_sha256"]
            and len(records) == len({row["record_id"] for row in records})
            and not (keys & forbidden)
            and not inventory["contains_utterance_or_text_fields"]
            and file_sha256(inventory_path) == result["inventory_sha256"]
        ),
        "structural_aggregates_reconstruct": bool(
            len(records) == inventory["counts"]["eligible_record_count"]
            and counts["active"]
            == inventory["counts"]["eligible_active_record_count"]
            and counts["none"] == inventory["counts"]["eligible_none_record_count"]
            and dict(sorted(service_counts.items()))
            == inventory["eligible_service_counts"]
            and dict(sorted(intent_counts.items()))
            == inventory["eligible_intent_counts"]
        ),
        "all_registered_source_gates_pass": bool(
            result["passed"] and all(result["gates"].values())
        ),
        "zero_manual_model_API_training_service_or_side_effect_access": all(
            result["access"][key] == 0
            for key in (
                "manual_utterance_inspection_count",
                "new_model_weight_download_count",
                "model_load_count",
                "model_generation_count",
                "LLM_API_call_count",
                "adapter_training_run_count",
                "real_service_call_count",
                "external_side_effect_count",
            )
        ),
        "license_and_source_provenance_are_preserved": bool(
            inventory["provenance"]["revision"] == config["revision"]
            and inventory["provenance"]["source_shard"]
            == config["newDialogueShard"]["path"]
            and inventory["provenance"]["license"] == "CC-BY-SA-4.0"
            and not inventory["provenance"]["contains_source_utterance_text"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "91-rank-only-source-outcome-audit",
        "experiment": "v91_rank_only_source_outcome_audit",
        "passed": passed,
        "decision": (
            "freeze_positive_fresh_rank_only_source_inventory"
            if passed
            else "reject_V91_source_outcome"
        ),
        "checks": checks,
        "independent_population": {
            "record_count": len(records),
            "label_counts": dict(sorted(counts.items())),
            "service_counts": dict(sorted(service_counts.items())),
            "intent_counts": dict(sorted(intent_counts.items())),
            "record_index_sha256": canonical_sha256(records),
        },
        "additional_access": {
            "manual_utterance_inspection_count": 0,
            "new_model_weight_download_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "91-rank-only-source-outcome-lock",
        "experiment": "v91_rank_only_source_outcome_lock",
        "source_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
        "source_lock_sha256": file_sha256(source_lock_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "source_file": str(source_path.relative_to(PROJECT_ROOT)),
        "source_file_sha256": file_sha256(source_path),
        "source_file_git_blob_sha1": git_blob_sha1(source_bytes),
        "verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "verifier_sha256": file_sha256(verifier_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "results_document": str(doc_path.relative_to(PROJECT_ROOT)),
        "results_document_sha256": file_sha256(doc_path),
        "outcome": {
            "passed": True,
            "decision": result["decision"],
            "inventory_summary": result["inventory_summary"],
        },
        "authorization": {
            "modify_or_rerun_source_extension": False,
            "preregister_fresh_hash_selected_rank_only_shadow_population": True,
            "select_population_before_utterance_extraction": True,
            "manually_inspect_source_language": False,
            "load_model_before_population_prompt_controls_gates_and_invariance_lock": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_pruning_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    outcome_lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(outcome_lock_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(outcome_lock_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
