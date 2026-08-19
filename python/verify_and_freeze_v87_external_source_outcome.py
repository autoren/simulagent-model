#!/usr/bin/env python3
"""Verify source integrity and freeze the text-free V87 inventory outcome."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()  # noqa: S324


def main() -> None:
    impl_path = PROJECT_ROOT / "configs/v87-external-language-source-implementation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v87-external-language-source-audit/inventory/result.json"
    inventory_path = PROJECT_ROOT / "outputs/v87-external-language-source-audit/inventory/structural-inventory.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v87_external_source_outcome.py"
    doc_path = PROJECT_ROOT / "docs/v87-external-language-source-audit-results.md"
    audit_path = PROJECT_ROOT / "outputs/v87-external-language-source-audit/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v87-external-language-source-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V87 source outcome is already frozen")

    impl = json.loads(impl_path.read_text())
    impl_payload = {key: value for key, value in impl.items() if key != "lock_payload_sha256"}
    config = impl["config_payload"]
    result = json.loads(result_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    records = inventory["record_index"]
    source_root = PROJECT_ROOT / "outputs/v87-external-language-source-audit/source"

    source_checks = {}
    for spec in config["selectedSource"]["files"]:
        path = source_root / Path(spec["path"]).name
        data = path.read_bytes()
        registered = result["source_integrity"][spec["path"]]
        source_checks[spec["path"]] = bool(
            len(data) == spec["byteSize"] == registered["byte_size"]
            and git_blob_sha1(data) == spec["gitBlobSha1"] == registered["git_blob_sha1"]
            and file_sha256(path) == registered["local_sha256"]
        )

    forbidden_record_keys = {"utterance", "text", "slot_values", "values", "question", "prompt"}
    record_key_union = set().union(*(set(row) for row in records)) if records else set()
    label_counts = Counter(row["label_kind"] for row in records)
    service_counts = Counter(row["service"] for row in records)
    intent_counts = Counter(f"{row['service']}::{row['active_intent']}" for row in records)
    expected_counts = inventory["counts"]
    summary = result["inventory_summary"]
    checks = {
        "implementation_lock_and_frozen_code_are_exact": bool(
            payload_hash(impl_payload) == impl["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / impl[key]) == impl[f"{key}_sha256"] for key in (
                "module", "test", "runner", "implementation_auditor"
            ))
        ),
        "both_pinned_source_files_match_size_Git_blob_and_local_hash": all(source_checks.values()),
        "inventory_artifact_is_exactly_the_registered_result_dependency": bool(
            result["inventory_artifact"] == str(inventory_path.relative_to(PROJECT_ROOT))
            and file_sha256(inventory_path) == result["inventory_artifact_sha256"]
        ),
        "record_index_has_no_language_or_value_fields": bool(
            not (record_key_union & forbidden_record_keys)
            and not inventory["contains_utterance_or_text_fields"]
            and not summary["contains_utterance_or_text_fields"]
        ),
        "record_index_identity_and_uniqueness_reconstruct": bool(
            canonical_sha256(records) == inventory["record_index_sha256"] == summary["record_index_sha256"]
            and len({row["record_id"] for row in records}) == len(records)
        ),
        "eligible_population_counts_reconstruct": bool(
            len(records) == expected_counts["eligible_record_count"] == 825
            and label_counts == Counter({"active": 757, "none": 68})
            and expected_counts["eligible_active_record_count"] == label_counts["active"]
            and expected_counts["eligible_none_record_count"] == label_counts["none"]
            and expected_counts["user_turn_count"] == 825
            and expected_counts["dialogue_count"] == 128
            and expected_counts["turn_count"] == 1650
        ),
        "service_and_intent_aggregates_reconstruct": bool(
            dict(sorted(service_counts.items())) == inventory["eligible_service_counts"] == summary["eligible_service_counts"]
            and dict(sorted(intent_counts.items())) == inventory["eligible_intent_counts"]
            and len(intent_counts) == summary["eligible_intent_count"] == 7
            and all(not any(row["service"].startswith(prefix) for prefix in config["postLockStructuralInventoryProtocol"]["excludedServicePrefixes"]) for row in records)
        ),
        "registered_result_passes_with_no_model_training_execution_or_manual_inspection": bool(
            result["passed"]
            and result["access"]["manual_utterance_inspection_count"] == 0
            and result["access"]["model_load_count"] == 0
            and result["access"]["model_generation_count"] == 0
            and result["access"]["LLM_API_call_count"] == 0
            and result["access"]["adapter_training_run_count"] == 0
            and result["access"]["real_service_call_count"] == 0
            and result["access"]["external_side_effect_count"] == 0
        ),
        "license_and_provenance_are_preserved": bool(
            inventory["provenance"]["source_name"] == "Schema-Guided Dialogue Dataset"
            and inventory["provenance"]["revision"] == "e852981ae34990f4358979625854259302feaa78"
            and inventory["provenance"]["license"] == "CC-BY-SA-4.0"
            and inventory["provenance"]["adapted_inventory_license"] == "CC-BY-SA-4.0"
            and not inventory["provenance"]["contains_source_utterance_text"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "87-external-language-source-outcome-audit",
        "experiment": "v87_external_language_source_outcome_audit",
        "passed": passed,
        "decision": "freeze_positive_source_and_structural_inventory" if passed else "reject_V87_source_outcome",
        "checks": checks,
        "source_checks": source_checks,
        "independent_population": {
            "record_count": len(records),
            "label_counts": dict(sorted(label_counts.items())),
            "service_counts": dict(sorted(service_counts.items())),
            "intent_count": len(intent_counts),
            "record_index_sha256": canonical_sha256(records),
        },
        "additional_access": {
            "source_JSON_parse_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": result["claim_boundary"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "87-external-language-source-outcome-lock",
        "experiment": "v87_external_language_source_outcome_lock",
        "implementation_lock": str(impl_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(impl_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "source_files": {
            spec["path"]: {
                "local_path": str((source_root / Path(spec["path"]).name).relative_to(PROJECT_ROOT)),
                "local_sha256": file_sha256(source_root / Path(spec["path"]).name),
                "git_blob_sha1": spec["gitBlobSha1"],
            }
            for spec in config["selectedSource"]["files"]
        },
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
            "modify_or_rerun_V87": False,
            "preregister_sealed_nonexecutable_external_language_shadow_subset": True,
            "select_subset_before_utterance_extraction_by_frozen_hash_rule": True,
            "manually_inspect_utterances_before_subset_lock": False,
            "access_local_or_API_model_before_subset_and_prompt_lock": False,
            "train_adapter": False,
            "grant_language_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
