#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v93_open_set_source import build_open_set_inventory, canonical_sha256, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    source_lock_path = PROJECT_ROOT / "configs/v93-controlled-open-set-source-lock.json"
    result_path = PROJECT_ROOT / "outputs/v93-controlled-open-set/source-inventory/result.json"
    inventory_path = PROJECT_ROOT / "outputs/v93-controlled-open-set/source-inventory/open-set-inventory.json"
    doc_path = PROJECT_ROOT / "docs/v93-controlled-open-set-source-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v93_source_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v93-controlled-open-set/source-outcome-audit.json"
    outcome_lock_path = PROJECT_ROOT / "configs/v93-controlled-open-set-source-outcome-lock.json"
    if audit_path.exists() or outcome_lock_path.exists():
        raise RuntimeError("V93 source outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the source-only results document before freezing V93")

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
    records = inventory["candidate_index"]
    class_counts = Counter(row["class_label"] for row in records)
    class_services: dict[str, set[str]] = defaultdict(set)
    for row in records:
        class_services[row["class_label"]].add(row["target_service"])
    forbidden_keys = {"utterance", "text", "tokens", "history", "slot_values", "values", "prompt"}
    emitted_keys = set().union(*(row.keys() for row in records)) if records else set()
    dependency_keys = (
        "config", "parent_architecture_lock", "source_authority_lock", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "source_lock_and_all_dependencies_are_exact": bool(
            payload_hash(lock_payload) == source_lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / source_lock[key]) == source_lock[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "download_matches_registered_size_blob_and_hash": bool(
            len(source_bytes) == config["newDialogueShard"]["byteSize"] == result["source_integrity"]["byte_size"]
            and git_blob_sha1(source_bytes) == config["newDialogueShard"]["gitBlobSha1"] == result["source_integrity"]["git_blob_sha1"]
            and file_sha256(source_path) == result["source_integrity"]["local_sha256"]
        ),
        "independent_inventory_reconstruction_is_exact": bool(
            reconstructed == {key: value for key, value in inventory.items() if key != "provenance"}
            and file_sha256(inventory_path) == result["inventory_sha256"]
        ),
        "candidate_identity_counts_and_coverage_reconstruct": bool(
            len(records) == len({row["candidate_id"] for row in records}) == inventory["candidate_count"]
            and canonical_sha256(records) == inventory["candidate_index_sha256"]
            and dict(sorted(class_counts.items())) == inventory["class_counts"]
            and {label: len(services) for label, services in sorted(class_services.items())} == inventory["class_service_counts"]
        ),
        "five_classes_are_distinct_and_present": set(class_counts) == {
            "known_familiar", "known_unfamiliar", "novel_valid", "unsupported", "insufficient_evidence"
        },
        "hidden_and_declared_intents_are_disjoint": all(
            split["hidden_intent"] not in split["declared_intents"]
            and set(split["complete_intents"]) == set(split["declared_intents"]) | {split["hidden_intent"]}
            for split in inventory["service_splits"].values()
        ),
        "unsupported_rows_cross_services_and_target_lacks_source_intent": all(
            row["source_service"] != row["target_service"]
            and row["gold_source_intent"] not in inventory["service_splits"][row["target_service"]]["complete_intents"]
            for row in records if row["class_label"] == "unsupported"
        ),
        "NONE_maps_only_to_insufficient_evidence": all(
            (row["gold_source_intent"] == "NONE") == (row["class_label"] == "insufficient_evidence")
            for row in records if row["class_label"] != "unsupported"
        ),
        "inventory_emits_no_language_tokens_values_or_histories": bool(
            not (emitted_keys & forbidden_keys)
            and not inventory["contains_language_tokens_slot_values_or_histories"]
            and not inventory["provenance"]["contains_source_language"]
            and not inventory["provenance"]["contains_derived_language_tokens"]
        ),
        "all_registered_source_gates_pass": bool(result["passed"] and all(result["gates"].values())),
        "zero_manual_model_API_training_service_or_side_effect_access": all(
            result["access"][key] == 0
            for key in (
                "emitted_language_record_count", "manual_utterance_inspection_count", "model_load_count",
                "model_generation_count", "LLM_API_call_count", "adapter_training_run_count",
                "real_service_call_count", "external_side_effect_count",
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "93-controlled-open-set-source-outcome-audit",
        "experiment": "v93_controlled_open_set_source_outcome_audit",
        "passed": passed,
        "decision": "freeze_positive_controlled_open_set_source_feasibility" if passed else "reject_V93_source_outcome",
        "checks": checks,
        "independent_summary": {
            "candidate_count": len(records),
            "class_counts": dict(sorted(class_counts.items())),
            "class_service_counts": {label: len(services) for label, services in sorted(class_services.items())},
            "candidate_index_sha256": canonical_sha256(records),
        },
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

    lock = {
        "schema_version": "93-controlled-open-set-source-outcome-lock",
        "experiment": "v93_controlled_open_set_source_outcome_lock",
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
            "modify_or_rerun_source_stage": False,
            "preregister_hash_selected_calibration_and_evaluation_population": True,
            "select_population_before_language_extraction": True,
            "manually_inspect_source_language": False,
            "load_model_before_population_prompt_controls_metrics_and_gates_lock": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    outcome_lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_lock_path)}, indent=2))


if __name__ == "__main__":
    main()
