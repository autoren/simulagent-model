#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v97_aggregate_open_set_source import build_aggregate_open_set_inventory, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    source_lock_path = PROJECT_ROOT / "configs/v97-aggregate-open-set-source-lock.json"
    result_path = PROJECT_ROOT / "outputs/v97-aggregate-open-set/source-inventory/result.json"
    inventory_path = PROJECT_ROOT / "outputs/v97-aggregate-open-set/source-inventory/aggregate-open-set-inventory.json"
    doc_path = PROJECT_ROOT / "docs/v97-aggregate-open-set-source-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v97_source_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v97-aggregate-open-set/source-outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v97-aggregate-open-set-source-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V97 source outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V97 source result before freezing")
    lock = json.loads(source_lock_path.read_text())
    result = json.loads(result_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    config = lock["config_payload"]
    source_entries = []
    identities_exact = True
    for shard, integrity in zip(config["aggregateDialogueShards"], result["source_integrity"], strict=True):
        source_path = PROJECT_ROOT / integrity["local_path"]
        source_bytes = source_path.read_bytes()
        identities_exact = identities_exact and bool(
            integrity["path"] == shard["path"]
            and len(source_bytes) == shard["byteSize"]
            and git_blob_sha1(source_bytes) == shard["gitBlobSha1"]
            and file_sha256(source_path) == integrity["local_sha256"]
        )
        source_entries.append((shard["path"], json.loads(source_bytes)))
    schema_path = PROJECT_ROOT / config["schemaDependency"]["localPath"]
    reconstructed = build_aggregate_open_set_inventory(
        json.loads(schema_path.read_text()), source_entries, config
    )
    records = inventory["candidate_index"]
    class_counts = Counter(row["class_label"] for row in records)
    class_services: dict[str, set[str]] = defaultdict(set)
    for row in records:
        class_services[row["class_label"]].add(row["source_service"])
    dependency_keys = (
        "config", "parent_source_outcome", "parent_architecture_lock", "source_authority_lock",
        "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "source_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "all_source_identities_are_exact": identities_exact,
        "inventory_reconstructs_exactly": reconstructed == {key: value for key, value in inventory.items() if key != "provenance"},
        "class_counts_and_service_coverage_reconstruct": bool(
            dict(sorted(class_counts.items())) == inventory["class_counts"]
            and {label: len(services) for label, services in sorted(class_services.items())} == inventory["class_service_counts"]
        ),
        "service_roles_and_pair_sets_are_disjoint": bool(
            not (set(inventory["catalog_services"]) & set(inventory["unsupported_services"]))
            and not (set(inventory["hidden_pairs"]) & set(inventory["declared_supported_pairs"]))
            and len(inventory["hidden_services"]) == len(inventory["hidden_pairs"])
            and all(pair.split("::", 1)[0] in inventory["hidden_services"] for pair in inventory["hidden_pairs"])
        ),
        "activation_boundary_is_exact": bool(
            inventory["all_non_none_candidates_are_source_intent_activations"]
            and all(
                row["source_intent_activation"]
                for row in records if row["class_label"] != "insufficient_evidence"
            )
        ),
        "result_and_gate_decision_are_consistent": bool(
            result["passed"] == all(result["gates"].values())
            and result["decision"] == (
                "freeze_source_and_preregister_aggregate_open_set_population"
                if result["passed"] else "stop_V97_before_population_or_model_access"
            )
        ),
        "text_free_and_zero_access_boundary_holds": bool(
            not inventory["contains_language_tokens_slot_values_or_histories"]
            and not inventory["provenance"]["contains_source_language"]
            and all(result["access"][key] == 0 for key in (
                "emitted_language_record_count", "manual_utterance_inspection_count",
                "model_load_count", "model_generation_count", "LLM_API_call_count",
                "adapter_training_run_count", "real_service_call_count", "external_side_effect_count",
            ))
        ),
    }
    integrity_passed = all(checks.values())
    failed_gates = sorted(key for key, value in result["gates"].items() if not value)
    audit = {
        "schema_version": "97-aggregate-open-set-source-outcome-audit",
        "experiment": "v97_aggregate_open_set_source_outcome_audit",
        "passed": integrity_passed,
        "scientific_source_feasibility_passed": result["passed"],
        "decision": (
            "freeze_positive_V97_aggregate_source_feasibility" if result["passed"]
            else "freeze_negative_V97_aggregate_source_feasibility"
        ) if integrity_passed else "reject_V97_source_outcome",
        "checks": checks,
        "independent_summary": {
            "candidate_count": len(records),
            "class_counts": dict(sorted(class_counts.items())),
            "class_service_counts": {label: len(services) for label, services in sorted(class_services.items())},
            "failed_gates": failed_gates,
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
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "source_lock": source_lock_path,
        "result": result_path,
        "inventory": inventory_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    for index, integrity in enumerate(result["source_integrity"], start=1):
        dependencies[f"source_file_{index:02d}"] = PROJECT_ROOT / integrity["local_path"]
    outcome: dict[str, Any] = {
        "schema_version": "97-aggregate-open-set-source-outcome-lock",
        "experiment": "v97_aggregate_open_set_source_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_source_feasibility_passed": result["passed"],
            "decision": audit["decision"],
            "inventory_summary": result["inventory_summary"],
        },
        "authorization": {
            "modify_or_rerun_V97_source_stage": False,
            "preregister_dialogue_disjoint_population": result["passed"],
            "select_population_before_language_extraction": result["passed"],
            "manually_inspect_source_language": False,
            "load_model_before_population_prompt_controls_metrics_and_gates_lock": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["source_file_git_blob_sha1s"] = [entry["git_blob_sha1"] for entry in result["source_integrity"]]
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
