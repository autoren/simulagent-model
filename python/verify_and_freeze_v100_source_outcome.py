#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import (
    build_massive_source_inventory,
    evaluate_massive_source_gates,
    parse_massive_archive,
    sha256_bytes,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    source_lock_path = PROJECT_ROOT / "configs/v100-massive-source-lock.json"
    result_path = PROJECT_ROOT / "outputs/v100-massive-open-set/source-inventory/result.json"
    inventory_path = PROJECT_ROOT / "outputs/v100-massive-open-set/source-inventory/massive-open-set-inventory.json"
    doc_path = PROJECT_ROOT / "docs/v100-massive-source-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v100_source_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v100-massive-open-set/source-outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v100-massive-source-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V100 source outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V100 source result before freezing")

    lock = json.loads(source_lock_path.read_text())
    result = json.loads(result_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    config = lock["config_payload"]
    source_path = PROJECT_ROOT / result["source_integrity"]["local_path"]
    source_bytes = source_path.read_bytes()
    records, locale_member = parse_massive_archive(
        source_bytes, config["archive"]["expectedLocaleMemberSuffix"]
    )
    reconstructed = build_massive_source_inventory(records, config)
    reconstructed_gates = evaluate_massive_source_gates(reconstructed, config)
    reconstructed_gates["zero_manual_model_API_training_service_or_side_effect_access"] = True
    candidate_rows = inventory["candidate_index"]
    class_counts = Counter(row["class_label"] for row in candidate_rows)
    class_scenarios: dict[str, set[str]] = defaultdict(set)
    class_partitions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in candidate_rows:
        class_scenarios[row["class_label"]].add(row["scenario"])
        class_partitions[row["class_label"]][row["partition"]] += 1

    dependency_keys = (
        "config", "parent_source_selection_lock", "plan", "protocol", "tests",
        "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "source_lock_and_dependencies_are_exact": bool(
            payload_hash(
                {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
            ) == lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "archive_identity_is_exact": bool(
            len(source_bytes) == config["archive"]["byteSize"]
            and result["source_integrity"]["etag"] == config["archive"]["etag"]
            and result["source_integrity"]["last_modified"] == config["archive"]["lastModified"]
            and result["source_integrity"]["archive_sha256"] == sha256_bytes(source_bytes)
            and file_sha256(source_path) == result["source_integrity"]["local_sha256"]
            and locale_member == result["source_integrity"]["locale_member"]
        ),
        "inventory_reconstructs_exactly": (
            reconstructed == {key: value for key, value in inventory.items() if key != "provenance"}
        ),
        "class_counts_coverage_and_partitions_reconstruct": bool(
            dict(sorted(class_counts.items())) == inventory["class_counts"]
            and {
                label: len(scenarios) for label, scenarios in sorted(class_scenarios.items())
            } == inventory["class_scenario_counts"]
            and {
                label: dict(sorted(counts.items()))
                for label, counts in sorted(class_partitions.items())
            } == inventory["class_partition_counts"]
        ),
        "scenario_and_intent_roles_are_disjoint": bool(
            not (set(inventory["catalog_scenarios"]) & set(inventory["unsupported_scenarios"]))
            and not (set(inventory["hidden_intents"]) & set(inventory["declared_intents"]))
            and set(inventory["hidden_scenarios"]) <= set(inventory["catalog_scenarios"])
        ),
        "result_gates_and_decision_are_consistent": bool(
            reconstructed_gates == result["gates"]
            and result["passed"] == all(result["gates"].values())
            and result["decision"] == (
                "freeze_MASSIVE_source_and_preregister_open_set_population"
                if result["passed"] else "stop_V100_before_population_language_or_model_access"
            )
        ),
        "text_free_and_zero_access_boundary_holds": bool(
            not inventory["contains_raw_or_annotated_utterances_tokens_or_slot_values"]
            and not inventory["provenance"]["contains_source_language"]
            and not inventory["provenance"]["contains_derived_language_tokens_or_slot_values"]
            and all(
                result["access"][key] == 0
                for key in (
                    "emitted_language_record_count", "manual_utterance_inspection_count",
                    "model_load_count", "model_generation_count", "LLM_API_call_count",
                    "adapter_training_run_count", "real_service_call_count",
                    "external_side_effect_count",
                )
            )
        ),
    }
    integrity_passed = all(checks.values())
    failed_gates = sorted(key for key, value in result["gates"].items() if not value)
    audit = {
        "schema_version": "100-massive-source-outcome-audit",
        "experiment": "v100_massive_source_outcome_audit",
        "passed": integrity_passed,
        "scientific_source_feasibility_passed": result["passed"],
        "decision": (
            "freeze_positive_V100_MASSIVE_source_feasibility"
            if result["passed"] else "freeze_negative_V100_MASSIVE_source_feasibility"
        ) if integrity_passed else "reject_V100_source_outcome",
        "checks": checks,
        "independent_summary": {
            "source_record_count": len(records),
            "candidate_count": len(candidate_rows),
            "class_counts": dict(sorted(class_counts.items())),
            "class_scenario_counts": {
                label: len(scenarios) for label, scenarios in sorted(class_scenarios.items())
            },
            "class_partition_counts": {
                label: dict(sorted(counts.items()))
                for label, counts in sorted(class_partitions.items())
            },
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
        "source_archive": source_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "100-massive-source-outcome-lock",
        "experiment": "v100_massive_source_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_source_feasibility_passed": result["passed"],
            "decision": audit["decision"],
            "inventory_summary": result["inventory_summary"],
        },
        "authorization": {
            "modify_or_rerun_V100_source_stage": False,
            "preregister_hash_selected_validation_and_test_population": result["passed"],
            "select_population_before_language_extraction": result["passed"],
            "extract_selected_language_before_population_lock": False,
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
    outcome["source_archive_payload_sha256"] = sha256_bytes(source_bytes)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({
        "lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(outcome_path),
    }, indent=2))


if __name__ == "__main__":
    main()
