#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v102r1_presto_context_source import (
    build_repaired_presto_context_inventory,
    evaluate_repaired_presto_source_gates,
    parse_presto_archive,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def main() -> None:
    source_lock_path = PROJECT_ROOT / "configs/v102r1-presto-context-source-lock.json"
    result_path = PROJECT_ROOT / "outputs/v102r1-presto-context-source/source-inventory/result.json"
    inventory_path = PROJECT_ROOT / "outputs/v102r1-presto-context-source/source-inventory/presto-context-dependency-inventory.json"
    doc_path = PROJECT_ROOT / "docs/v102r1-presto-context-source-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v102r1_presto_source_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v102r1-presto-context-source/source-outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v102r1-presto-context-source-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V102r1 source outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V102r1 source result before freezing")
    lock = json.loads(source_lock_path.read_text())
    result = json.loads(result_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    config = lock["scientific_config_payload"]
    source_path = PROJECT_ROOT / result["source_integrity"]["local_path"]
    source_bytes = source_path.read_bytes()
    source_records, members = parse_presto_archive(
        source_bytes, config["archive"]["requiredMemberBasenames"]
    )
    reconstructed = build_repaired_presto_context_inventory(source_records, config)
    reconstructed_gates = evaluate_repaired_presto_source_gates(reconstructed, config)
    reconstructed_gates["repair_encountered_non_string_optional_context_leaves"] = (
        reconstructed["ignored_non_string_optional_context_leaf_count"] > 0
    )
    reconstructed_gates["zero_manual_model_API_training_service_or_side_effect_access"] = True
    rows = inventory["candidate_index"]
    role_counts = Counter(row["role"] for row in rows)
    dependency_counts = Counter(
        kind for row in rows for kind in row["dependency_source_kinds"]
    )
    dependency_keys = (
        "repair_config", "parent_technical_outcome", "scientific_config", "plan",
        "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
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
        "archive_identity_and_persisted_bytes_are_exact": bool(
            len(source_bytes) == config["archive"]["byteSize"]
            and md5_hex(source_bytes) == config["archive"]["etagMd5"]
            and result["source_integrity"]["etag_md5"] == config["archive"]["etagMd5"]
            and result["source_integrity"]["generation"] == config["archive"]["generation"]
            and result["source_integrity"]["last_modified"] == config["archive"]["lastModified"]
            and file_sha256(source_path) == result["source_integrity"]["archive_sha256"]
            and members == result["source_integrity"]["parsed_members"]
        ),
        "inventory_reconstructs_exactly": (
            reconstructed == {key: value for key, value in inventory.items() if key != "provenance"}
        ),
        "candidate_counts_and_dependency_sources_reconstruct": bool(
            dict(sorted(role_counts.items())) == inventory["role_candidate_counts"]
            and dict(sorted(dependency_counts.items())) == inventory["dependency_source_kind_counts"]
        ),
        "repair_was_exercised_without_language_coercion_or_emission": bool(
            inventory["ignored_non_string_optional_context_leaf_count"] > 0
            and inventory["provenance"]["parser_repair"]
            == "ignore_non_string_optional_context_leaves_without_coercion"
            and not inventory["contains_input_target_argument_context_tokens_seeded_values_or_prompts"]
            and not inventory["provenance"]["contains_source_language"]
        ),
        "pairs_and_identifiers_are_unique_and_disjoint": bool(
            len({row["source_id"] for row in rows}) == len(rows)
            and len({row["full_context_pair_id"] for row in rows}) == len(rows)
            and len({row["ablated_context_pair_id"] for row in rows}) == len(rows)
            and inventory["development_test_identifiers_are_disjoint"]
            and inventory["pairs_share_source_id_input_and_target_by_construction"]
        ),
        "result_gates_and_decision_are_consistent": bool(
            reconstructed_gates == result["gates"]
            and result["passed"] == all(result["gates"].values())
            and result["decision"] == (
                "freeze_PRESTO_source_and_preregister_paired_context_population"
                if result["passed"] else "stop_V102r1_before_population_language_or_model_access"
            )
        ),
        "human_only_and_zero_access_boundary_holds": bool(
            inventory["synthetic_context_candidate_count"] == 0
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
        "schema_version": "102r1-presto-context-source-outcome-audit",
        "experiment": "v102r1_presto_context_source_outcome_audit",
        "passed": integrity_passed,
        "scientific_source_feasibility_passed": result["passed"],
        "decision": (
            "freeze_positive_V102r1_PRESTO_context_source_feasibility"
            if result["passed"] else "freeze_negative_V102r1_PRESTO_context_source_feasibility"
        ) if integrity_passed else "reject_V102r1_source_outcome",
        "checks": checks,
        "independent_summary": {
            "source_record_count": len(source_records),
            "eligible_candidate_count": len(rows),
            "role_candidate_counts": dict(sorted(role_counts.items())),
            "dependency_source_kind_counts": dict(sorted(dependency_counts.items())),
            "ignored_non_string_optional_context_leaf_count": (
                inventory["ignored_non_string_optional_context_leaf_count"]
            ),
            "failed_gates": failed_gates,
        },
        "additional_access": {
            "manual_utterance_inspection_count": 0, "model_load_count": 0,
            "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "source_lock": source_lock_path, "result": result_path,
        "inventory": inventory_path, "source_archive": source_path,
        "verifier": verifier_path, "audit": audit_path, "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "102r1-presto-context-source-outcome-lock",
        "experiment": "v102r1_presto_context_source_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_source_feasibility_passed": result["passed"],
            "decision": audit["decision"],
            "inventory_summary": result["inventory_summary"],
        },
        "authorization": {
            "modify_or_rerun_V102r1_source_stage": False,
            "preregister_paired_development_and_protected_test_population": result["passed"],
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
    outcome["source_archive_payload_sha256"] = file_sha256(source_path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({
        "lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(outcome_path),
    }, indent=2))


if __name__ == "__main__":
    main()
