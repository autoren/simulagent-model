#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v104_massive_language_extraction import build_selected_language_artifacts, evaluate_extraction_gates


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    extraction_lock_path = PROJECT_ROOT / "configs/v104-massive-language-extraction-lock.json"
    result_path = PROJECT_ROOT / "outputs/v104-massive-language/selected-language/result.json"
    doc_path = PROJECT_ROOT / "docs/v104-massive-language-extraction-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v104_language_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v104-massive-language/language-outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v104-massive-language-extraction-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V104 extraction outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V104 extraction result before freezing")
    lock = json.loads(extraction_lock_path.read_text())
    result = json.loads(result_path.read_text())
    config = lock["config_payload"]
    population = json.loads((PROJECT_ROOT / config["selectedPopulation"]).read_text())
    inventory = json.loads((PROJECT_ROOT / config["sourceInventory"]).read_text())
    archive_bytes = (PROJECT_ROOT / config["sourceArchive"]).read_bytes()
    source_records, member = parse_massive_archive(archive_bytes, config["expectedLocaleMemberSuffix"])
    reconstructed = build_selected_language_artifacts(population, inventory, source_records, config)
    reconstructed_gates = evaluate_extraction_gates(reconstructed, config)
    reconstructed_gates["zero_manual_model_API_training_service_or_side_effect_access"] = True
    dependency_keys = (
        "config", "parent_population_outcome", "presto_closure_outcome", "selected_population",
        "source_inventory", "source_archive", "plan", "protocol", "tests", "runner",
        "verifier", "auditor", "design_audit",
    )
    output_exact = True
    for role, integrity in result["output_integrity"].items():
        path = PROJECT_ROOT / integrity["path"]
        output_exact = output_exact and bool(
            file_sha256(path) == integrity["file_sha256"]
            and read_jsonl(path) == reconstructed["role_records"][role]
            and len(read_jsonl(path)) == integrity["record_count"]
            and reconstructed["role_payload_sha256"][role] == integrity["payload_sha256"]
        )
    checks = {
        "extraction_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "source_member_and_artifacts_reconstruct_exactly": bool(output_exact and member == result["source_locale_member"]),
        "result_gates_and_decision_are_consistent": bool(
            reconstructed_gates == result["gates"]
            and result["passed"] == all(result["gates"].values())
            and result["decision"] == ("freeze_language_and_preregister_benchmark_interface" if result["passed"] else "stop_V104_before_prompt_or_model_access")
        ),
        "exact_selection_structure_familiarity_slots_and_disjointness_hold": all(
            reconstructed[key] for key in (
                "exact_selected_identifier_set", "exact_structural_ground_truth_match",
                "exact_familiarity_reconstruction", "exact_slot_type_count_reconstruction",
                "development_protected_test_disjoint",
            )
        ),
        "zero_manual_model_and_side_effect_access_boundary_holds": all(
            result["access"][key] == 0 for key in (
                "unselected_language_record_emission_count",
                "manual_development_utterance_inspection_count",
                "manual_protected_test_utterance_inspection_count", "model_load_count",
                "model_generation_count", "LLM_API_call_count", "adapter_training_run_count",
                "real_service_call_count", "external_side_effect_count",
            )
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "104-massive-language-extraction-outcome-audit",
        "experiment": "v104_massive_language_extraction_outcome_audit",
        "passed": integrity_passed,
        "scientific_extraction_passed": result["passed"],
        "decision": "freeze_positive_V104_MASSIVE_language_artifacts" if integrity_passed and result["passed"] else "reject_V104_language_outcome",
        "checks": checks,
        "independent_summary": result["extraction_summary"],
        "additional_access": {
            "manual_development_utterance_inspection_count": 0,
            "manual_protected_test_utterance_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0,
            "LLM_API_call_count": 0, "adapter_training_run_count": 0,
            "real_service_call_count": 0, "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "extraction_lock": extraction_lock_path, "result": result_path,
        "verifier": verifier_path, "audit": audit_path, "results_document": doc_path,
    }
    for role, integrity in result["output_integrity"].items():
        dependencies[f"{role}_language"] = PROJECT_ROOT / integrity["path"]
    outcome: dict[str, Any] = {
        "schema_version": "104-massive-language-extraction-outcome-lock",
        "experiment": "v104_massive_language_extraction_outcome_lock",
        "outcome": {
            "passed": True, "scientific_extraction_passed": result["passed"],
            "decision": audit["decision"], "extraction_summary": result["extraction_summary"],
            "output_integrity": result["output_integrity"],
        },
        "authorization": {
            "modify_or_rerun_V104_extraction": False,
            "preregister_benchmark_interface_prompt_controls_metrics_and_gates": result["passed"],
            "automatically_read_development_language_after_interface_lock": result["passed"],
            "read_protected_test_before_interface_lock": False,
            "manually_inspect_protected_test_language": False,
            "load_model_before_interface_and_baseline_outcomes": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
