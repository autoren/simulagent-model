#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v105_open_world_interface import (
    compile_visible_catalog, complete_hypothesis_universe, evaluate_interface_gates,
    select_controlled_insufficient_identifiers, validate_response,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    interface_lock_path = PROJECT_ROOT / "configs/v105-open-world-interface-lock.json"
    result_path = PROJECT_ROOT / "outputs/v105-open-world-interface/interface/result.json"
    doc_path = PROJECT_ROOT / "docs/v105-open-world-interface-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v105_interface_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v105-open-world-interface/interface-outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v105-open-world-interface-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V105 interface outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V105 interface result before freezing")
    lock = json.loads(interface_lock_path.read_text())
    result = json.loads(result_path.read_text())
    config = lock["config_payload"]
    archive_bytes = (PROJECT_ROOT / config["sourceArchive"]).read_bytes()
    records, member = parse_massive_archive(archive_bytes, config["expectedLocaleMemberSuffix"])
    inventory = json.loads((PROJECT_ROOT / config["sourceInventory"]).read_text())
    population = json.loads((PROJECT_ROOT / config["selectedPopulation"]).read_text())
    compiled = compile_visible_catalog(records, inventory, config)
    hypotheses = complete_hypothesis_universe(compiled["catalog"])
    controlled = select_controlled_insufficient_identifiers(population, config)
    reconstructed_gates = evaluate_interface_gates(compiled, len(hypotheses), controlled, config)
    fallback, fallback_valid, fallback_reason = validate_response("not-json", compiled["catalog"], config)
    reconstructed_gates["invalid_response_maps_to_safe_abstention"] = bool(
        not fallback_valid and fallback_reason == "invalid_json"
        and fallback == config["responseContract"]["invalidResponseFallback"]
    )
    reconstructed_gates["zero_selected_language_model_API_training_or_effect_access"] = True
    dependency_keys = (
        "config", "parent_language_outcome", "source_archive", "source_inventory",
        "selected_population", "plan", "protocol", "tests", "runner", "verifier",
        "auditor", "design_audit",
    )
    expected_outputs = {
        "visible_catalog": compiled["catalog"],
        "safe_hypothesis_universe": {"hypotheses": hypotheses, "payload_sha256": payload_hash({"hypotheses": hypotheses})},
        "controlled_insufficient_identifiers": controlled,
        "interface_manifest": {
            "schema_version": "105-open-world-interface-manifest",
            "catalog_sha256": compiled["catalog_sha256"],
            "hypothesis_count": len(hypotheses),
            "hypothesis_payload_sha256": payload_hash({"hypotheses": hypotheses}),
            "controlled_payload_sha256": controlled["payload_sha256"],
            "response_contract": config["responseContract"],
            "prompt_contract": config["promptContract"],
            "authority_boundary": config["authorityBoundary"],
        },
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, value in expected_outputs.items()
    )
    checks = {
        "interface_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "source_and_interface_artifacts_reconstruct_exactly": bool(outputs_exact and member == result["source_locale_member"]),
        "result_gates_and_decision_are_consistent": bool(
            reconstructed_gates == result["gates"]
            and result["passed"] == all(result["gates"].values())
            and result["decision"] == ("freeze_interface_and_preregister_language_benchmark" if result["passed"] else "stop_V105_before_language_or_model_access")
        ),
        "catalog_hypothesis_and_control_invariants_hold": bool(
            len(compiled["catalog"]["scenarios"]) == 3
            and len(compiled["catalog"]["intents"]) == 12
            and len(hypotheses) == 17
            and compiled["hidden_or_unsupported_schema_leak_count"] == 0
            and controlled["role_counts"] == {"development": 64, "protected_test": 64}
            and not controlled["contains_source_language"]
        ),
        "zero_selected_language_model_and_effect_access_boundary_holds": all(
            result["access"][key] == 0 for key in (
                "selected_development_language_read_count", "protected_test_language_read_count",
                "manual_utterance_inspection_count", "model_load_count", "model_generation_count",
                "LLM_API_call_count", "adapter_training_run_count", "real_service_call_count",
                "external_side_effect_count",
            )
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "105-open-world-interface-outcome-audit",
        "experiment": "v105_open_world_interface_outcome_audit",
        "passed": integrity_passed,
        "scientific_interface_passed": result["passed"],
        "decision": "freeze_positive_V105_open_world_interface" if integrity_passed and result["passed"] else "reject_V105_interface_outcome",
        "checks": checks,
        "independent_summary": result["interface_summary"],
        "additional_access": {
            "selected_development_language_read_count": 0,
            "protected_test_language_read_count": 0,
            "manual_utterance_inspection_count": 0,
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
        "interface_lock": interface_lock_path, "result": result_path,
        "verifier": verifier_path, "audit": audit_path, "results_document": doc_path,
    }
    for key, integrity in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / integrity["path"]
    outcome: dict[str, Any] = {
        "schema_version": "105-open-world-interface-outcome-lock",
        "experiment": "v105_open_world_interface_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_interface_passed": result["passed"],
            "decision": audit["decision"],
            "interface_summary": result["interface_summary"],
            "output_integrity": result["output_integrity"],
        },
        "authorization": {
            "modify_or_rerun_V105_interface": False,
            "preregister_baselines_metrics_costs_calibration_and_one_local_model": result["passed"],
            "automatically_read_development_language_after_next_benchmark_lock": result["passed"],
            "read_protected_test_before_baseline_and_development_outcomes": False,
            "manually_inspect_protected_test_language": False,
            "load_model_before_baseline_and_benchmark_lock": False,
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
