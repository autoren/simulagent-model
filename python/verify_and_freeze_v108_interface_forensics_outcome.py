#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from run_v108_open_world_interface_forensics import payload_hash, reconstruct
from v108_open_world_interface_forensics import aggregate_only_analysis, evaluate_forensics_gates


def main() -> None:
    diagnostic_lock_path = PROJECT_ROOT / "configs/v108-open-world-interface-forensics-lock.json"
    result_path = PROJECT_ROOT / "outputs/v108-open-world-interface-forensics/forensics/result.json"
    doc_path = PROJECT_ROOT / "docs/v108-open-world-interface-forensics-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v108_interface_forensics_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v108-open-world-interface-forensics/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v108-open-world-interface-forensics-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V108 diagnostic outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V108 result before freezing")
    lock = json.loads(diagnostic_lock_path.read_text())
    result = json.loads(result_path.read_text())
    analysis, model_result = reconstruct(lock)
    aggregate = aggregate_only_analysis(analysis)
    access = result["access"]
    gates = evaluate_forensics_gates(analysis, model_result["metrics"], access, lock["config_payload"])
    gates["aggregate_output_contains_no_raw_response_or_individual_identifier"] = analysis["raw_response_or_identifier_emission_count"] == 0
    dependency_keys = (
        "config", "parent_model_outcome", "implementation_lock", "baseline_outcome",
        "model_result", "selected_population", "development_membership", "visible_catalog",
        "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    aggregate_path = PROJECT_ROOT / result["output_integrity"]["aggregate_diagnostic"]["path"]
    checks = {
        "diagnostic_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "aggregate_diagnostic_reconstructs_exactly": bool(
            json.loads(aggregate_path.read_text()) == aggregate
            and file_sha256(aggregate_path) == result["output_integrity"]["aggregate_diagnostic"]["sha256"]
            and result["analysis"] == aggregate
        ),
        "gates_pass_and_decision_reconstruct_exactly": bool(
            gates == result["gates"] and result["passed"] == all(gates.values())
            and result["decision"] == (
                "format_mismatch_is_dominant_preregister_fresh_constrained_typed_interface"
                if result["passed"] else "format_mismatch_is_not_sufficient_prioritize_sequential_clarification"
            )
        ),
        "original_V107_metrics_are_unchanged": analysis["original_metrics"] == model_result["metrics"],
        "zero_language_model_test_and_effect_access_boundary_holds": all(
            access[key] == 0 for key in (
                "development_language_read_count", "protected_test_language_read_count",
                "manual_raw_response_inspection_count", "model_load_count", "model_generation_count",
                "LLM_API_call_count", "adapter_training_run_count", "real_service_call_count",
                "external_side_effect_count",
            )
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "108-open-world-interface-forensics-outcome-audit",
        "experiment": "v108_open_world_interface_forensics_outcome_audit",
        "passed": integrity_passed,
        "format_dominance_passed": result["passed"],
        "decision": (
            "freeze_positive_format_dominance_diagnosis" if integrity_passed and result["passed"]
            else "freeze_negative_format_dominance_diagnosis"
        ),
        "checks": checks, "independent_analysis": aggregate,
        "additional_access": {
            "development_language_read_count": 0, "protected_test_language_read_count": 0,
            "manual_raw_response_inspection_count": 0, "model_load_count": 0,
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
        "diagnostic_lock": diagnostic_lock_path, "result": result_path,
        "aggregate_diagnostic": aggregate_path, "verifier": verifier_path,
        "audit": audit_path, "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "108-open-world-interface-forensics-outcome-lock",
        "experiment": "v108_open_world_interface_forensics_outcome_lock",
        "outcome": {
            "passed": True, "format_dominance_passed": result["passed"],
            "decision": audit["decision"], "analysis": aggregate,
        },
        "authorization": {
            "modify_or_rerun_V108_or_replace_V107": False,
            "preregister_fresh_constrained_typed_interface_development_study": bool(result["passed"]),
            "preregister_sequential_clarification_study": not result["passed"],
            "read_protected_test_or_run_model_before_fresh_lock": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_capability_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
