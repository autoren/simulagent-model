#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from run_v111_open_world_separability_audit import (
    decision_for, evaluate_integrity_gates, payload_hash, persisted_analysis, reconstruct,
)
from v111_open_world_separability_audit import selected_rule_passes


def main() -> None:
    analysis_lock_path = PROJECT_ROOT / "configs/v111-open-world-separability-audit-lock.json"
    result_path = PROJECT_ROOT / "outputs/v111-open-world-separability-audit/development-census/result.json"
    doc_path = PROJECT_ROOT / "docs/v111-open-world-separability-audit-results.md"
    roadmap_path = PROJECT_ROOT / "docs/open-world-language-research-direction.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v111_separability_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v111-open-world-separability-audit/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v111-open-world-separability-audit-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V111 outcome is already frozen")
    if not doc_path.is_file() or not roadmap_path.is_file():
        raise RuntimeError("write the V111 result and update the roadmap before freezing")

    lock = json.loads(analysis_lock_path.read_text())
    result = json.loads(result_path.read_text())
    analysis, metadata = reconstruct(lock)
    aggregate = persisted_analysis(analysis, metadata)
    access = result["access"]
    integrity_gates = evaluate_integrity_gates(analysis, metadata, access, lock["config_payload"])
    integrity_passed = all(integrity_gates.values())
    quality_gate_pass = selected_rule_passes(analysis, lock["config_payload"])
    oracle_feasible_count = analysis["evaluation_oracle"]["feasible_candidate_count"]
    expected_decision = decision_for(quality_gate_pass, oracle_feasible_count)
    dependency_keys = (
        "config", "parent_outcome", "parent_analysis_lock", "parent_result", "V109_result",
        "baseline_outcome", "baseline_lock", "source_archive", "development_language",
        "visible_catalog", "secondary_membership", "plan", "protocol", "tests",
        "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "analysis_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "aggregate_analysis_integrity_gates_quality_and_decision_reconstruct_exactly": bool(
            aggregate == result["analysis"]
            and integrity_gates == result["integrity_gates"]
            and integrity_passed == result["passed"]
            and quality_gate_pass == result["quality_gate_pass"]
            and expected_decision == result["decision"]
        ),
        "persisted_aggregate_output_is_exact": all(
            file_sha256(PROJECT_ROOT / value["path"]) == value["sha256"]
            for value in result["output_integrity"].values()
        ),
        "no_individual_feature_identifier_language_or_raw_response_was_emitted": (
            aggregate["individual_feature_or_identifier_emission_count"] == 0
        ),
        "evaluation_oracle_is_diagnostic_only": bool(
            not lock["config_payload"]["decisionRule"]["evaluationOracleMayAuthorizePolicy"]
            and not lock["config_payload"]["decisionRule"]["passAuthorizesProtectedTestInductionPlanningAPITrainingOrExecution"]
        ),
        "zero_model_protected_test_API_training_service_and_effect_access": all(
            access[key] == 0 for key in (
                "protected_test_language_read_count", "manual_language_or_raw_response_inspection_count",
                "model_load_count", "model_generation_count", "LLM_API_call_count",
                "adapter_training_run_count", "real_service_call_count", "external_side_effect_count",
            )
        ),
    }
    audit_passed = all(checks.values())
    audit = {
        "schema_version": "111-open-world-separability-audit-outcome-audit",
        "experiment": "v111_existing_evidence_deterministic_novelty_separability_outcome_audit",
        "passed": audit_passed,
        "quality_gate_pass": quality_gate_pass,
        "decision": expected_decision,
        "checks": checks,
        "independent_analysis": aggregate,
        "independent_integrity_gates": integrity_gates,
        "additional_access": {
            "source_archive_read_count": 1, "development_language_read_count": 1,
            "V109_result_automatic_read_count": 1, "V110_result_automatic_read_count": 0,
            "protected_test_language_read_count": 0,
            "manual_language_or_raw_response_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    close_interface = not quality_gate_pass and oracle_feasible_count == 0
    require_fresh_evidence = not quality_gate_pass and oracle_feasible_count > 0
    dependencies = {
        "analysis_lock": analysis_lock_path, "result": result_path,
        "verifier": verifier_path, "audit": audit_path,
        "results_document": doc_path, "research_roadmap": roadmap_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "111-open-world-separability-audit-outcome-lock",
        "experiment": "v111_existing_evidence_deterministic_novelty_separability_outcome_lock",
        "outcome": {
            "passed": True,
            "quality_gate_pass": quality_gate_pass,
            "decision": expected_decision,
            "selected_rule": aggregate["calibration"]["selected"]["rule"],
            "selected_evaluation_metrics": aggregate["selected_evaluation_metrics"],
            "calibration_feasible_candidate_count": aggregate["calibration"]["feasible_candidate_count"],
            "evaluation_oracle_feasible_candidate_count": oracle_feasible_count,
        },
        "authorization": {
            "modify_rerun_or_retune_V111": False,
            "preregister_full_deterministic_development_policy": bool(quality_gate_pass),
            "close_current_single_turn_evidence_interface_for_simple_deterministic_novelty_gating": close_interface,
            "seek_genuinely_fresh_evidence_population": require_fresh_evidence or close_interface,
            "read_protected_test_before_separate_lock": False,
            "proceed_to_schema_or_mechanic_induction": False,
            "proceed_to_richer_sequential_decision_problem": False,
            "run_additional_local_or_API_model_or_train_adapter": False,
            "prune_hypotheses_define_capability_or_grant_belief_action_execution_authority": False,
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
