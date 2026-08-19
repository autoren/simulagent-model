#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v175_certification_aware_planner_development import DEPENDENCY_KEYS, reconstruct
from v175_certification_aware_planner_development import (
    evaluate_benefit,
    evaluate_safety_gates,
    evaluate_strong,
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v175-certification-aware-planner-development-lock.json"
    result_path = PROJECT_ROOT / "outputs/v175-certification-aware-planner-development/evaluation/result.json"
    doc_path = PROJECT_ROOT / "docs/v175-certification-aware-planner-development-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v175_certification_aware_planner_development_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v175-certification-aware-planner-development/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v175-certification-aware-planner-development-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V175 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V175 results before freezing")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    evaluation = reconstruct(lock)
    config = lock["config_payload"]
    safety = evaluate_safety_gates(evaluation, result["access"], config)
    benefit = evaluate_benefit(evaluation, config)
    strong = evaluate_strong(evaluation, config)
    safety_passed = all(safety.values())
    beneficial = all(benefit.values())
    strong_development = all(strong.values())
    if safety_passed and beneficial and strong_development:
        expected_decision = config["decisionRule"]["ifSafetyBenefitAndStrongPass"]
    elif safety_passed and beneficial:
        expected_decision = config["decisionRule"]["ifSafetyAndBenefitPassButStrongFails"]
    elif safety_passed:
        expected_decision = config["decisionRule"]["ifSafetyPassesButBenefitFails"]
    else:
        expected_decision = config["decisionRule"]["otherwise"]
    expected_outputs = {
        "state_policy_results": {
            "state_policy_results": evaluation["state_policy_results"],
            "target_cases_subsampled": False,
        },
        "development_summary": evaluation["summary"],
        "target_result_digest": {
            "target_policy_score_count": evaluation["summary"]["target_policy_score_count"],
            "target_result_payload_sha256": evaluation["summary"]["target_result_payload_sha256"],
            "full_target_payload_reconstructed_not_persisted": True,
        },
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, value in expected_outputs.items()
    )
    checks = {
        "lock_and_every_dependency_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "state_scores_summary_and_target_digest_reconstruct_exactly": outputs_exact,
        "gates_thresholds_and_decision_reconstruct_exactly": bool(
            result["summary"] == evaluation["summary"]
            and result["integrity_and_safety_gates"] == safety
            and result["benefit_thresholds"] == benefit
            and result["strong_development_thresholds"] == strong
            and result["passed"] == safety_passed
            and result["beneficial"] == beneficial
            and result["strong_development"] == strong_development
            and result["decision"] == expected_decision
        ),
        "complete_population_policy_and_DP_reconstruction_hold": bool(
            evaluation["summary"]["state_count"] == 132
            and evaluation["summary"]["target_count"] == 4224
            and evaluation["summary"]["target_policy_score_count"] == 29568
            and evaluation["summary"]["exact_DP_risk_reconstruction_rate"] == 1.0
        ),
        "gate_sandbox_authority_and_no_real_effect_boundary_hold": bool(
            evaluation["summary"]["false_trusted_route_probability"] == 0.0
            and evaluation["summary"]["provisional_sandbox_entry_probability"] == 0.0
            and evaluation["summary"]["planner_commit_authorization_count"] == 0
            and evaluation["summary"]["sandbox_exactness"] == 1.0
            and evaluation["summary"]["invariant_preservation"] == 1.0
            and evaluation["summary"]["provenance_and_restart_verification"] == 1.0
            and all(
                result["access"][key] == 0
                for key in (
                    "evaluation_record_count",
                    "manual_judgment_count",
                    "model_load_count",
                    "model_generation_count",
                    "API_call_count",
                    "training_run_count",
                    "ontology_registration_count",
                    "trusted_real_state_mutation_count",
                    "real_service_call_count",
                    "external_side_effect_count",
                    "actual_execution_count",
                )
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "175-certification-aware-planner-development-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_safety_gates_passed": safety_passed,
        "beneficial": beneficial,
        "strong_development": strong_development,
        "decision": "freeze_verified_V175_development_outcome" if passed else "reject_V175_outcome",
        "checks": checks,
        "independent_summary": evaluation["summary"],
        "additional_access": {
            "formal_target_policy_score_count": evaluation["summary"]["target_policy_score_count"],
            "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "development_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V174_outcome": PROJECT_ROOT / lock["parent_V174_outcome"],
        "source_V173_outcome": PROJECT_ROOT / lock["source_V173_outcome"],
        "source_V171_outcome": PROJECT_ROOT / lock["source_V171_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "175-certification-aware-planner-development-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_safety_gates_passed": safety_passed,
            "beneficial": beneficial,
            "strong_development": strong_development,
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rerun_select_subsample_or_tune_V175": False,
            "design_fresh_exact_signature_disjoint_confirmation": bool(
                safety_passed and beneficial
            ),
            "run_confirmation_without_separate_population_and_lock": False,
            "allow_planner_model_hidden_target_or_provisional_commit_authority": False,
            "run_model_register_mutate_real_state_call_service_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
