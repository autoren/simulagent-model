#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v182_triple_repetition_robust_planner_fresh_confirmation import (
    DEPENDENCY_KEYS,
    reconstruct,
)
from v182_triple_repetition_robust_planner_fresh_confirmation import (
    evaluate_primary_confirmation,
    evaluate_safety_gates,
    evaluate_strong_confirmation,
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = (
        PROJECT_ROOT
        / "configs/v182-triple-repetition-robust-planner-fresh-confirmation-lock.json"
    )
    result_path = (
        PROJECT_ROOT
        / "outputs/v182-triple-repetition-robust-planner-fresh-confirmation/confirmation/result.json"
    )
    doc_path = (
        PROJECT_ROOT
        / "docs/v182-triple-repetition-robust-planner-fresh-confirmation-results.md"
    )
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v182_triple_repetition_robust_planner_fresh_confirmation_outcome.py"
    )
    audit_path = (
        PROJECT_ROOT
        / "outputs/v182-triple-repetition-robust-planner-fresh-confirmation/outcome-audit.json"
    )
    outcome_path = (
        PROJECT_ROOT
        / "configs/v182-triple-repetition-robust-planner-fresh-confirmation-outcome-lock.json"
    )
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V182 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V182 results before freezing")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    evaluation = reconstruct(lock)
    config = lock["config_payload"]
    safety = evaluate_safety_gates(evaluation, result["access"], config)
    primary = evaluate_primary_confirmation(evaluation, config)
    strong = evaluate_strong_confirmation(evaluation, config)
    safety_passed = all(safety.values())
    confirmed = all(primary.values())
    strong_confirmation = all(strong.values())
    if safety_passed and confirmed and strong_confirmation:
        expected_decision = config["decisionRule"]["ifSafetyPrimaryAndStrongPass"]
    elif safety_passed and confirmed:
        expected_decision = config["decisionRule"][
            "ifSafetyAndPrimaryPassButStrongFails"
        ]
    elif safety_passed:
        expected_decision = config["decisionRule"][
            "ifSafetyPassesButPrimaryFails"
        ]
    else:
        expected_decision = config["decisionRule"]["otherwise"]
    expected_outputs = {
        "state_policy_results": {
            "state_policy_results": evaluation["state_policy_results"],
            "target_cases_subsampled": False,
        },
        "confirmation_summary": evaluation["summary"],
        "target_result_digest": {
            "target_policy_score_count": evaluation["summary"][
                "target_policy_score_count"
            ],
            "target_result_payload_sha256": evaluation["summary"][
                "target_result_payload_sha256"
            ],
            "full_target_payload_reconstructed_not_persisted": True,
        },
        "oracle_certificate_digest": evaluation["certificate_digest"],
    }
    outputs_exact = all(
        json.loads(
            (PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()
        )
        == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, value in expected_outputs.items()
    )
    checks = {
        "lock_and_every_dependency_are_exact": bool(
            payload_hash(
                {
                    key: value
                    for key, value in lock.items()
                    if key != "lock_payload_sha256"
                }
            )
            == lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
                for key in DEPENDENCY_KEYS
            )
        ),
        "scores_summary_target_and_certificate_digests_reconstruct_exactly": outputs_exact,
        "gates_thresholds_and_decision_reconstruct_exactly": bool(
            result["summary"] == evaluation["summary"]
            and result["integrity_and_safety_gates"] == safety
            and result["primary_confirmation_thresholds"] == primary
            and result["strong_confirmation_thresholds"] == strong
            and result["passed"] == safety_passed
            and result["confirmed"] == confirmed
            and result["strong_confirmation"] == strong_confirmation
            and result["decision"] == expected_decision
        ),
        "complete_fresh_population_policy_and_DP_reconstruction_hold": bool(
            evaluation["summary"]["state_count"] == 66
            and evaluation["summary"]["target_count"] == 528
            and evaluation["summary"]["target_policy_score_count"] == 3696
            and evaluation["summary"]["exact_DP_risk_reconstruction_rate"] == 1.0
            and evaluation["summary"]["corruption_scenario_route_invariance_rate"]
            == 1.0
            and evaluation["summary"]["oracle_certificate_validity_rate"] == 1.0
        ),
        "gate_sandbox_authority_and_no_real_effect_boundary_hold": bool(
            evaluation["summary"]["false_trusted_route_probability"] == 0.0
            and evaluation["summary"]["provisional_sandbox_entry_probability"]
            == 0.0
            and evaluation["summary"]["planner_commit_authorization_count"] == 0
            and evaluation["summary"]["sandbox_exactness"] == 1.0
            and evaluation["summary"]["invariant_preservation"] == 1.0
            and evaluation["summary"]["provenance_and_restart_verification"]
            == 1.0
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
                    "real_sensor_or_service_call_count",
                    "external_side_effect_count",
                    "actual_execution_count",
                )
            )
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "182-triple-repetition-robust-planner-fresh-confirmation-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_safety_gates_passed": safety_passed,
        "confirmed": confirmed,
        "strong_confirmation": strong_confirmation,
        "decision": (
            "freeze_verified_V182_confirmation" if passed else "reject_V182_outcome"
        ),
        "checks": checks,
        "independent_summary": evaluation["summary"],
        "additional_access": {
            "formal_target_policy_score_count": evaluation["summary"][
                "target_policy_score_count"
            ],
            "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "confirmation_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V181_outcome": PROJECT_ROOT / lock["parent_V181_outcome"],
        "source_V180_outcome": PROJECT_ROOT / lock["source_V180_outcome"],
        "source_V171_outcome": PROJECT_ROOT / lock["source_V171_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "182-triple-repetition-robust-planner-fresh-confirmation-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_safety_gates_passed": safety_passed,
            "confirmed": confirmed,
            "strong_confirmation": strong_confirmation,
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rerun_select_subsample_or_tune_V182": False,
            "fixed_ontology_one_corruption_branch_closed_as_confirmed": bool(
                safety_passed and confirmed
            ),
            "run_additional_robustness_without_new_question_population_and_lock": False,
            "allow_planner_model_hidden_target_or_provisional_commit_authority": False,
            "run_model_register_mutate_real_state_call_sensor_service_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(outcome_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(outcome_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
