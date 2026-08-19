#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v178_one_corruption_robust_certificate_feasibility import (
    DEPENDENCY_KEYS,
    reconstruct,
    terminal_adaptive_completion,
)
from v178_one_corruption_robust_certificate_feasibility import evaluate_gates


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v178-one-corruption-robust-certificate-feasibility-lock.json"
    result_path = PROJECT_ROOT / "outputs/v178-one-corruption-robust-certificate-feasibility/census/result.json"
    doc_path = PROJECT_ROOT / "docs/v178-one-corruption-robust-certificate-feasibility-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v178_one_corruption_robust_certificate_feasibility_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v178-one-corruption-robust-certificate-feasibility/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v178-one-corruption-robust-certificate-feasibility-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V178 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V178 results before freezing")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    evaluation = reconstruct(lock)
    config = lock["config_payload"]
    gates = evaluate_gates(evaluation, result["access"], config)
    passed_gates = all(gates.values())
    positive = terminal_adaptive_completion(evaluation, config) > 0
    if passed_gates and positive:
        expected_decision = config["decisionRule"][
            "ifEveryGatePassesAndTargetBlindTrustedCompletionIsPositive"
        ]
    elif passed_gates:
        expected_decision = config["decisionRule"][
            "ifEveryGatePassesButTargetBlindTrustedCompletionIsZero"
        ]
    else:
        expected_decision = config["decisionRule"]["otherwise"]
    expected_outputs = {
        "target_robust_certificate_results": {
            "target_results": evaluation["target_results"],
            "target_subsampling": False,
        },
        "state_adaptive_opportunity_results": {
            "state_results": evaluation["state_results"],
            "policy_cost_scored": False,
        },
        "feasibility_summary": evaluation["summary"],
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
            and all(
                file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
                for key in DEPENDENCY_KEYS
            )
        ),
        "target_state_and_summary_outputs_reconstruct_exactly": outputs_exact,
        "gates_feasibility_and_decision_reconstruct_exactly": bool(
            result["summary"] == evaluation["summary"]
            and result["feasibility_gates"] == gates
            and result["passed"] == passed_gates
            and result["single_pass_target_blind_robust_feasibility_positive"] == positive
            and result["decision"] == expected_decision
        ),
        "complete_adversarial_structural_census_is_cost_and_sandbox_free": bool(
            evaluation["summary"]["state_count"] == 135
            and evaluation["summary"]["target_count"] == 2160
            and evaluation["summary"]["adversarial_target_scenario_count"] == 10800
            and result["access"]["planner_risk_or_cost_score_count"] == 0
            and result["access"]["sandbox_transaction_count"] == 0
        ),
        "robust_containment_unanimity_and_no_effect_boundary_hold": bool(
            evaluation["summary"]["robust_target_containment_rate"] == 1.0
            and evaluation["summary"]["false_trusted_route_probability"] == 0.0
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
    outcome_audit = {
        "schema_version": "178-one-corruption-robust-certificate-feasibility-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_feasibility_gates_passed": passed_gates,
        "single_pass_target_blind_robust_feasibility_positive": positive,
        "decision": "freeze_verified_V178_structural_boundary" if passed else "reject_V178_outcome",
        "checks": checks,
        "independent_summary": evaluation["summary"],
        "additional_access": {
            "planner_risk_or_cost_score_count": 0,
            "sandbox_transaction_count": 0,
            "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "feasibility_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V177_outcome": PROJECT_ROOT / lock["parent_V177_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "178-one-corruption-robust-certificate-feasibility-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_feasibility_gates_passed": passed_gates,
            "single_pass_target_blind_robust_feasibility_positive": positive,
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rerun_select_subsample_or_tune_V178": False,
            "design_separate_robust_planner_development": bool(
                passed_gates and positive
            ),
            "design_fixed_repeated_measurement_feasibility": bool(
                passed_gates and not positive
            ),
            "weaken_unanimity_or_add_posterior_threshold": False,
            "run_model_register_mutate_real_state_call_service_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
