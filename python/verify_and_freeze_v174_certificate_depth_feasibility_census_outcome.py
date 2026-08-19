#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v174_certificate_depth_feasibility_census import DEPENDENCY_KEYS, reconstruct
from v174_certificate_depth_feasibility_census import evaluate_gates


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v174-certificate-depth-feasibility-census-lock.json"
    result_path = PROJECT_ROOT / "outputs/v174-certificate-depth-feasibility-census/census/result.json"
    doc_path = PROJECT_ROOT / "docs/v174-certificate-depth-feasibility-census-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v174_certificate_depth_feasibility_census_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v174-certificate-depth-feasibility-census/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v174-certificate-depth-feasibility-census-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V174 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V174 results before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    evaluation = reconstruct(lock)
    config = lock["config_payload"]
    gates = evaluate_gates(evaluation, result["access"], config)
    expected = {
        "target_certificate_results": {"target_results": evaluation["target_results"], "target_subsampling": False},
        "state_horizon_results": {"state_results": evaluation["state_results"], "policy_cost_scored": False},
        "feasibility_summary": evaluation["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, value in expected.items()
    )
    expected_decision = config["decisionRule"]["ifEveryFeasibilityGatePasses"] if all(gates.values()) else config["decisionRule"]["otherwise"]
    checks = {
        "lock_and_every_dependency_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "targets_states_and_summary_reconstruct_exactly": outputs_exact,
        "gates_summary_and_decision_reconstruct_exactly": bool(
            result["summary"] == evaluation["summary"]
            and result["gates"] == gates
            and result["passed"] == all(gates.values())
            and result["decision"] == expected_decision
        ),
        "structural_census_is_complete_exact_and_cost_free": bool(
            evaluation["summary"]["state_count"] == 132
            and evaluation["summary"]["target_count"] == 4224
            and evaluation["summary"]["certificate_validity_rate"] == 1.0
            and evaluation["summary"]["certificate_minimality_rate"] == 1.0
            and result["access"]["planner_risk_or_cost_score_count"] == 0
            and result["access"]["sandbox_transaction_count"] == 0
        ),
        "model_authority_effect_and_execution_boundary_holds": all(
            result["access"][key] == 0
            for key in (
                "evaluation_record_count",
                "manual_judgment_count",
                "model_load_count",
                "model_generation_count",
                "API_call_count",
                "training_run_count",
                "ontology_registration_count",
                "trusted_state_mutation_count",
                "real_service_call_count",
                "external_side_effect_count",
                "actual_execution_count",
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "174-certificate-depth-feasibility-census-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_feasibility_gates_passed": result["passed"],
        "decision": "freeze_verified_V174_structural_census" if passed else "reject_V174_outcome",
        "checks": checks,
        "independent_summary": evaluation["summary"],
        "additional_access": {"planner_risk_or_cost_score_count": 0, "model_load_count": 0, "actual_execution_count": 0},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "feasibility_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V173_outcome": PROJECT_ROOT / lock["parent_V173_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "174-certificate-depth-feasibility-census-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_feasibility_gates_passed": result["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rerun_select_or_tune_V174": False,
            "use_structural_horizon_curve_to_preregister_V175": bool(result["passed"]),
            "score_V175_without_separate_design_lock": False,
            "change_or_rerun_V173": False,
            "run_model_register_mutate_state_call_service_or_execute": False,
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
