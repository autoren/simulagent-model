#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v168_fixed_ontology_reversible_sandbox import DEPENDENCY_KEYS, reconstruct
from v168_fixed_ontology_reversible_sandbox import evaluate_gates


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox-lock.json"
    result_path = PROJECT_ROOT / "outputs/v168-fixed-ontology-reversible-sandbox/census/result.json"
    doc_path = PROJECT_ROOT / "docs/v168-fixed-ontology-reversible-sandbox-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v168_fixed_ontology_reversible_sandbox_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v168-fixed-ontology-reversible-sandbox/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V168 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V168 results before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    artifacts = reconstruct(lock)
    config = lock["config_payload"]
    gates = evaluate_gates(artifacts["evaluation"], result["access"], config)
    expected_outputs = {
        "fixture_manifest": {"fixtures": artifacts["fixtures"], "project_authored_development": True},
        "transaction_results": {"results": artifacts["evaluation"]["results"], "contains_language": False},
        "sandbox_summary": artifacts["evaluation"]["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"]) == result["output_integrity"][key]["sha256"]
        for key, value in expected_outputs.items()
    )
    expected_decision = config["decisionRule"]["ifEverySandboxGatePasses"] if all(gates.values()) else config["decisionRule"]["otherwise"]
    checks = {
        "sandbox_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "fixtures_results_and_summary_reconstruct_exactly": bool(outputs_exact and result["summary"] == artifacts["evaluation"]["summary"]),
        "gates_pass_and_decision_reconstructs": bool(gates == result["gates"] and result["passed"] == all(gates.values()) and result["decision"] == expected_decision),
        "all_state_integrity_recovery_and_provenance_metrics_are_exact": all(result["summary"][key] == 1.0 for key in (
            "expected_disposition_accuracy", "exact_final_target_state", "rejected_state_immutability",
            "preview_nonmutation", "preview_commit_parity", "atomic_multi_entity_commit",
            "explicit_rollback_recovery", "verification_failure_rollback_recovery", "invariant_preservation",
            "zero_unauthorized_commit_mutation", "fault_detection", "provenance_chain_validity",
        )),
        "fixed_ontology_model_and_real_effect_boundary_holds": all(result["access"][key] == 0 for key in (
            "evaluation_record_count", "manual_judgment_count", "model_load_count", "model_generation_count",
            "API_call_count", "training_run_count", "provisional_ontology_use_count", "real_service_call_count",
            "external_side_effect_count", "real_execution_count",
        )),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "168-fixed-ontology-reversible-sandbox-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_sandbox_gates_passed": result["passed"],
        "decision": "freeze_positive_V168_simulated_sandbox_outcome" if passed and result["passed"] else "reject_V168_outcome",
        "checks": checks,
        "independent_summary": artifacts["evaluation"]["summary"],
        "additional_access": {"model_load_count": 0, "API_call_count": 0, "real_service_call_count": 0, "real_execution_count": 0},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "sandbox_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V167r1_outcome": PROJECT_ROOT / lock["parent_V167r1_outcome"],
    }
    for key, integrity in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / integrity["path"]
    outcome: dict[str, Any] = {
        "schema_version": "168-fixed-ontology-reversible-sandbox-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {"passed": True, "scientific_sandbox_gates_passed": result["passed"], "decision": result["decision"], "summary": result["summary"]},
        "authorization": {
            "modify_or_rerun_V168": False,
            "retain_V168_as_positive_project_authored_simulated_development_evidence": True,
            "claim_external_or_deployment_validation": False,
            "create_or_open_evaluation_population_without_separate_design": False,
            "integrate_provisional_ontology_candidate": False,
            "run_local_or_API_model": False,
            "call_real_service_or_tool": False,
            "perform_external_side_effect_or_real_execution": False,
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
