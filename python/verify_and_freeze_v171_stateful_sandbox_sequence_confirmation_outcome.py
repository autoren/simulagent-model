#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v171_stateful_sandbox_sequence_confirmation import DEPENDENCY_KEYS, reconstruct
from v171_stateful_sandbox_sequence_confirmation import evaluate_gates


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v171-stateful-sandbox-sequence-confirmation/census/result.json"
    doc_path = PROJECT_ROOT / "docs/v171-stateful-sandbox-sequence-confirmation-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v171_stateful_sandbox_sequence_confirmation_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v171-stateful-sandbox-sequence-confirmation/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V171 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V171 results before freezing")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    artifacts = reconstruct(lock)
    config = lock["composed_config_payload"]
    design = lock["config_payload"]
    gates = evaluate_gates(artifacts["evaluation"], result["access"], config)
    expected_outputs = {
        "sequence_manifest": {"sequences": artifacts["fixtures"], "project_authored_procedural_confirmation": True},
        "sequence_results": {"results": artifacts["evaluation"]["results"], "contains_language": False},
        "confirmation_summary": artifacts["evaluation"]["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, value in expected_outputs.items()
    )
    passed_gates = all(gates.values())
    expected_decision = (
        design["decisionRule"]["ifEveryConfirmationGatePasses"]
        if passed_gates
        else design["decisionRule"]["otherwise"]
    )
    exact_metric_keys = (
        "expected_disposition_accuracy",
        "exact_oracle_final_state",
        "serializable_boundary_rate",
        "revision_race_rejection",
        "replay_idempotence",
        "crash_recovery",
        "partial_write_recovery",
        "repeated_rollback_idempotence",
        "provenance_restart_validity",
        "provenance_tamper_detection",
        "atomic_multi_entity_retention",
        "post_recovery_continuation",
        "invariant_preservation",
        "zero_unauthorized_retained_mutation",
        "provenance_chain_validity",
    )
    checks = {
        "lock_and_every_dependency_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "population_results_and_summary_reconstruct_exactly": bool(
            outputs_exact and result["summary"] == artifacts["evaluation"]["summary"]
        ),
        "gates_and_decision_reconstruct_exactly": bool(
            result["gates"] == gates
            and result["passed"] == passed_gates
            and result["decision"] == expected_decision
        ),
        "every_stateful_safety_recovery_and_provenance_metric_is_exact": all(
            result["summary"][key] == 1.0 for key in exact_metric_keys
        ),
        "fixed_ontology_model_and_real_effect_boundary_holds": all(
            result["access"][key] == 0
            for key in (
                "evaluation_record_count",
                "manual_judgment_count",
                "model_load_count",
                "model_generation_count",
                "API_call_count",
                "training_run_count",
                "provisional_ontology_use_count",
                "real_service_call_count",
                "external_side_effect_count",
                "real_execution_count",
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "171-stateful-sandbox-sequence-confirmation-outcome-audit",
        "experiment": design["experiment"],
        "passed": passed,
        "scientific_confirmation_gates_passed": result["passed"],
        "decision": "freeze_verified_V171_stateful_confirmation" if passed else "reject_V171_outcome",
        "checks": checks,
        "independent_summary": artifacts["evaluation"]["summary"],
        "additional_access": {"model_load_count": 0, "API_call_count": 0, "real_execution_count": 0},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "confirmation_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V170_outcome": PROJECT_ROOT / lock["parent_V170_outcome"],
        "source_V168_outcome": PROJECT_ROOT / lock["source_V168_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "171-stateful-sandbox-sequence-confirmation-outcome-lock",
        "experiment": design["experiment"],
        "outcome": {
            "passed": True,
            "scientific_confirmation_gates_passed": result["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rerun_select_or_tune_V171": False,
            "retain_outcome_without_posthoc_selection": True,
            "design_trusted_only_shadow_integration": bool(result["passed"]),
            "run_integration_without_separate_lock": False,
            "allow_provisional_candidate_to_authorize_commit": False,
            "run_model_call_real_service_mutate_real_state_or_execute": False,
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
