#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v173_trusted_only_shadow_integration import (
    evaluate_benefit_thresholds,
    evaluate_integration,
    evaluate_integrity_and_safety_gates,
    evaluate_strong_thresholds,
)


DEPENDENCY_KEYS = (
    "config",
    "parent_V172_outcome",
    "source_V170_outcome",
    "source_V171_outcome",
    "source_V171_lock",
    "source_V167_planner_lock",
    "source_V167r1_metric_repair",
    "constraint_states",
    "eligible_state_ids",
    "target_cases",
    "plan",
    "protocol",
    "tests",
    "runner",
    "verifier",
    "auditor",
    "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> dict[str, Any]:
    states = json.loads((PROJECT_ROOT / lock["constraint_states"]).read_text())
    eligible = json.loads((PROJECT_ROOT / lock["eligible_state_ids"]).read_text())
    targets = json.loads((PROJECT_ROOT / lock["target_cases"]).read_text())
    return evaluate_integration(
        states,
        eligible,
        targets,
        lock["V167_config_payload"],
        lock["V171_config_payload"],
    )


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v173-trusted-only-shadow-integration-lock.json"
    output_root = PROJECT_ROOT / "outputs/v173-trusted-only-shadow-integration/evaluation"
    if output_root.exists():
        raise RuntimeError("V173 formal integration may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V173 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V173 dependency drifted: {key}")

    evaluation = reconstruct(lock)
    config = lock["config_payload"]
    access = {
        "formal_target_policy_score_count": evaluation["summary"]["target_policy_score_count"],
        "simulated_sandbox_transaction_count": evaluation["summary"]["simulated_sandbox_transaction_count"],
        "evaluation_record_count": 0,
        "manual_judgment_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_real_state_mutation_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    safety = evaluate_integrity_and_safety_gates(evaluation, access, config)
    benefit = evaluate_benefit_thresholds(evaluation, config)
    strong = evaluate_strong_thresholds(evaluation, config)
    safety_passed = all(safety.values())
    benefit_passed = all(benefit.values())
    strong_passed = all(strong.values())
    if safety_passed and benefit_passed and strong_passed:
        decision = config["decisionRule"]["ifSafetyBenefitAndStrongThresholdsPass"]
    elif safety_passed and benefit_passed:
        decision = config["decisionRule"]["ifSafetyAndBenefitPassButStrongThresholdsFail"]
    elif safety_passed:
        decision = config["decisionRule"]["ifSafetyPassesButBenefitFails"]
    else:
        decision = config["decisionRule"]["otherwise"]

    state_path = output_root / "state-policy-results.json"
    summary_path = output_root / "integration-summary.json"
    digest_path = output_root / "target-result-digest.json"
    write_json(state_path, {"state_policy_results": evaluation["state_policy_results"], "target_cases_subsampled": False})
    write_json(summary_path, evaluation["summary"])
    write_json(
        digest_path,
        {
            "target_policy_score_count": evaluation["summary"]["target_policy_score_count"],
            "target_result_payload_sha256": evaluation["target_result_payload_sha256"],
            "full_target_payload_reconstructed_not_persisted": True,
        },
    )
    output_integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in {
            "state_policy_results": state_path,
            "integration_summary": summary_path,
            "target_result_digest": digest_path,
        }.items()
    }
    result = {
        "schema_version": "173-trusted-only-shadow-integration-result",
        "experiment": config["experiment"],
        "passed": safety_passed,
        "beneficial": benefit_passed,
        "strong_integration": strong_passed,
        "decision": decision,
        "summary": evaluation["summary"],
        "integrity_and_safety_gates": safety,
        "benefit_thresholds": benefit,
        "strong_integration_thresholds": strong,
        "access": access,
        "output_integrity": output_integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not safety_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
