#!/usr/bin/env python3
from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v179_triple_repetition_robust_feasibility import (
    evaluate_feasibility,
    evaluate_gates,
)


DEPENDENCY_KEYS = (
    "config",
    "parent_V178_outcome",
    "source_V176_outcome",
    "source_V167_planner_lock",
    "constraint_states",
    "eligible_state_ids",
    "target_cases",
    "roadmap",
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
    return evaluate_feasibility(
        states,
        eligible,
        targets,
        lock["V167_config_payload"],
        lock["config_payload"]["structuralCensus"]["blockHorizons"],
    )


def terminal_completion(evaluation: dict[str, Any], config: dict[str, Any]) -> Fraction:
    horizon = str(max(config["structuralCensus"]["blockHorizons"]))
    value = evaluation["summary"][
        "adaptive_worst_case_trusted_completion_by_block_horizon"
    ][horizon]
    return Fraction(value["numerator"], value["denominator"])


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v179-triple-repetition-robust-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v179-triple-repetition-robust-feasibility/census"
    if output_root.exists():
        raise RuntimeError("V179 formal census may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V179 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V179 dependency drifted: {key}")
    evaluation = reconstruct(lock)
    config = lock["config_payload"]
    access = {
        "formal_state_count": evaluation["summary"]["state_count"],
        "formal_target_count": evaluation["summary"]["target_count"],
        "planner_risk_or_cost_score_count": 0,
        "sandbox_transaction_count": 0,
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
    gates = evaluate_gates(evaluation, access, config)
    passed = all(gates.values())
    positive = terminal_completion(evaluation, config) > 0
    if passed and positive:
        decision = config["decisionRule"]["ifEveryGatePassesAndTargetBlindTrustedCompletionIsPositive"]
    elif passed:
        decision = config["decisionRule"]["ifEveryGatePassesButTargetBlindTrustedCompletionIsZero"]
    else:
        decision = config["decisionRule"]["otherwise"]
    targets_path = output_root / "target-certificate-results.json"
    states_path = output_root / "state-adaptive-opportunity-results.json"
    summary_path = output_root / "feasibility-summary.json"
    write_json(targets_path, {"target_results": evaluation["target_results"], "target_subsampling": False})
    write_json(states_path, {"state_results": evaluation["state_results"], "policy_cost_scored": False})
    write_json(summary_path, evaluation["summary"])
    output_integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in {
            "target_certificate_results": targets_path,
            "state_adaptive_opportunity_results": states_path,
            "feasibility_summary": summary_path,
        }.items()
    }
    result = {
        "schema_version": "179-triple-repetition-robust-feasibility-result",
        "experiment": config["experiment"],
        "passed": passed,
        "triple_repetition_target_blind_robust_feasibility_positive": positive,
        "decision": decision,
        "summary": evaluation["summary"],
        "feasibility_gates": gates,
        "access": access,
        "output_integrity": output_integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
