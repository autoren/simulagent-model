#!/usr/bin/env python3
"""Bind the immutable failed V63 outcome and authorize only a narrow measurement repair."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default="outputs/v63-external-unknown-dynamics/evaluation/result.json"
    )
    parser.add_argument("--summary", default="docs/v63-results.md")
    parser.add_argument(
        "--audit", default="outputs/v63-external-unknown-dynamics/post-result-audit.json"
    )
    parser.add_argument("--output", default="configs/v63-outcome-lock.json")
    args = parser.parse_args()
    result_path = (PROJECT_ROOT / args.result).resolve()
    summary_path = (PROJECT_ROOT / args.summary).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V63 outcome already frozen")
    result = json.loads(result_path.read_text())
    evaluation_lock_path = (
        PROJECT_ROOT / result["bindings"]["evaluation_implementation_lock"]
    ).resolve()
    evaluation_lock = json.loads(evaluation_lock_path.read_text())
    population_seal_path = (PROJECT_ROOT / evaluation_lock["population_seal"]).resolve()
    errors = []
    bindings_ok = bool(
        file_sha256(evaluation_lock_path)
        == result["bindings"]["evaluation_implementation_lock_sha256"]
        and file_sha256(population_seal_path) == result["bindings"]["population_seal_sha256"]
        and file_sha256(population_seal_path) == evaluation_lock["population_seal_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for path, digest in evaluation_lock["source_sha256"].items()
        )
    )
    if not bindings_ok:
        errors.append("V63 result, population, or evaluation source binding is not intact")
    failed_gates = [name for name, passed in result["gate_checks"].items() if not passed]
    failure_ok = bool(
        not result["passed"]
        and failed_gates == ["primary_mean_joint_tv"]
        and result["decision"] == "repair_or_reject_v63_inference_before_active_selection"
    )
    if not failure_ok:
        errors.append("V63 failure is not the single registered joint-TV gate")
    access = result["access"]
    access_ok = bool(
        access["logical_evaluation_attempts"] == 1
        and access["unexpected_evaluation_attempt_count"] == 0
        and access["candidate_truth_fields_passed_to_inference"] == 0
        and access["human_record_access_count"] == 0
        and access["simulated_human_record_count"] == 0
        and access["model_forward_pass_count"] == 0
        and access["adapter_training_run_count"] == 0
    )
    if not access_ok:
        errors.append("V63 result crossed an attempt, truth, human, or model firewall")
    audit = {
        "schema_version": 63,
        "experiment": "v63_failed_outcome_audit",
        "passed": not errors,
        "decision": "freeze_failed_v63_and_authorize_narrow_pooling_repair" if not errors else "reject_v63_outcome_binding",
        "errors": errors,
        "checks": {
            "result_population_and_evaluator_bindings": bindings_ok,
            "single_expected_failed_gate": failure_ok,
            "attempt_truth_human_and_model_firewalls": access_ok,
        },
        "failed_gates": failed_gates,
        "primary_mean_joint_tv": result["exact_benchmark"]["by_budget"]["509"]["mean_joint_tv"],
        "primary_joint_tv_gate": 0.06,
        "data_access": result["access"],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": 63,
        "experiment": "v63_failed_outcome_lock",
        "qualification_passed": False,
        "decision": result["decision"],
        "failed_gates": failed_gates,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "evaluation_implementation_lock": str(evaluation_lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_lock_path),
        "authorization": {
            "modify_or_rerun_original_v63": False,
            "write_and_audit_narrow_repeat_pooling_repair": True,
            "change_population_seeds_particles_bins_or_gates": False,
            "active_intervention_selection": False,
            "reward_or_planning_evaluation": False,
            "access_human_v58_records": False,
            "simulate_human_records": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
