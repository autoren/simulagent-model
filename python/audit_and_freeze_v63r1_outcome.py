#!/usr/bin/env python3
"""Audit and freeze the passing V63r1 measurement-repair outcome."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def subsection_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default="outputs/v63r1-repeat-pooling-repair/evaluation/result.json"
    )
    parser.add_argument("--summary", default="docs/v63r1-results.md")
    parser.add_argument(
        "--audit", default="outputs/v63r1-repeat-pooling-repair/post-result-audit.json"
    )
    parser.add_argument("--output", default="configs/v63r1-outcome-lock.json")
    args = parser.parse_args()
    result_path = (PROJECT_ROOT / args.result).resolve()
    summary_path = (PROJECT_ROOT / args.summary).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V63r1 outcome already frozen")
    result = json.loads(result_path.read_text())
    evaluation_lock_path = (
        PROJECT_ROOT / result["bindings"]["evaluation_implementation_lock"]
    ).resolve()
    evaluation_lock = json.loads(evaluation_lock_path.read_text())
    design_path = (PROJECT_ROOT / evaluation_lock["design_lock"]).resolve()
    design = json.loads(design_path.read_text())
    original_outcome_path = (PROJECT_ROOT / design["source_v63_outcome_lock"]).resolve()
    original_outcome = json.loads(original_outcome_path.read_text())
    original_result_path = (PROJECT_ROOT / original_outcome["result"]).resolve()
    original = json.loads(original_result_path.read_text())
    errors = []
    bindings_ok = bool(
        file_sha256(evaluation_lock_path)
        == result["bindings"]["evaluation_implementation_lock_sha256"]
        and file_sha256(design_path) == evaluation_lock["design_lock_sha256"]
        and file_sha256(original_result_path)
        == result["reuse_bindings"]["original_v63_result_sha256"]
        and file_sha256(original_result_path) == original_outcome["result_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for path, digest in evaluation_lock["source_sha256"].items()
        )
    )
    if not bindings_ok:
        errors.append("V63r1 repair, original result, or evaluator binding is not intact")
    gates_ok = bool(
        result["passed"]
        and all(result["gate_checks"].values())
        and result["decision"]
        == "authorize_preregistration_of_separate_multi_action_external_EIG_stage"
        and result["exact_benchmark"]["by_budget"]["509"]["mean_joint_tv"] <= 0.06
    )
    if not gates_ok:
        errors.append("V63r1 did not pass every frozen gate")
    reuse_ok = bool(
        subsection_sha256(original["simulation_based_calibration"])
        == result["reuse_bindings"]["sbc_summary_sha256"]
        and subsection_sha256(original["scale_stress"])
        == result["reuse_bindings"]["scale_summary_sha256"]
        and subsection_sha256(original["runtime_crosscheck"])
        == result["reuse_bindings"]["runtime_summary_sha256"]
        and file_sha256(
            PROJECT_ROOT / result["reuse_bindings"]["original_runtime_result"]
        ) == result["reuse_bindings"]["original_runtime_result_sha256"]
    )
    if not reuse_ok:
        errors.append("V63r1 reused SBC, scale, or runtime subsection is not byte-bound")
    access = result["access"]
    access_ok = bool(
        access["logical_repair_attempts"] == 1
        and access["original_v63_reruns"] == 0
        and access["exact_repair_records"] == 32
        and access["SBC_reruns"] == 0
        and access["scale_reruns"] == 0
        and access["runtime_reruns"] == 0
        and access["candidate_truth_fields_passed_to_inference"] == 0
        and access["human_record_access_count"] == 0
        and access["simulated_human_record_count"] == 0
        and access["model_forward_pass_count"] == 0
        and access["adapter_training_run_count"] == 0
    )
    if not access_ok:
        errors.append("V63r1 repair crossed a rerun, truth, human, or model firewall")
    boundary_ok = bool(
        result["measurement_repair_not_independent_replication"]
        and result["original_v63_remains_failed"]
        and not original_outcome["qualification_passed"]
    )
    if not boundary_ok:
        errors.append("V63r1 measurement-repair or original-failure boundary is missing")
    audit = {
        "schema_version": "63r1",
        "experiment": "v63r1_post_result_audit",
        "passed": not errors,
        "decision": "freeze_v63r1_and_authorize_multi_action_EIG_preregistration_only" if not errors else "reject_v63r1_outcome",
        "errors": errors,
        "checks": {
            "repair_original_and_evaluator_bindings": bindings_ok,
            "all_noncompensatory_gates": gates_ok,
            "SBC_scale_and_runtime_reuse": reuse_ok,
            "attempt_rerun_truth_human_and_model_firewalls": access_ok,
            "measurement_repair_and_original_failure_boundary": boundary_ok,
        },
        "failed_gates": [name for name, passed in result["gate_checks"].items() if not passed],
        "primary_mean_joint_tv": result["exact_benchmark"]["by_budget"]["509"]["mean_joint_tv"],
        "data_access": access,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "63r1",
        "experiment": "v63r1_outcome_lock",
        "qualification_passed": True,
        "original_v63_qualification_passed": False,
        "measurement_repair_not_independent_replication": True,
        "decision": result["decision"],
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "evaluation_implementation_lock": str(evaluation_lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_lock_path),
        "source_v63_outcome_lock": str(original_outcome_path.relative_to(PROJECT_ROOT)),
        "source_v63_outcome_lock_sha256": file_sha256(original_outcome_path),
        "authorization": {
            "modify_or_rerun_v63_or_v63r1": False,
            "preregister_separate_multi_action_external_EIG_stage": True,
            "construct_or_run_EIG_population": False,
            "use_tiger_as_substantive_EIG_benchmark": False,
            "reward_or_planning_evaluation": False,
            "formal_verification": False,
            "access_human_v58_records": False,
            "simulate_human_records": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
