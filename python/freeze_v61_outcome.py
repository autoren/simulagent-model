#!/usr/bin/env python3
"""Freeze the independently audited V61 result."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v61-long-horizon-policy-verification/post-result-audit.json"
    )
    parser.add_argument("--output", default="configs/v61-outcome-lock.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V61 outcome already frozen")
    audit = json.loads(audit_path.read_text())
    bindings = (
        ("result", "result_sha256"),
        ("evaluation_implementation_lock", "evaluation_implementation_lock_sha256"),
        ("verification_bundle_seal", "verification_bundle_seal_sha256"),
        ("evaluation_attempt", "evaluation_attempt_sha256"),
        ("results_summary", "results_summary_sha256"),
    )
    if (
        not audit["passed"]
        or not all(audit["checks"].values())
        or any(file_sha256(PROJECT_ROOT / audit[path]) != audit[digest] for path, digest in bindings)
    ):
        raise RuntimeError("V61 post-result audit is not intact and passing")
    lock = {
        "schema_version": 61,
        "experiment": "v61_outcome_lock",
        "qualification_passed": True,
        "result": audit["result"], "result_sha256": audit["result_sha256"],
        "evaluation_implementation_lock": audit["evaluation_implementation_lock"],
        "evaluation_implementation_lock_sha256": audit["evaluation_implementation_lock_sha256"],
        "verification_bundle_seal": audit["verification_bundle_seal"],
        "verification_bundle_seal_sha256": audit["verification_bundle_seal_sha256"],
        "evaluation_attempt": audit["evaluation_attempt"],
        "evaluation_attempt_sha256": audit["evaluation_attempt_sha256"],
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "results_summary": audit["results_summary"],
        "results_summary_sha256": audit["results_summary_sha256"],
        "claim_boundary": {
            "bounded_exact_posterior_execution_of_frozen_v60_policies": True,
            "search_algorithm_optimality": False,
            "formal_or_worst_case_safety": False,
            "parameter_uniform_or_unbounded_guarantee": False,
            "human_authored_language_robustness": False,
            "model_or_adapter_performance": False,
        },
        "authorization": {
            "continue_to_next_preregistered_stage": True,
            "rerun_v61_candidate_verification": False,
            "modify_v61_design_implementation_bundle_or_gates": False,
            "treat_synthetic_v58_as_human": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
