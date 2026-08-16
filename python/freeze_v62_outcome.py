#!/usr/bin/env python3
"""Freeze the independently audited V62 result."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v62-external-pomdp-transfer/post-result-audit.json"
    )
    parser.add_argument("--output", default="configs/v62-outcome-lock.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V62 outcome already frozen")
    audit = json.loads(audit_path.read_text())
    bindings = (
        ("result", "result_sha256"),
        ("evaluation_implementation_lock", "evaluation_implementation_lock_sha256"),
        ("external_bundle_seal", "external_bundle_seal_sha256"),
        ("evaluation_attempt", "evaluation_attempt_sha256"),
        ("results_summary", "results_summary_sha256"),
    )
    if (
        not audit["passed"]
        or not all(audit["checks"].values())
        or any(file_sha256(PROJECT_ROOT / audit[path]) != audit[digest] for path, digest in bindings)
    ):
        raise RuntimeError("V62 post-result audit is not intact and passing")
    lock = {
        "schema_version": 62,
        "experiment": "v62_outcome_lock",
        "qualification_passed": True,
        "result": audit["result"],
        "result_sha256": audit["result_sha256"],
        "evaluation_implementation_lock": audit["evaluation_implementation_lock"],
        "evaluation_implementation_lock_sha256": audit["evaluation_implementation_lock_sha256"],
        "external_bundle_seal": audit["external_bundle_seal"],
        "external_bundle_seal_sha256": audit["external_bundle_seal_sha256"],
        "evaluation_attempt": audit["evaluation_attempt"],
        "evaluation_attempt_sha256": audit["evaluation_attempt_sha256"],
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "results_summary": audit["results_summary"],
        "results_summary_sha256": audit["results_summary_sha256"],
        "claim_boundary": {
            "exact_finite_external_pomdp_transfer": True,
            "smc2_unknown_mechanic_portability": False,
            "generic_pomdp_scalability_or_long_horizon_control": False,
            "formal_safety": False,
            "human_authored_language_robustness": False,
            "model_or_adapter_performance": False,
        },
        "authorization": {
            "continue_to_next_preregistered_stage": True,
            "rerun_v62_candidate_evaluation": False,
            "modify_v62_design_implementation_bundle_or_gates": False,
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
