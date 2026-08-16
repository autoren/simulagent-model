#!/usr/bin/env python3
"""Freeze the audited failed V62 result without upgrading its claim."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default="outputs/v62-external-pomdp-transfer/failed-outcome-audit.json",
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
        or audit["failed_checks"] != ["independent_bellman_residual"]
        or any(
            file_sha256(PROJECT_ROOT / audit[path]) != audit[digest]
            for path, digest in bindings
        )
    ):
        raise RuntimeError("V62 failed outcome is not intact and isolated")
    lock = {
        "schema_version": "62-failed",
        "experiment": "v62_outcome_lock",
        "scientific_decision": "preregister_terminal_residual_measurement_repair_only",
        "qualification_passed": False,
        "metrics": audit["metrics"],
        "gate_checks": audit["gate_checks"],
        "failed_checks": audit["failed_checks"],
        "failed_residual_rows": audit["failed_residual_rows"],
        "result": audit["result"],
        "result_sha256": audit["result_sha256"],
        "evaluation_implementation_lock": audit["evaluation_implementation_lock"],
        "evaluation_implementation_lock_sha256": audit[
            "evaluation_implementation_lock_sha256"
        ],
        "external_bundle_seal": audit["external_bundle_seal"],
        "external_bundle_seal_sha256": audit["external_bundle_seal_sha256"],
        "evaluation_attempt": audit["evaluation_attempt"],
        "evaluation_attempt_sha256": audit["evaluation_attempt_sha256"],
        "failed_outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "failed_outcome_audit_sha256": file_sha256(audit_path),
        "results_summary": audit["results_summary"],
        "results_summary_sha256": audit["results_summary_sha256"],
        "authorization": {
            "preregister_terminal_residual_measurement_repair": True,
            "rerun_v62_candidate_evaluation": False,
            "rerun_external_rollouts": False,
            "modify_v62_design_implementation_bundle_result_or_gates": False,
            "treat_v62_as_passing": False,
            "access_human_v58_records": False,
            "simulate_human_records": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
