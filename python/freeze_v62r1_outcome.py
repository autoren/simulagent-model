#!/usr/bin/env python3
"""Freeze the independently audited V62r1 measurement-repair outcome."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v62r1-terminal-residual-repair/post-result-audit.json"
    )
    parser.add_argument("--output", default="configs/v62r1-outcome-lock.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V62r1 outcome already frozen")
    audit = json.loads(audit_path.read_text())
    bindings = (
        ("result", "result_sha256"),
        ("evaluation_implementation_lock", "evaluation_implementation_lock_sha256"),
        ("implementation_lock", "implementation_lock_sha256"),
        ("source_v62_outcome_lock", "source_v62_outcome_lock_sha256"),
        ("source_v62_result", "source_v62_result_sha256"),
        ("rescore_attempt", "rescore_attempt_sha256"),
        ("results_summary", "results_summary_sha256"),
    )
    if (
        not audit["passed"]
        or not all(audit["gate_checks"].values())
        or not all(audit["binding_checks"].values())
        or not all(audit["reproduction_checks"].values())
        or any(
            file_sha256(PROJECT_ROOT / audit[path]) != audit[digest]
            for path, digest in bindings
        )
    ):
        raise RuntimeError("V62r1 post-result audit is not passing and intact")
    lock = {
        "schema_version": "62r1",
        "experiment": "v62r1_outcome_lock",
        "repair_qualification_passed": True,
        "original_v62_qualification_passed": False,
        "scientific_decision": "accept_v62_external_transfer_as_measurement_repaired",
        "metrics": audit["metrics"],
        "gate_checks": audit["gate_checks"],
        "result": audit["result"],
        "result_sha256": audit["result_sha256"],
        "evaluation_implementation_lock": audit["evaluation_implementation_lock"],
        "evaluation_implementation_lock_sha256": audit[
            "evaluation_implementation_lock_sha256"
        ],
        "implementation_lock": audit["implementation_lock"],
        "implementation_lock_sha256": audit["implementation_lock_sha256"],
        "source_v62_outcome_lock": audit["source_v62_outcome_lock"],
        "source_v62_outcome_lock_sha256": audit["source_v62_outcome_lock_sha256"],
        "source_v62_result": audit["source_v62_result"],
        "source_v62_result_sha256": audit["source_v62_result_sha256"],
        "rescore_attempt": audit["rescore_attempt"],
        "rescore_attempt_sha256": audit["rescore_attempt_sha256"],
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "results_summary": audit["results_summary"],
        "results_summary_sha256": audit["results_summary_sha256"],
        "claim_boundary": {
            "exact_finite_external_pomdp_transfer_on_three_pinned_models": True,
            "measurement_repair_not_independent_replication": True,
            "smc2_unknown_mechanic_portability": False,
            "generic_pomdp_scalability_or_long_horizon_control": False,
            "formal_safety": False,
            "human_authored_language_robustness": False,
            "model_or_adapter_performance": False
        },
        "authorization": {
            "continue_to_next_preregistered_stage": True,
            "rerun_v62_or_v62r1": False,
            "modify_v62_or_v62r1_artifacts": False,
            "treat_original_v62_as_passing": False,
            "access_human_v58_records": False,
            "simulate_human_records": False,
            "model_access": False
        }
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
