#!/usr/bin/env python3
"""Freeze the audited V62 candidate implementation."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v62-external-pomdp-transfer/implementation-audit.json"
    )
    parser.add_argument("--output", default="configs/v62-implementation-lock.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V62 implementation already frozen")
    audit = json.loads(audit_path.read_text())
    design_path = PROJECT_ROOT / audit["design_lock"]
    if (
        not audit["passed"]
        or not all(audit["checks"].values())
        or file_sha256(design_path) != audit["design_lock_sha256"]
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in audit["implementation_files_sha256"].items()
        )
    ):
        raise RuntimeError("V62 implementation audit is not passing and intact")
    lock = {
        "schema_version": 62,
        "experiment": "v62_implementation_lock",
        "design_lock": audit["design_lock"],
        "design_lock_sha256": audit["design_lock_sha256"],
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation_files_sha256": audit["implementation_files_sha256"],
        "runtime_versions": audit["runtime_versions"],
        "independent_parser_max_errors": audit["independent_parser_max_errors"],
        "independent_planner_max_value_error": audit["independent_planner_max_value_error"],
        "independent_planner_action_agreement_rate": audit["independent_planner_action_agreement_rate"],
        "analytic_fixture_pass_rate": audit["analytic_fixture_pass_rate"],
        "mutation_kill_rate": audit["mutation_kill_rate"],
        "authorization": {
            "build_and_audit_one_external_source_bundle": True,
            "run_candidate_evaluation": False,
            "change_candidate_implementation": False,
            "change_tasks_horizons_metrics_gates_or_controls": False,
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
