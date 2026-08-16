#!/usr/bin/env python3
"""Freeze the audited V62 external benchmark design."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v62-external-pomdp-transfer/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v62-design-lock.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V62 design already frozen")
    audit = json.loads(audit_path.read_text())
    config_path = PROJECT_ROOT / audit["config"]
    plan_path = PROJECT_ROOT / audit["preregistration"]
    requirements_path = PROJECT_ROOT / audit["runtime_requirements"]
    source_path = PROJECT_ROOT / audit["source_v61_outcome_lock"]
    if (
        not audit["passed"]
        or not all(audit["checks"].values())
        or file_sha256(config_path) != audit["config_sha256"]
        or file_sha256(plan_path) != audit["preregistration_sha256"]
        or file_sha256(requirements_path) != audit["runtime_requirements_sha256"]
        or file_sha256(source_path) != audit["source_v61_outcome_lock_sha256"]
    ):
        raise RuntimeError("V62 design audit is not passing and intact")
    config = json.loads(config_path.read_text())
    lock = {
        "schema_version": 62,
        "experiment": "v62_design_lock",
        "config": audit["config"],
        "config_sha256": audit["config_sha256"],
        "config_payload": config,
        "preregistration": audit["preregistration"],
        "preregistration_sha256": audit["preregistration_sha256"],
        "runtime_requirements": audit["runtime_requirements"],
        "runtime_requirements_sha256": audit["runtime_requirements_sha256"],
        "source_v61_outcome_lock": audit["source_v61_outcome_lock"],
        "source_v61_outcome_lock_sha256": audit["source_v61_outcome_lock_sha256"],
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "external_commit": audit["external_commit"],
        "external_files_sha256": audit["external_files_sha256"],
        "external_license_sha256": audit["external_license_sha256"],
        "authorization": {
            "write_and_audit_v62_implementation": True,
            "build_external_bundle": False,
            "run_candidate_evaluation": False,
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
