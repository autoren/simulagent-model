#!/usr/bin/env python3
"""Freeze the audited V60 design."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v60-approximate-belief-decision-calibration/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v60-design-lock.json")
    args = parser.parse_args()
    audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V60 design is already frozen")
    audit = json.loads(audit_path.read_text())
    config_path = PROJECT_ROOT / audit["config"]
    plan_path = PROJECT_ROOT / audit["preregistration"]
    if (
        not audit["passed"]
        or audit["config_sha256"] != file_sha256(config_path)
        or audit["preregistration_sha256"] != file_sha256(plan_path)
        or audit["population_seal_sha256"]
        != file_sha256(PROJECT_ROOT / audit["population_seal"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in audit["source_outcome_locks_sha256"].items()
        )
    ):
        raise RuntimeError("V60 design audit is not intact and bound")
    lock = {
        "schema_version": 60,
        "experiment": "v60_design_lock",
        "config": audit["config"],
        "config_sha256": audit["config_sha256"],
        "config_payload": json.loads(config_path.read_text()),
        "preregistration": audit["preregistration"],
        "preregistration_sha256": audit["preregistration_sha256"],
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_outcome_locks_sha256": audit["source_outcome_locks_sha256"],
        "population_seal": audit["population_seal"],
        "population_seal_sha256": audit["population_seal_sha256"],
        "authorization": {
            "write_and_audit_v60_implementation": True,
            "run_v60_evaluation": False,
            "construct_v60_population": False,
            "access_v59_audit_truth": False,
            "simulate_human_v58_records": False,
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
