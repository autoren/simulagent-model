#!/usr/bin/env python3
"""Freeze the audited V60 belief-to-planner implementation."""
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
        default="outputs/v60-approximate-belief-decision-calibration/implementation-audit.json",
    )
    parser.add_argument("--design-lock", default="configs/v60-design-lock.json")
    parser.add_argument("--output", default="configs/v60-implementation-lock.json")
    args = parser.parse_args()
    audit_path, design_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.design_lock, args.output)
    )
    if output.exists():
        raise RuntimeError("V60 implementation is already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["design_lock_sha256"] != file_sha256(design_path)
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for section in ("implementation_files_sha256", "base_dependencies_sha256")
            for path, digest in audit[section].items()
        )
    ):
        raise RuntimeError("V60 implementation audit is not intact and bound")
    lock = {
        "schema_version": 60,
        "experiment": "v60_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation_files_sha256": audit["implementation_files_sha256"],
        "base_dependencies_sha256": audit["base_dependencies_sha256"],
        "authorization": {
            "write_and_audit_v60_evaluator": True,
            "run_v60_evaluation": False,
            "change_v60_design_or_implementation": False,
            "access_v59_audit_truth": False,
            "construct_v60_population": False,
            "collect_human_language": False,
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
