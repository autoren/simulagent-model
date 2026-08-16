#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v55-implementation-lock.json"
    )
    parser.add_argument(
        "--audit",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/evaluation-implementation-audit.json",
    )
    parser.add_argument(
        "--output", default="configs/v55-evaluation-implementation-lock.json"
    )
    args = parser.parse_args()
    implementation_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.implementation_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V55 evaluation implementation already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["implementation_lock_sha256"] != file_sha256(implementation_path)
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for section in ("implementation_files_sha256", "base_dependencies_sha256")
            for path, digest in audit[section].items()
        )
    ):
        raise RuntimeError("V55 evaluation implementation audit is not intact")
    lock = {
        "schema_version": 55,
        "experiment": "v55_evaluation_implementation_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "evaluation_implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "implementation_files_sha256": audit["implementation_files_sha256"],
        "base_dependencies_sha256": audit["base_dependencies_sha256"],
        "authorization": {
            "construct_v55_planning_population": True,
            "run_v55_planning_evaluation": False,
            "change_v55_design_implementation_or_analysis": False,
            "formal_verification": False,
            "language_grounding": False,
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
