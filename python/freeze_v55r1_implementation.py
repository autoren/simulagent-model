#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v55r1-design-lock.json")
    parser.add_argument(
        "--audit",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/implementation-audit.json",
    )
    parser.add_argument("--output", default="configs/v55r1-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.design_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V55r1 implementation already frozen")
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
        raise RuntimeError("V55r1 implementation audit is not intact and bound")
    lock = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation_files_sha256": audit["implementation_files_sha256"],
        "base_dependencies_sha256": audit["base_dependencies_sha256"],
        "authorization": {
            "construct_v55r1_population": True,
            "run_v55r1_evaluation": False,
            "change_v55r1_design_or_implementation": False,
            "rerun_v55_population": False,
            "preregister_formal_verification": False,
            "run_formal_verification": False,
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
