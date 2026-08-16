#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v59-design-lock.json")
    parser.add_argument(
        "--audit",
        default="outputs/v59-budgeted-root-sampled-planning/implementation-audit.json",
    )
    parser.add_argument("--output", default="configs/v59-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.design_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V59 implementation already frozen")
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
        raise RuntimeError("V59 implementation audit is not intact and bound")
    lock = {
        "schema_version": 59,
        "experiment": "v59_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation_files_sha256": audit["implementation_files_sha256"],
        "base_dependencies_sha256": audit["base_dependencies_sha256"],
        "authorization": {
            "construct_v59_population": True,
            "run_v59_evaluation": False,
            "change_v59_design_or_implementation": False,
            "access_v59_audit_truth_during_candidate_evaluation": False,
            "simulate_human_v58_records": False,
            "formal_safety_claim": False,
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

