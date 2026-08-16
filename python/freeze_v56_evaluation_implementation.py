#!/usr/bin/env python3
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
        default="outputs/v56-symbolic-probabilistic-policy-verification/evaluation-implementation-audit.json",
    )
    parser.add_argument(
        "--bundle-seal", default="configs/v56-verification-bundle-seal.json"
    )
    parser.add_argument(
        "--output", default="configs/v56-evaluation-implementation-lock.json"
    )
    args = parser.parse_args()
    audit_path, seal_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.bundle_seal, args.output)
    )
    if output.exists():
        raise RuntimeError("V56 evaluation implementation already frozen")
    audit = json.loads(audit_path.read_text())
    seal = json.loads(seal_path.read_text())
    if (
        not audit["passed"]
        or audit["verification_bundle_seal_sha256"] != file_sha256(seal_path)
        or audit["manifest_sha256"]
        != file_sha256(PROJECT_ROOT / seal["manifest"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for section in ("evaluation_files_sha256", "frozen_dependencies_sha256")
            for path, digest in audit[section].items()
        )
    ):
        raise RuntimeError("V56 evaluation audit is not intact and bound")
    lock = {
        "schema_version": 56,
        "experiment": "v56_evaluation_implementation_lock",
        "verification_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "verification_bundle_seal_sha256": file_sha256(seal_path),
        "implementation_lock": audit["implementation_lock"],
        "implementation_lock_sha256": audit["implementation_lock_sha256"],
        "manifest": audit["manifest"],
        "manifest_sha256": audit["manifest_sha256"],
        "evaluation_implementation_audit": str(
            audit_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "evaluation_files_sha256": audit["evaluation_files_sha256"],
        "frozen_dependencies_sha256": audit["frozen_dependencies_sha256"],
        "tool_versions": audit["tool_versions"],
        "authorization": {
            "run_one_v56_candidate_verification": True,
            "run_additional_v56_candidate_verification": False,
            "change_evaluation_implementation": False,
            "modify_v56_bundle": False,
            "run_additional_v55_or_v55r1_evaluation": False,
            "formal_safety_claim": False,
            "worst_case_safety_claim": False,
            "parameter_uniform_claim": False,
            "long_horizon_claim": False,
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
