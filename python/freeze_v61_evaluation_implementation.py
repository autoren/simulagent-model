#!/usr/bin/env python3
"""Freeze the audited V61 candidate runner."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v61-long-horizon-policy-verification/evaluation-implementation-audit.json"
    )
    parser.add_argument(
        "--bundle-seal", default="configs/v61-verification-bundle-seal.json"
    )
    parser.add_argument(
        "--output", default="configs/v61-evaluation-implementation-lock.json"
    )
    args = parser.parse_args()
    audit_path, seal_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.bundle_seal, args.output)
    )
    if output.exists():
        raise RuntimeError("V61 evaluation implementation already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["verification_bundle_seal_sha256"] != file_sha256(seal_path)
        or audit["manifest_sha256"]
        != file_sha256(PROJECT_ROOT / audit["manifest"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for section in ("evaluation_files_sha256", "frozen_dependencies_sha256")
            for path, digest in audit[section].items()
        )
    ):
        raise RuntimeError("V61 evaluation audit is not intact and bound")
    lock = {
        "schema_version": 61,
        "experiment": "v61_evaluation_implementation_lock",
        "verification_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "verification_bundle_seal_sha256": file_sha256(seal_path),
        "implementation_lock": audit["implementation_lock"],
        "implementation_lock_sha256": audit["implementation_lock_sha256"],
        "manifest": audit["manifest"], "manifest_sha256": audit["manifest_sha256"],
        "evaluation_implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "evaluation_files_sha256": audit["evaluation_files_sha256"],
        "frozen_dependencies_sha256": audit["frozen_dependencies_sha256"],
        "tool_versions": audit["tool_versions"],
        "authorization": {
            "run_one_v61_candidate_verification": True,
            "run_additional_v61_candidate_verification": False,
            "change_evaluation_implementation": False,
            "modify_v61_bundle": False,
            "run_additional_v60_evaluation": False,
            "access_v59_audit_truth": False,
            "formal_safety_claim": False,
            "worst_case_safety_claim": False,
            "parameter_uniform_claim": False,
            "unbounded_temporal_claim": False,
            "human_language_claim": False,
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
