#!/usr/bin/env python3
"""Freeze the audited V62 evaluation implementation."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v62-external-pomdp-transfer/evaluation-implementation-audit.json"
    )
    parser.add_argument("--output", default="configs/v62-evaluation-implementation-lock.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V62 evaluation implementation already frozen")
    audit = json.loads(audit_path.read_text())
    seal_path = PROJECT_ROOT / audit["external_bundle_seal"]
    if (
        not audit["passed"]
        or not all(audit["checks"].values())
        or file_sha256(seal_path) != audit["external_bundle_seal_sha256"]
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for section in ("evaluation_files_sha256", "frozen_dependencies_sha256")
            for path, digest in audit[section].items()
        )
    ):
        raise RuntimeError("V62 evaluation audit is not passing and intact")
    lock = {
        "schema_version": 62,
        "experiment": "v62_evaluation_implementation_lock",
        "external_bundle_seal": audit["external_bundle_seal"],
        "external_bundle_seal_sha256": audit["external_bundle_seal_sha256"],
        "implementation_lock": audit["implementation_lock"],
        "implementation_lock_sha256": audit["implementation_lock_sha256"],
        "manifest": audit["manifest"],
        "manifest_sha256": audit["manifest_sha256"],
        "evaluation_implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "evaluation_files_sha256": audit["evaluation_files_sha256"],
        "frozen_dependencies_sha256": audit["frozen_dependencies_sha256"],
        "runtime_versions": audit["runtime_versions"],
        "authorization": {
            "run_one_v62_candidate_evaluation": True,
            "run_additional_v62_candidate_evaluation": False,
            "change_evaluation_implementation": False,
            "modify_external_bundle": False,
            "network_access_during_candidate_evaluation": False,
            "access_human_v58_records": False,
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
