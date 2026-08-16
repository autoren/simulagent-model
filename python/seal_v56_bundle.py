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
        "--bundle", default="data/v56-symbolic-probabilistic-policy-verification"
    )
    parser.add_argument(
        "--audit",
        default="outputs/v56-symbolic-probabilistic-policy-verification/bundle-audit.json",
    )
    parser.add_argument(
        "--implementation-lock", default="configs/v56-implementation-lock.json"
    )
    parser.add_argument(
        "--output", default="configs/v56-verification-bundle-seal.json"
    )
    args = parser.parse_args()
    bundle, audit_path, implementation_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (
            args.bundle, args.audit, args.implementation_lock, args.output
        )
    )
    if output.exists():
        raise RuntimeError("V56 verification bundle already sealed")
    audit = json.loads(audit_path.read_text())
    manifest_path = bundle / "manifest.json"
    if (
        not audit["passed"]
        or audit["manifest_sha256"] != file_sha256(manifest_path)
        or audit["implementation_lock_sha256"] != file_sha256(implementation_path)
    ):
        raise RuntimeError("V56 bundle audit is not intact and bound")
    manifest = json.loads(manifest_path.read_text())
    files = []
    for policy in manifest["policies"]:
        for row in policy["files"]:
            path = bundle / row["path"]
            if file_sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise RuntimeError("V56 bundle file changed after audit")
            files.append(row)
    content_payload = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    seal = {
        "schema_version": 56,
        "experiment": "v56_verification_bundle_seal",
        "bundle": str(bundle.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "bundle_content_payload_sha256": content_payload,
        "policy_count": manifest["policy_count"],
        "cohort_counts": manifest["cohort_counts"],
        "sealed_files": len(files),
        "bundle_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "bundle_audit_sha256": file_sha256(audit_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "authorization": {
            "modify_v56_bundle": False,
            "write_and_audit_v56_candidate_runner": True,
            "run_v56_candidate_formal_verification": False,
            "formal_safety_claim": False,
            "language_grounding": False,
            "model_access": False,
        },
    }
    seal["seal_payload_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
