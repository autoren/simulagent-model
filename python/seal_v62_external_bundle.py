#!/usr/bin/env python3
"""Seal the independently audited V62 external source bundle."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v62-external-pomdp-transfer/bundle-audit.json"
    )
    parser.add_argument("--output", default="configs/v62-external-bundle-seal.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V62 external bundle already sealed")
    audit = json.loads(audit_path.read_text())
    bundle = PROJECT_ROOT / audit["bundle"]
    manifest_path = PROJECT_ROOT / audit["manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        not audit["passed"]
        or not all(audit["checks"].values())
        or file_sha256(manifest_path) != audit["manifest_sha256"]
        or any(
            file_sha256(bundle / relative) != binding["sha256"]
            for relative, binding in manifest["files"].items()
        )
    ):
        raise RuntimeError("V62 external bundle audit is not passing and intact")
    content_hash = hashlib.sha256(
        json.dumps(
            {path: binding["sha256"] for path, binding in manifest["files"].items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    seal = {
        "schema_version": 62,
        "experiment": "v62_external_bundle_seal",
        "bundle": audit["bundle"],
        "bundle_content_sha256": content_hash,
        "manifest": audit["manifest"],
        "manifest_sha256": audit["manifest_sha256"],
        "bundle_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "bundle_audit_sha256": file_sha256(audit_path),
        "implementation_lock": audit["implementation_lock"],
        "implementation_lock_sha256": audit["implementation_lock_sha256"],
        "source_files_sha256": audit["source_files_sha256"],
        "license_sha256": audit["license_sha256"],
        "independent_parser_agreement_rate": audit["independent_parser_agreement_rate"],
        "maximum_array_errors": audit["maximum_array_errors"],
        "authorization": {
            "write_and_audit_evaluation_implementation": True,
            "run_one_candidate_evaluation": False,
            "modify_external_bundle": False,
            "network_access_during_candidate_evaluation": False,
            "access_human_v58_records": False,
            "model_access": False,
        },
    }
    seal["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
