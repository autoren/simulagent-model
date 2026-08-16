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
        "--manifest",
        default="data/v55-short-horizon-bayes-adaptive-planning/manifest.json",
    )
    parser.add_argument(
        "--audit",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/population-audit.json",
    )
    parser.add_argument("--output", default="configs/v55-population-seal.json")
    args = parser.parse_args()
    manifest_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.manifest, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V55 population already sealed")
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    artifact = PROJECT_ROOT / manifest["file"]["path"]
    if (
        not audit["passed"]
        or audit["manifest_sha256"] != file_sha256(manifest_path)
        or file_sha256(artifact) != manifest["file"]["sha256"]
    ):
        raise RuntimeError("V55 population audit or manifest is not intact")
    seal = {
        "schema_version": 55,
        "experiment": "v55_population_seal",
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "population_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "population_audit_sha256": file_sha256(audit_path),
        "implementation_lock": audit["implementation_lock"],
        "implementation_lock_sha256": audit["implementation_lock_sha256"],
        "evaluation_implementation_lock": audit["evaluation_implementation_lock"],
        "evaluation_implementation_lock_sha256": audit[
            "evaluation_implementation_lock_sha256"
        ],
        "population": manifest["file"],
        "population_hash": manifest["population_hash"],
        "authorization": {
            "run_v55_planning_evaluation_once": True,
            "change_v55_design_implementation_or_population": False,
            "formal_verification": False,
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
