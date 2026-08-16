#!/usr/bin/env python3
"""Seal the audited V59 public and audit-truth population artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population", default="data/v59-budgeted-root-sampled-planning"
    )
    parser.add_argument(
        "--audit",
        default="outputs/v59-budgeted-root-sampled-planning/population-audit.json",
    )
    parser.add_argument("--output", default="configs/v59-population-seal.json")
    args = parser.parse_args()
    population, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.population, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V59 population already sealed")
    audit = json.loads(audit_path.read_text())
    manifest_path = population / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    implementation_path = PROJECT_ROOT / manifest["implementation_lock"]
    if (
        not audit["passed"]
        or audit["manifest_sha256"] != file_sha256(manifest_path)
        or audit["implementation_lock_sha256"]
        != file_sha256(implementation_path)
    ):
        raise RuntimeError("V59 population audit is not intact and bound")
    artifacts = {}
    for name in ("public_file", "audit_truth_file"):
        row = manifest[name]
        path = PROJECT_ROOT / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"V59 population changed after audit: {name}")
        artifacts[name] = row
    seal = {
        "schema_version": 59,
        "experiment": "v59_population_seal",
        "population": str(population.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "artifacts": artifacts,
        "public_audit_pairing_sha256": manifest["public_audit_pairing_sha256"],
        "population_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "population_audit_sha256": file_sha256(audit_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "authorization": {
            "modify_v59_population": False,
            "write_and_audit_v59_candidate_runner": True,
            "run_v59_candidate_evaluation": False,
            "candidate_access_v59_audit_truth": False,
            "collect_human_language": False,
            "formal_safety_claim": False,
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
