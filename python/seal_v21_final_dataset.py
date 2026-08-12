"""Seal materialized V21 data and its pre-extraction audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v21-multimechanic-execution-lock.json")
    parser.add_argument("--manifest", default="data/v21-final/manifest.json")
    parser.add_argument("--audit", default="outputs/v21-final/pre-extraction-audit.json")
    parser.add_argument("--output", default="configs/v21-final-dataset-seal.json")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise RuntimeError(f"V21 dataset seal already exists: {output}")
    if (PROJECT_ROOT / "outputs/v21-final/features").exists():
        raise RuntimeError("V21 features exist before dataset seal")
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V21 locked implementation changed: {path}")
    manifest_path = PROJECT_ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    audit_path = PROJECT_ROOT / args.audit
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_single_v21_feature_extraction":
        raise RuntimeError("V21 pre-extraction audit did not authorize extraction")
    if audit["manifest_sha256"] != file_sha256(manifest_path):
        raise RuntimeError("V21 manifest changed after audit")
    seal = {
        "schema_version": 21,
        "experiment": "v21_final_dataset_seal",
        "execution_lock": args.lock,
        "execution_lock_sha256": file_sha256(lock_path),
        "manifest": args.manifest,
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "pre_extraction_audit": args.audit,
        "pre_extraction_audit_sha256": file_sha256(audit_path),
        "model": lock["model"],
        "implementation": lock["implementation"],
        "limits": lock["limits"],
        "feature_extraction_count_before_seal": 0,
        "evaluation_count_before_seal": 0,
    }
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": args.output,
        "seal_sha256": file_sha256(output),
        "dataset_sha256": seal["dataset_sha256"],
        "feature_extraction_count_before_seal": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
