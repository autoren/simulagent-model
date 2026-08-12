#!/usr/bin/env python3
"""Freeze the V9 deterministic symbolic-oracle audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/v9-symbolic.json")
    plan_path = Path("docs/v9-symbolic-plan.md")
    source_lock_path = Path("configs/v8-development-lock.json")
    output_path = Path("configs/v9-symbolic-lock.json")
    config = json.loads(config_path.read_text())
    source_lock = json.loads(source_lock_path.read_text())
    manifest_path = Path(config["sourceManifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_sha256"] != source_lock["dataset_sha256"]:
        raise RuntimeError("V9 symbolic source does not match the locked V8 dataset")
    artifacts = {}
    for source_path_text in config["sourceSplits"]:
        source_path = Path(source_path_text)
        relative = str(source_path.relative_to(manifest_path.parent))
        expected = source_lock["dataset_artifact_sha256"][relative]
        if sha256(source_path) != expected:
            raise RuntimeError(f"V9 symbolic source artifact changed: {source_path}")
        artifacts[str(source_path)] = expected
    implementation_paths = [
        "src/v9-contracts.ts",
        "src/v9-symbolic.ts",
        "src/audit-v9-symbolic.ts",
        "tests/v9-symbolic.test.ts",
    ]
    lock = {
        "schema_version": 9,
        "experiment": "v9_locked_symbolic_oracle_audit",
        "preregistration": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "config": config,
        "source": {
            "dataset_sha256": source_lock["dataset_sha256"],
            "development_lock": str(source_lock_path),
            "development_lock_sha256": sha256(source_lock_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "artifacts": artifacts,
        },
        "implementation": {
            path: sha256(Path(path)) for path in implementation_paths
        },
        "limits": {
            "symbolic_oracle_audits_permitted": 1,
            "model_extractions_permitted": 0,
            "adapter_training_runs_permitted": 0,
            "final_mechanic_evaluations_permitted": 0,
        },
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
            "final_v9_mechanic_records_read": 0,
        },
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V9 symbolic lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
