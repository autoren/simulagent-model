#!/usr/bin/env python3
"""Freeze V10 corpus generation and shortcut auditing before data creation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


IMPLEMENTATION_PATHS = [
    "src/compile-v10.ts",
    "src/v10-contracts.ts",
    "src/v10-grounding.ts",
    "src/v10-validation.ts",
    "src/v9-symbolic.ts",
    "tests/v10-grounding.test.ts",
    "python/v10_protocol.py",
    "python/audit_v10_shortcuts.py",
    "python/test_v10_protocol.py",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/dataset.v10.json")
    plan_path = Path("docs/v10-experiment-plan.md")
    symbolic_lock_path = Path("configs/v9-symbolic-lock.json")
    symbolic_audit_path = Path("outputs/v9-symbolic/oracle-audit.json")
    source_manifest_path = Path("data/v8/manifest.json")
    output_path = Path("configs/v10-grounding-lock.json")
    config = json.loads(config_path.read_text())
    symbolic_audit = json.loads(symbolic_audit_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    if not symbolic_audit["passed"] or symbolic_audit["decision"] != "authorize_v9_grounding_generation":
        raise RuntimeError("V9 symbolic audit does not authorize V10")
    if symbolic_audit["protocol_lock_sha256"] != file_sha256(symbolic_lock_path):
        raise RuntimeError("V9 symbolic audit and lock differ")
    if config["sourceManifest"] != str(source_manifest_path):
        raise RuntimeError("V10 source manifest path differs")
    source_hashes = {}
    for source_text in config["sourceRecords"]:
        source_path = Path(source_text)
        relative = str(source_path.relative_to(source_manifest_path.parent))
        if file_sha256(source_path) != source_manifest["artifact_sha256"][relative]:
            raise RuntimeError(f"V10 source artifact changed: {source_text}")
        source_hashes[source_text] = file_sha256(source_path)
    lock = {
        "schema_version": 10,
        "experiment": "v10_locked_corpus_generation_and_shortcut_audit",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config": config,
        "symbolic_lock": str(symbolic_lock_path),
        "symbolic_lock_sha256": file_sha256(symbolic_lock_path),
        "symbolic_audit": str(symbolic_audit_path),
        "symbolic_audit_sha256": file_sha256(symbolic_audit_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_dataset_sha256": source_manifest["dataset_sha256"],
        "source_artifact_sha256": source_hashes,
        "implementation": {path: file_sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "expected_manifest": "data/v10/manifest.json",
        "limits": {
            "development_corpus_generations_permitted": 1,
            "pre_model_shortcut_audits_permitted": 1,
            "frozen_feature_extractions_permitted": 0,
            "adapter_training_runs_permitted": 0,
            "larger_model_extractions_permitted": 0,
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
        raise RuntimeError(f"Refusing to overwrite changed V10 grounding lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
