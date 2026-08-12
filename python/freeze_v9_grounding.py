#!/usr/bin/env python3
"""Freeze V9 natural-language generation and pre-model auditing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = [
    "src/compile-v9.ts",
    "src/v9-grounding.ts",
    "src/v9-validation.ts",
    "src/v9-symbolic.ts",
    "src/v9-contracts.ts",
    "tests/v9-grounding.test.ts",
    "python/audit_v9_grounding_shortcuts.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/dataset.v9.json")
    plan_path = Path("docs/v9-experiment-plan.md")
    symbolic_lock_path = Path("configs/v9-symbolic-lock.json")
    symbolic_audit_path = Path("outputs/v9-symbolic/oracle-audit.json")
    source_lock_path = Path("configs/v8-development-lock.json")
    output_path = Path("configs/v9-grounding-lock.json")
    config = json.loads(config_path.read_text())
    symbolic_lock = json.loads(symbolic_lock_path.read_text())
    symbolic_audit = json.loads(symbolic_audit_path.read_text())
    source_lock = json.loads(source_lock_path.read_text())
    if symbolic_audit["protocol_lock_sha256"] != sha256(symbolic_lock_path):
        raise RuntimeError("V9 symbolic audit does not share its lock")
    if not symbolic_audit["passed"] or symbolic_audit["records"] != 6480:
        raise RuntimeError("V9 symbolic audit did not authorize grounding")
    if symbolic_lock["source"]["dataset_sha256"] != source_lock["dataset_sha256"]:
        raise RuntimeError("V9 symbolic and V8 source locks differ")
    source_manifest_path = Path(config["sourceManifest"])
    if sha256(source_manifest_path) != source_lock["dataset_manifest_sha256"]:
        raise RuntimeError("V9 source manifest changed")
    for source_text in config["sourceRecords"]:
        source_path = Path(source_text)
        relative = str(source_path.relative_to(source_manifest_path.parent))
        if sha256(source_path) != source_lock["dataset_artifact_sha256"][relative]:
            raise RuntimeError(f"V9 source artifact changed: {source_path}")
    lock = {
        "schema_version": 9,
        "experiment": "v9_locked_grounding_generation_and_audit",
        "preregistration": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "config": config,
        "symbolic_lock": str(symbolic_lock_path),
        "symbolic_lock_sha256": sha256(symbolic_lock_path),
        "symbolic_audit": str(symbolic_audit_path),
        "symbolic_audit_sha256": sha256(symbolic_audit_path),
        "source_dataset_sha256": source_lock["dataset_sha256"],
        "source_manifest_sha256": source_lock["dataset_manifest_sha256"],
        "implementation": {path: sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "expected_manifest": "data/v9/manifest.json",
        "limits": {
            "development_corpus_generations_permitted": 1,
            "pre_model_shortcut_audits_permitted": 1,
            "model_feature_extractions_permitted": 0,
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
        raise RuntimeError(f"Refusing to overwrite changed V9 grounding lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
