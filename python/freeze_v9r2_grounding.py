#!/usr/bin/env python3
"""Freeze the V9r2 context-identifier removal and renewed shortcut audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = [
    "src/compile-v9r2.ts",
    "src/v9r2-grounding.ts",
    "tests/v9r2-grounding.test.ts",
    "python/audit_v9r2_grounding_shortcuts.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/dataset.v9r2.json")
    plan_path = Path("docs/v9r2-experiment-plan.md")
    source_manifest_path = Path("data/v9/manifest.json")
    failed_audit_path = Path("outputs/v9-pre-model/shortcut-audit.json")
    source_lock_path = Path("configs/v9-grounding-lock.json")
    output_path = Path("configs/v9r2-grounding-lock.json")
    config = json.loads(config_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    failed_audit = json.loads(failed_audit_path.read_text())
    source_lock = json.loads(source_lock_path.read_text())
    failed_checks = [check["name"] for check in failed_audit["gates"]["checks"] if not check["passed"]]
    if failed_audit["grounding_lock_sha256"] != sha256(source_lock_path):
        raise RuntimeError("V9 shortcut audit does not share the V9 grounding lock")
    if failed_checks != ["context_code_maximum_fold_balanced_accuracy"]:
        raise RuntimeError(f"V9r2 narrow revision is not authorized by failures: {failed_checks}")
    if not failed_audit["structural_passed"]:
        raise RuntimeError("V9r2 cannot reuse a structurally invalid V9 corpus")
    if source_manifest["grounding_lock_sha256"] != sha256(source_lock_path):
        raise RuntimeError("V9 source corpus does not share its grounding lock")
    for source_text in config["sourceRecords"]:
        source_path = Path(source_text)
        relative = str(source_path.relative_to(source_manifest_path.parent))
        if sha256(source_path) != source_manifest["artifact_sha256"][relative]:
            raise RuntimeError(f"V9r2 source changed: {source_path}")
    if config["protocol"] != source_lock["config"]["protocol"]:
        raise RuntimeError("V9r2 may not change the frozen-model protocol or gates")
    lock = {
        "schema_version": 9,
        "revision": 2,
        "experiment": "v9r2_locked_grounding_generation_and_audit",
        "preregistration": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "config": config,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_dataset_sha256": source_manifest["dataset_sha256"],
        "source_grounding_lock_sha256": sha256(source_lock_path),
        "failed_shortcut_audit": str(failed_audit_path),
        "failed_shortcut_audit_sha256": sha256(failed_audit_path),
        "implementation": {path: sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "expected_manifest": "data/v9r2/manifest.json",
        "limits": {
            "development_corpus_transformations_permitted": 1,
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
        raise RuntimeError(f"Refusing to overwrite changed V9r2 lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
