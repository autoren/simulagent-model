#!/usr/bin/env python3
"""Freeze the V9r2 frozen-grounding extraction and 13-fold evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = [
    "python/extract_v9_grounding_features_mlx.py",
    "python/evaluate_v9_frozen_grounding.py",
    "python/v9_symbolic.py",
    "python/test_v9_symbolic.py",
    "python/test_v9_frozen_grounding.py",
    "python/extract_v6_development_features_mlx.py",
    "python/audit_v9_grounding_shortcuts.py",
    "python/binary_metrics.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    dataset_lock_path = Path("configs/v9r2-grounding-lock.json")
    manifest_path = Path("data/v9r2/manifest.json")
    audit_path = Path("outputs/v9r2-pre-model/shortcut-audit.json")
    output_path = Path("configs/v9-frozen-grounding-lock.json")
    dataset_lock = json.loads(dataset_lock_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest["grounding_lock_sha256"] != sha256(dataset_lock_path):
        raise RuntimeError("V9r2 manifest does not share its grounding lock")
    if audit["grounding_lock_sha256"] != sha256(dataset_lock_path):
        raise RuntimeError("V9r2 shortcut audit does not share the grounding lock")
    if not audit["gates"]["passed"] or audit["decision"] != "authorize_frozen_grounding":
        raise RuntimeError("V9r2 shortcut gates did not authorize model access")
    if audit["dataset_sha256"] != manifest["dataset_sha256"]:
        raise RuntimeError("V9r2 audit and manifest dataset identities differ")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha256(manifest_path.parent / relative) != expected:
            raise RuntimeError(f"V9r2 artifact changed: {relative}")
    protocol = dataset_lock["config"]["protocol"]
    if (
        protocol["model"] != "mlx-community/Qwen3.5-0.8B-4bit"
        or protocol["feature"] != "layer_06_mean"
        or protocol["cValue"] != 1.0
        or protocol["seed"] != 0
    ):
        raise RuntimeError("V9 frozen grounding protocol differs from preregistration")
    lock = {
        "schema_version": 9,
        "revision": 2,
        "experiment": "v9_locked_frozen_neuro_symbolic_grounding",
        "preregistration": {
            "primary": {
                "path": "docs/v9-experiment-plan.md",
                "sha256": sha256(Path("docs/v9-experiment-plan.md")),
            },
            "amendment": {
                "path": "docs/v9r2-experiment-plan.md",
                "sha256": sha256(Path("docs/v9r2-experiment-plan.md")),
            },
        },
        "dataset_lock": str(dataset_lock_path),
        "dataset_lock_sha256": sha256(dataset_lock_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_artifact_sha256": manifest["artifact_sha256"],
        "pre_model_audit": {
            "path": str(audit_path),
            "sha256": sha256(audit_path),
            "gates_passed": True,
        },
        "protocol": protocol,
        "folds": audit["folds"],
        "implementation": {path: sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "limits": {
            "frozen_feature_extractions_permitted": 1,
            "complete_13_fold_evaluations_permitted": 1,
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
        raise RuntimeError(f"Refusing to overwrite changed V9 frozen-grounding lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
