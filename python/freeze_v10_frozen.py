#!/usr/bin/env python3
"""Freeze the single V10 0.8B extraction and 24-fold evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


IMPLEMENTATION_PATHS = [
    "python/extract_v10_features_mlx.py",
    "python/evaluate_v10_frozen.py",
    "python/test_v10_frozen.py",
    "python/test_v10_protocol.py",
    "python/v10_protocol.py",
    "python/v9_symbolic.py",
    "python/binary_metrics.py",
    "python/extract_v6_development_features_mlx.py",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    dataset_lock_path = Path("configs/v10-grounding-lock.json")
    manifest_path = Path("data/v10/manifest.json")
    audit_path = Path("outputs/v10-pre-model/shortcut-audit.json")
    output_path = Path("configs/v10-frozen-lock.json")
    dataset_lock = json.loads(dataset_lock_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest["grounding_lock_sha256"] != file_sha256(dataset_lock_path):
        raise RuntimeError("V10 manifest does not share the dataset lock")
    if audit["grounding_lock_sha256"] != file_sha256(dataset_lock_path):
        raise RuntimeError("V10 audit does not share the dataset lock")
    if not audit["gates"]["passed"] or audit["decision"] != "authorize_v10_frozen_extraction":
        raise RuntimeError("V10 pre-model audit did not authorize extraction")
    if audit["dataset_sha256"] != manifest["dataset_sha256"]:
        raise RuntimeError("V10 audit and manifest dataset identities differ")
    for relative, expected in manifest["artifact_sha256"].items():
        if file_sha256(manifest_path.parent / relative) != expected:
            raise RuntimeError(f"V10 artifact changed: {relative}")
    protocol = dataset_lock["config"]["protocol"]
    if (
        protocol["model"] != "mlx-community/Qwen3.5-0.8B-4bit"
        or protocol["layer"] != 6
        or protocol["cValue"] != 1.0
        or protocol["seed"] != 0
        or protocol["primaryRepresentation"] != "nli_final"
        or protocol["diagnosticRepresentations"] != ["mean_direct", "evidence_span_direct"]
    ):
        raise RuntimeError("V10 frozen protocol differs from preregistration")
    lock = {
        "schema_version": 10,
        "experiment": "v10_locked_frozen_current_state_polarity",
        "preregistration": {
            "path": "docs/v10-experiment-plan.md",
            "sha256": file_sha256(Path("docs/v10-experiment-plan.md")),
        },
        "dataset_lock": str(dataset_lock_path),
        "dataset_lock_sha256": file_sha256(dataset_lock_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_artifact_sha256": manifest["artifact_sha256"],
        "pre_model_audit": {
            "path": str(audit_path),
            "sha256": file_sha256(audit_path),
            "gates_passed": True,
        },
        "protocol": protocol,
        "folds": audit["folds"],
        "implementation": {path: file_sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "limits": {
            "frozen_0_8b_feature_extractions_permitted": 1,
            "complete_24_fold_evaluations_permitted": 1,
            "larger_frozen_model_extractions_permitted": 0,
            "adapter_training_runs_permitted": 0,
            "final_mechanic_evaluations_permitted": 0,
        },
        "data_access": dataset_lock["data_access"],
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V10 frozen lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
