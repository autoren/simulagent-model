#!/usr/bin/env python3
"""Freeze V8 development data, gates, representation, and LOMO diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = {
    "compiler": "src/compile-v8.ts",
    "builder": "src/v8.ts",
    "validator": "src/v8-validation.ts",
    "contracts": "src/contracts.ts",
    "shortcut_audit": "python/audit_v8_shortcuts.py",
    "extractor": "python/extract_v8_development_features_mlx.py",
    "forward_core": "python/extract_v6_development_features_mlx.py",
    "diagnostic": "python/run_v8_lomo_diagnostics.py",
    "binary_metrics": "python/binary_metrics.py",
    "simulator": "../simulagent/src/simulation.ts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v8.json")
    parser.add_argument("--manifest", default="data/v8/manifest.json")
    parser.add_argument("--shortcut", default="outputs/v8-pre-model/shortcut-audit.json")
    parser.add_argument("--plan", default="docs/v8-experiment-plan.md")
    parser.add_argument("--output", default="configs/v8-development-lock.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    manifest_path = Path(args.manifest)
    shortcut_path = Path(args.shortcut)
    plan_path = Path(args.plan)
    output_path = Path(args.output)
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    shortcut = json.loads(shortcut_path.read_text())
    if manifest["schema_version"] != 8 or manifest["validation"]["errors"]:
        raise RuntimeError("V8 requires a clean schema-8 manifest")
    if manifest["config"] != config:
        raise RuntimeError("V8 config does not match the manifest")
    if shortcut["dataset_sha256"] != manifest["dataset_sha256"] or not shortcut["gates"]["passed"]:
        raise RuntimeError("V8 pre-model gates did not pass for this dataset")
    forbidden = (
        "v3_test_records_read",
        "prior_holdout_records_read",
        "v7_tone_drift_records_read",
        "v7_model_results_read",
    )
    if any(manifest["data_access"][key] != 0 for key in forbidden):
        raise RuntimeError("V8 manifest crossed the data firewall")
    if any(shortcut["data_access"][key] != 0 for key in forbidden):
        raise RuntimeError("V8 shortcut audit crossed the data firewall")
    dataset_root = manifest_path.parent
    for relative, expected in manifest["artifact_sha256"].items():
        if file_sha256(dataset_root / relative) != expected:
            raise RuntimeError(f"V8 artifact hash mismatch: {relative}")
    implementation = {
        name: {"path": path, "sha256": file_sha256(Path(path))}
        for name, path in IMPLEMENTATION_PATHS.items()
    }
    lock = {
        "schema_version": 8,
        "experiment": "v8_locked_development_lomo",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_artifact_sha256": manifest["artifact_sha256"],
        "source_simulation_sha256": manifest["source_simulation_sha256"],
        "shortcut_audit": str(shortcut_path),
        "shortcut_audit_sha256": file_sha256(shortcut_path),
        "pre_model_gates": shortcut["gates"],
        "reported_relational_character_baseline": shortcut["audits"]["relational_character_ngram"],
        "mechanics": config["mechanics"],
        "surfaces": config["surfaceVariants"],
        "method": {
            "model": config["protocol"]["model"],
            "adapter_path": None,
            "frozen": True,
            "feature": config["protocol"]["feature"],
            "layer": 6,
            "pooling": "mean",
            "feature_dtype": "float32",
            "c_value": config["protocol"]["cValue"],
            "seed": config["protocol"]["seed"],
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_seq_length": config["protocol"]["maxSeqLength"],
            "pair_training": "symmetric_signed_differences",
            "threshold_selection": "other_mechanics_canonical_calibration_only",
        },
        "gates": config["protocol"]["gates"],
        "implementation": implementation,
        "limits": {
            "development_feature_extractions_permitted": 1,
            "lomo_diagnostic_runs_permitted": 1,
            "untouched_v8_mechanic_evaluations_permitted": 0,
        },
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
        },
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V8 lock: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
