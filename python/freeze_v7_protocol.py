#!/usr/bin/env python3
"""Freeze the V7 corpus, pre-model gates, method, implementation, and decision rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = {
    "compiler": "src/compile-v7.ts",
    "builder": "src/v7.ts",
    "validator": "src/v7-validation.ts",
    "shortcut_audit": "python/audit_v7_shortcuts.py",
    "extractor": "python/extract_v7_development_features_mlx.py",
    "extractor_core": "python/extract_v6_development_features_mlx.py",
    "trainer": "python/train_v7_frozen_probe.py",
    "probe_freezer": "python/freeze_v7_probe.py",
    "evaluator": "python/evaluate_v7_untouched_mlx.py",
    "summarizer": "python/summarize_v7.py",
    "v7_metrics": "python/v7_metrics.py",
    "binary_metrics": "python/binary_metrics.py",
    "probe_core": "python/train_frozen_linear_probe.py",
    "paired_metrics": "python/evaluate_v5_challenge_mlx.py",
    "simulator": "../simulagent/src/simulation.ts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v7.json")
    parser.add_argument("--manifest", default="data/v7/manifest.json")
    parser.add_argument("--shortcut", default="outputs/v7-pre-model/shortcut-audit.json")
    parser.add_argument("--plan", default="docs/v7-experiment-plan.md")
    parser.add_argument("--output", default="configs/v7-protocol-lock.json")
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
    if manifest["schema_version"] != 7 or manifest["validation"]["errors"]:
        raise RuntimeError("V7 manifest is not a clean schema-7 corpus")
    if config != manifest["config"]:
        raise RuntimeError("V7 config does not match the generated manifest")
    if any(manifest["data_access"][key] != 0 for key in (
        "v3_test_records_read",
        "prior_holdout_records_read",
        "prior_model_results_read",
        "untouched_mechanic_model_scores_read",
    )):
        raise RuntimeError("V7 manifest reports forbidden data access")
    if shortcut["dataset_sha256"] != manifest["dataset_sha256"] or not shortcut["gates"]["passed"]:
        raise RuntimeError("V7 shortcut rejection gates did not pass for this dataset")
    if any(shortcut["data_access"][key] != 0 for key in (
        "v3_test_records_read",
        "prior_holdout_records_read",
        "untouched_mechanic_records_read",
        "model_features_read",
    )):
        raise RuntimeError("V7 shortcut audit crossed the data firewall")
    dataset_root = manifest_path.parent
    for relative_path, expected in manifest["artifact_sha256"].items():
        if file_sha256(dataset_root / relative_path) != expected:
            raise RuntimeError(f"V7 artifact hash mismatch for {relative_path}")
    implementation = {
        name: {"path": path, "sha256": file_sha256(Path(path))}
        for name, path in IMPLEMENTATION_PATHS.items()
    }
    lock = {
        "schema_version": 7,
        "experiment": "v7_causal_evidence_frozen_probe",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_artifact_sha256": manifest["artifact_sha256"],
        "source_commit": manifest["source_commit"],
        "source_simulation_sha256": manifest["source_simulation_sha256"],
        "source_worktree_diff_sha256": manifest["source_worktree_diff_sha256"],
        "pre_model_gates": shortcut["gates"],
        "conditional_label_gap": shortcut["conditional_label_gap"],
        "shortcut_audit": str(shortcut_path),
        "shortcut_audit_sha256": file_sha256(shortcut_path),
        "method": {
            "model": config["protocol"]["model"],
            "frozen": True,
            "adapter_path": None,
            "input_variant": "full",
            "feature": config["protocol"]["feature"],
            "pooling": "mean",
            "layer": 6,
            "c_value": config["protocol"]["cValue"],
            "seed": config["protocol"]["seed"],
            "solver": "saga",
            "class_weight": "balanced",
            "surface_supervision": "complete_triplets_with_shared_binary_target",
            "threshold_selection": "canonical_calibration_only",
            "max_seq_length": config["protocol"]["maxSeqLength"],
            "bootstrap_samples": config["protocol"]["bootstrapSamples"],
            "bootstrap_seed": config["protocol"]["bootstrapSeed"],
        },
        "partitions": {
            "training": "train",
            "threshold_calibration": "calibration/canonical",
            "one_shot_evaluation": "untouched_mechanic/tonedrift",
        },
        "gates": config["protocol"]["gates"],
        "implementation": implementation,
        "limits": {
            "development_feature_extractions_permitted": 1,
            "probe_trainings_permitted": 1,
            "untouched_mechanic_evaluations_permitted": 1,
        },
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "untouched_mechanic_records_read_before_probe_lock": 0,
        },
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V7 protocol lock: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
