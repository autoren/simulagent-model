#!/usr/bin/env python3
"""Freeze the V6 corpus, method, implementation, and gates before feature extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = {
    "compiler": "src/compile-v6.ts",
    "builder": "src/v6.ts",
    "validator": "src/v6-validation.ts",
    "extractor": "python/extract_v6_development_features_mlx.py",
    "trainer": "python/train_v6_frozen_probe.py",
    "probe_freezer": "python/freeze_v6_probe.py",
    "evaluator": "python/evaluate_v6_mechanic_holdout_mlx.py",
    "summarizer": "python/summarize_v6.py",
    "binary_metrics": "python/binary_metrics.py",
    "probe_metrics": "python/train_frozen_linear_probe.py",
    "paired_metrics": "python/evaluate_v5_challenge_mlx.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v6.json")
    parser.add_argument("--manifest", default="data/v6/manifest.json")
    parser.add_argument("--reference", default="outputs/v5-challenge/frozen-probe/result.json")
    parser.add_argument("--output", default="configs/v6-protocol-lock.json")
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
    reference_path = Path(args.reference)
    output_path = Path(args.output)
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads(reference_path.read_text())
    if manifest["schema_version"] != 6 or manifest["validation"]["errors"]:
        raise RuntimeError("V6 manifest is not a clean schema-6 corpus")
    if manifest["source_test_records_read"] != 0:
        raise RuntimeError("V6 manifest reports test access")
    if config != manifest["config"]:
        raise RuntimeError("V6 config does not match the generated manifest")
    if reference["challenge_evaluation_number"] != 1 or reference["test_records_read"] != 0:
        raise RuntimeError("V5 reference does not have the expected firewall")
    expected_reference = config["protocol"]["referenceV5ChallengeBalancedAccuracy"]
    if reference["canonical"]["balanced_accuracy"] != expected_reference:
        raise RuntimeError("V5 reference metric changed")
    for relative_path, expected in manifest["artifact_sha256"].items():
        observed = file_sha256(Path("data/v6") / relative_path)
        if observed != expected:
            raise RuntimeError(f"V6 artifact hash mismatch for {relative_path}")
    implementation = {
        name: {"path": path, "sha256": file_sha256(Path(path))}
        for name, path in IMPLEMENTATION_PATHS.items()
    }
    lock = {
        "schema_version": 6,
        "experiment": "v6_shortcut_resistant_frozen_probe",
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_artifact_sha256": manifest["artifact_sha256"],
        "source_commit": manifest["source_commit"],
        "source_simulation_sha256": manifest["source_simulation_sha256"],
        "source_worktree_diff_sha256": manifest["source_worktree_diff_sha256"],
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
            "one_shot_evaluation": "mechanic_holdout/mirrorreject",
        },
        "gates": config["protocol"]["gates"],
        "reference": {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
            "v5_challenge_balanced_accuracy": expected_reference,
        },
        "implementation": implementation,
        "limits": {
            "development_feature_extractions_permitted": 1,
            "probe_trainings_permitted": 1,
            "mechanic_holdout_evaluations_permitted": 1,
        },
        "data_access": {
            "v3_test_records_read": 0,
            "mechanic_holdout_records_read_before_probe_lock": 0,
        },
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed protocol lock: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
