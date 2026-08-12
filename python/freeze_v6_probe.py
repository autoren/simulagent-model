#!/usr/bin/env python3
"""Hash-lock the trained V6 probe and every development artifact before holdout scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", default="configs/v6-protocol-lock.json")
    parser.add_argument("--features", default="outputs/v6-frozen-probe/features")
    parser.add_argument("--probe-dir", default="outputs/v6-frozen-probe/probe")
    parser.add_argument("--output", default="configs/v6-frozen-probe-lock.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol_lock)
    feature_root = Path(args.features)
    probe_root = Path(args.probe_dir)
    output_path = Path(args.output)
    protocol = json.loads(protocol_path.read_text())
    metadata_path = feature_root / "metadata.json"
    train_features = feature_root / "train.npz"
    calibration_features = feature_root / "calibration.npz"
    probe_path = probe_root / "selected-probe.npz"
    result_path = probe_root / "result.json"
    metadata = json.loads(metadata_path.read_text())
    result = json.loads(result_path.read_text())
    if metadata["mechanic_holdout_records_read"] != 0 or result["mechanic_holdout_records_read"] != 0:
        raise RuntimeError("Mechanic holdout was accessed before probe lock")
    if metadata["v3_test_records_read"] != 0 or result["v3_test_records_read"] != 0:
        raise RuntimeError("V3 test was accessed during V6 development")
    protocol_hash = file_sha256(protocol_path)
    if metadata["protocol_lock_sha256"] != protocol_hash or result["protocol_lock_sha256"] != protocol_hash:
        raise RuntimeError("Development artifacts do not share the active protocol lock")
    if result["probe_training_number"] != 1:
        raise RuntimeError("V6 protocol permits exactly one probe training")
    expected_hashes = {
        "feature_metadata": (metadata_path, result["feature_metadata_sha256"]),
        "train_features": (train_features, result["feature_artifact_sha256"]["train"]),
        "calibration_features": (
            calibration_features,
            result["feature_artifact_sha256"]["calibration"],
        ),
        "probe": (probe_path, result["probe_artifact_sha256"]),
    }
    for name, (path, expected) in expected_hashes.items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"V6 {name} changed before probe lock")
    lock = {
        "schema_version": 6,
        "experiment": "v6_locked_mechanic_transfer",
        "protocol_lock": str(protocol_path),
        "protocol_lock_sha256": protocol_hash,
        "dataset_manifest": protocol["dataset_manifest"],
        "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
        "dataset_sha256": protocol["dataset_sha256"],
        "holdout_records": "data/v6/records/mechanic_holdout.jsonl",
        "holdout_records_sha256": protocol["dataset_artifact_sha256"][
            "records/mechanic_holdout.jsonl"
        ],
        "method": protocol["method"],
        "gates": protocol["gates"],
        "reference": protocol["reference"],
        "feature_metadata": str(metadata_path),
        "feature_metadata_sha256": file_sha256(metadata_path),
        "train_features": str(train_features),
        "train_features_sha256": file_sha256(train_features),
        "calibration_features": str(calibration_features),
        "calibration_features_sha256": file_sha256(calibration_features),
        "training_result": str(result_path),
        "training_result_sha256": file_sha256(result_path),
        "probe_artifact": str(probe_path),
        "probe_artifact_sha256": file_sha256(probe_path),
        "threshold": result["threshold"],
        "calibration_gate": result["calibration_gate"],
        "implementation": protocol["implementation"],
        "mechanic_holdout_evaluations_permitted": 1,
        "probe_training_number": 1,
        "mechanic_holdout_records_read_before_lock": 0,
        "v3_test_records_read": 0,
    }
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed frozen probe lock: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
