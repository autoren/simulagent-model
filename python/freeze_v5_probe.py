#!/usr/bin/env python3
"""Freeze and hash the exact V5 probe before challenge-holdout construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from extract_frozen_qwen_features_mlx import SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="outputs/v5-frozen-probe/qwen35-0.8b/full/probe/seed-0/result.json",
    )
    parser.add_argument(
        "--probe",
        default="outputs/v5-frozen-probe/qwen35-0.8b/full/probe/seed-0/selected-probe.npz",
    )
    parser.add_argument(
        "--feature-metadata",
        default="outputs/v5-frozen-probe/qwen35-0.8b/full/features/metadata.json",
    )
    parser.add_argument("--dataset-manifest", default="data/v4/manifest.json")
    parser.add_argument("--output", default="configs/v5-frozen-probe-lock.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    result_path = Path(args.result)
    probe_path = Path(args.probe)
    metadata_path = Path(args.feature_metadata)
    dataset_manifest_path = Path(args.dataset_manifest)
    result = read_json(result_path)
    metadata = read_json(metadata_path)
    dataset_manifest = read_json(dataset_manifest_path)
    if result["test_records_read"] != 0 or metadata["test_records_read"] != 0:
        raise RuntimeError("Cannot freeze a probe that read test records")
    expected_numerics = {
        "feature_dtype": "float32",
        "coefficient_dtype": "float32",
        "decision_score_dtype": "float32",
    }
    if result["numerics"] != expected_numerics:
        raise RuntimeError(f"Unexpected probe numerics: {result['numerics']}")
    if result["input_variant"] != "full" or metadata["input_variant"] != "full":
        raise RuntimeError("The challenge lock must use the preregistered full-input probe")
    if result["selected"]["feature"] != "layer_06_mean":
        raise RuntimeError(f"Unexpected selected feature: {result['selected']['feature']}")
    lock = {
        "schema_version": 1,
        "experiment": "v5_frozen_probe_challenge_lock",
        "model": result["model"],
        "seed": result["seed"],
        "input_variant": result["input_variant"],
        "feature": result["selected"]["feature"],
        "c_value": result["selected"]["c_value"],
        "threshold": result["selected"]["threshold"],
        "max_seq_length": 1024,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "numerics": expected_numerics,
        "probe_artifact": str(probe_path),
        "probe_artifact_sha256": file_sha256(probe_path),
        "source_result": str(result_path),
        "source_result_sha256": file_sha256(result_path),
        "source_feature_metadata": str(metadata_path),
        "source_feature_metadata_sha256": file_sha256(metadata_path),
        "source_dataset_manifest": str(dataset_manifest_path),
        "source_dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "source_dataset_sha256": dataset_manifest["dataset_sha256"],
        "selection_split": result["selection_split"],
        "prior_evaluation_split": result["evaluation_split"],
        "challenge_evaluations_permitted": 1,
        "test_records_read": 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
