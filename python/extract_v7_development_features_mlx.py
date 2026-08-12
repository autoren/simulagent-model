#!/usr/bin/env python3
"""Extract locked V7 development features without opening the untouched mechanic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from mlx_lm import load

from extract_v6_development_features_mlx import extract_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v7-protocol-lock.json")
    parser.add_argument("--output-dir", default="outputs/v7-frozen-probe/features")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_protocol(lock_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = json.loads(lock_path.read_text())
    manifest_path = Path(lock["dataset_manifest"])
    if file_sha256(manifest_path) != lock["dataset_manifest_sha256"]:
        raise RuntimeError("V7 dataset manifest changed after protocol lock")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_sha256"] != lock["dataset_sha256"]:
        raise RuntimeError("V7 dataset identity changed after protocol lock")
    implementation = lock["implementation"]["extractor"]
    if file_sha256(Path(implementation["path"])) != implementation["sha256"]:
        raise RuntimeError("V7 extractor changed after protocol lock")
    if not lock["pre_model_gates"]["passed"]:
        raise RuntimeError("V7 pre-model rejection gates did not pass")
    if lock["data_access"]["untouched_mechanic_records_read_before_probe_lock"] != 0:
        raise RuntimeError("Protocol lock reports pre-lock untouched-mechanic access")
    return lock, manifest


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    lock, manifest = verify_protocol(lock_path)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"V7 features already exist; refusing a second extraction: {metadata_path}")
    model, tokenizer = load(lock["method"]["model"])
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "experiment": "v7_development_feature_extraction",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "dataset_manifest_sha256": lock["dataset_manifest_sha256"],
        "dataset_sha256": lock["dataset_sha256"],
        "model": lock["method"]["model"],
        "feature": lock["method"]["feature"],
        "layer": lock["method"]["layer"],
        "pooling": lock["method"]["pooling"],
        "frozen": True,
        "adapter_path": None,
        "splits": {},
        "untouched_mechanic_records_read": 0,
        "prior_holdout_records_read": 0,
        "v3_test_records_read": 0,
    }
    for split in ("train", "calibration"):
        relative_path = f"records/{split}.jsonl"
        path = Path(lock["dataset_manifest"]).parent / relative_path
        if file_sha256(path) != manifest["artifact_sha256"][relative_path]:
            raise RuntimeError(f"V7 {split} records changed after generation")
        records = read_jsonl(path)
        arrays, split_metadata = extract_split(
            model,
            tokenizer,
            records,
            lock["method"]["max_seq_length"],
            args.progress_every,
            split,
        )
        arrays["action_templates"] = np.asarray([record["action_template"] for record in records])
        arrays["evidence_intervention_kinds"] = np.asarray(
            [record["evidence_intervention_kind"] for record in records]
        )
        np.savez_compressed(output_dir / f"{split}.npz", **arrays)
        metadata["splits"][split] = split_metadata
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
