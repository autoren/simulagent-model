#!/usr/bin/env python3
"""Extract only V6 train/calibration layer-6 mean features; never read the mechanic holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

from extract_frozen_qwen_features_mlx import SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v6-protocol-lock.json")
    parser.add_argument("--output-dir", default="outputs/v6-frozen-probe/features")
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
        raise RuntimeError("V6 dataset manifest changed after protocol lock")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_sha256"] != lock["dataset_sha256"]:
        raise RuntimeError("V6 dataset identity changed after protocol lock")
    implementation = lock["implementation"]["extractor"]
    if file_sha256(Path(implementation["path"])) != implementation["sha256"]:
        raise RuntimeError("V6 extractor changed after protocol lock")
    if lock["data_access"]["mechanic_holdout_records_read_before_probe_lock"] != 0:
        raise RuntimeError("Protocol lock reports pre-lock holdout access")
    return lock, manifest


def forward_layer_six(model: Any, inputs: mx.array) -> mx.array:
    text_model = model.language_model.model
    hidden = text_model.embed_tokens(inputs)
    cache = [None] * len(text_model.layers)
    full_attention_mask = create_attention_mask(hidden, cache[text_model.fa_idx])
    state_space_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx])
    for layer_number, (layer, layer_cache) in enumerate(zip(text_model.layers, cache), start=1):
        mask = state_space_mask if layer.is_linear else full_attention_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)
        if layer_number == 6:
            return hidden
    raise RuntimeError("Qwen model has fewer than six layers")


def prompt_for(record: dict[str, Any], tokenizer: Any) -> list[int]:
    user_input = json.loads(json.dumps(record["agent_input"]))
    user_input["task"] = "classify_identifiability"
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_input, sort_keys=True, separators=(",", ":")),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(prompt)


def extract_split(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_seq_length: int,
    progress_every: int,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    features: list[np.ndarray] = []
    lengths: list[int] = []
    hidden_dtypes: set[str] = set()
    truncated = 0
    for index, record in enumerate(records, start=1):
        tokens = prompt_for(record, tokenizer)
        if len(tokens) > max_seq_length:
            tokens = tokens[-max_seq_length:]
            truncated += 1
        hidden = forward_layer_six(model, mx.array([tokens]))[0]
        mx.eval(hidden)
        hidden_dtypes.add(str(hidden.dtype))
        pooled = mx.mean(hidden.astype(mx.float32), axis=0)
        features.append(np.asarray(pooled, dtype=np.float32))
        lengths.append(len(tokens))
        if progress_every > 0 and (index % progress_every == 0 or index == len(records)):
            print(f"{split}: extracted {index}/{len(records)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    arrays = {
        "ids": np.asarray([record["id"] for record in records]),
        "groups": np.asarray([record["split_group"] for record in records]),
        "surface_pair_ids": np.asarray([record["surface_pair_id"] for record in records]),
        "surface_variants": np.asarray([record["surface_variant"] for record in records]),
        "evidence_intervention_ids": np.asarray(
            [record["evidence_intervention_id"] or "" for record in records]
        ),
        "evidence_variants": np.asarray([record["evidence_variant"] for record in records]),
        "mechanics": np.asarray([record["mechanic"] for record in records]),
        "gold_ambiguous": np.asarray(
            [record["target"]["ambiguous"] for record in records], dtype=np.uint8
        ),
        "prompt_lengths": np.asarray(lengths, dtype=np.int32),
        "layer_06_mean": np.stack(features).astype(np.float32),
    }
    metadata = {
        "records": len(records),
        "base_records": len(set(arrays["surface_pair_ids"].tolist())),
        "context_groups": len(set(arrays["groups"].tolist())),
        "truncated_prompts": truncated,
        "minimum_prompt_tokens": min(lengths),
        "maximum_prompt_tokens": max(lengths),
        "hidden_dtypes": sorted(hidden_dtypes),
        "feature_dtype": str(arrays["layer_06_mean"].dtype),
    }
    return arrays, metadata


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    lock, manifest = verify_protocol(lock_path)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"Development features already exist; refusing a second extraction: {metadata_path}")
    model, tokenizer = load(lock["method"]["model"])
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "experiment": "v6_development_feature_extraction",
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
        "mechanic_holdout_records_read": 0,
        "v3_test_records_read": 0,
    }
    for split in ("train", "calibration"):
        relative_path = f"records/{split}.jsonl"
        path = Path("data/v6") / relative_path
        if file_sha256(path) != manifest["artifact_sha256"][relative_path]:
            raise RuntimeError(f"V6 {split} records changed after generation")
        records = read_jsonl(path)
        arrays, split_metadata = extract_split(
            model,
            tokenizer,
            records,
            lock["method"]["max_seq_length"],
            args.progress_every,
            split,
        )
        np.savez_compressed(output_dir / f"{split}.npz", **arrays)
        metadata["splits"][split] = split_metadata
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
