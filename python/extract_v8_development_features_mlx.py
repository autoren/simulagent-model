#!/usr/bin/env python3
"""Extract the locked V8 development-only layer-6 mean representation."""

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

from extract_v6_development_features_mlx import forward_layer_six


SYSTEM_PROMPT = " ".join((
    "Use the supplied action-dependency schema and evidence ledger.",
    "The transition table gives rules but not the current hidden values.",
    "Determine each evidence fact's determinant status and whether all compatible assignments have one transition code.",
    "Do not infer an unstated mechanic or treat additional information as automatically decisive.",
))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v8-development-lock.json")
    parser.add_argument("--output-dir", default="outputs/v8-frozen-diagnostics/features")
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


def prompt_for(record: dict[str, Any], tokenizer: Any) -> tuple[str, list[int]]:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(record["agent_input"], sort_keys=True, separators=(",", ":")),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt, tokenizer.encode(prompt)


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"V8 features already exist: {metadata_path}")
    lock = json.loads(lock_path.read_text())
    if not lock["pre_model_gates"]["passed"]:
        raise RuntimeError("V8 pre-model shortcut gates did not pass")
    manifest_path = Path(lock["dataset_manifest"])
    if file_sha256(manifest_path) != lock["dataset_manifest_sha256"]:
        raise RuntimeError("V8 manifest changed after lock")
    manifest = json.loads(manifest_path.read_text())
    dataset_root = manifest_path.parent
    records: list[dict[str, Any]] = []
    for split in ("train", "calibration"):
        relative = f"records/{split}.jsonl"
        path = dataset_root / relative
        if file_sha256(path) != lock["dataset_artifact_sha256"][relative]:
            raise RuntimeError(f"V8 {split} records changed after lock")
        records.extend(read_jsonl(path))
    if any("tonedrift" in record["source_scenario_id"] for record in records):
        raise RuntimeError("V8 development contains Tone Drift")

    model, tokenizer = load(lock["method"]["model"])
    model.eval()
    features_by_prompt: dict[str, np.ndarray] = {}
    lengths_by_prompt: dict[str, int] = {}
    dtype_by_prompt: dict[str, str] = {}
    prompts: list[str] = []
    token_lists: list[list[int]] = []
    for record in records:
        prompt, tokens = prompt_for(record, tokenizer)
        prompts.append(prompt)
        token_lists.append(tokens)
    unique_prompts = list(dict.fromkeys(prompts))
    first_tokens = {prompt: tokens for prompt, tokens in zip(prompts, token_lists)}
    truncated = 0
    for index, prompt in enumerate(unique_prompts, start=1):
        tokens = first_tokens[prompt]
        if len(tokens) > lock["method"]["max_seq_length"]:
            tokens = tokens[-lock["method"]["max_seq_length"]:]
            truncated += 1
        hidden = forward_layer_six(model, mx.array([tokens]))[0]
        mx.eval(hidden)
        feature = np.asarray(mx.mean(hidden.astype(mx.float32), axis=0), dtype=np.float32)
        features_by_prompt[prompt] = feature
        lengths_by_prompt[prompt] = len(tokens)
        dtype_by_prompt[prompt] = str(hidden.dtype)
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(unique_prompts)):
            print(f"v8 development: extracted {index}/{len(unique_prompts)} unique prompts", file=sys.stderr, flush=True)
        mx.clear_cache()

    arrays = {
        "ids": np.asarray([record["id"] for record in records]),
        "splits": np.asarray([record["split"] for record in records]),
        "groups": np.asarray([record["split_group"] for record in records]),
        "mechanics": np.asarray([record["mechanic"] for record in records]),
        "surface_variants": np.asarray([record["surface_variant"] for record in records]),
        "surface_group_ids": np.asarray([record["surface_group_id"] for record in records]),
        "intervention_group_ids": np.asarray([record["intervention_group_id"] for record in records]),
        "intervention_kinds": np.asarray([record["intervention_kind"] for record in records]),
        "intervention_members": np.asarray([record["intervention_member"] for record in records]),
        "primary_determinant_ids": np.asarray([record["primary_determinant_id"] for record in records]),
        "primary_resolved_values": np.asarray([record["primary_resolved_value"] for record in records], dtype=np.uint8),
        "gold_ambiguous": np.asarray([record["target"]["ambiguous"] for record in records], dtype=np.uint8),
        "prompt_lengths": np.asarray([lengths_by_prompt[prompt] for prompt in prompts], dtype=np.int32),
        "layer_06_mean": np.stack([features_by_prompt[prompt] for prompt in prompts]).astype(np.float32),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "development.npz"
    np.savez_compressed(feature_path, **arrays)
    metadata = {
        "schema_version": 8,
        "experiment": "v8_development_feature_extraction",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "dataset_manifest_sha256": lock["dataset_manifest_sha256"],
        "dataset_sha256": lock["dataset_sha256"],
        "model": lock["method"]["model"],
        "feature": "layer_06_mean",
        "frozen": True,
        "adapter_path": None,
        "records": len(records),
        "unique_prompts": len(unique_prompts),
        "mechanics": sorted(set(arrays["mechanics"].tolist())),
        "feature_dtype": str(arrays["layer_06_mean"].dtype),
        "hidden_dtypes": sorted(set(dtype_by_prompt.values())),
        "minimum_prompt_tokens": int(arrays["prompt_lengths"].min()),
        "maximum_prompt_tokens": int(arrays["prompt_lengths"].max()),
        "truncated_unique_prompts": truncated,
        "feature_artifact": str(feature_path),
        "feature_artifact_sha256": file_sha256(feature_path),
        "v3_test_records_read": 0,
        "prior_holdout_records_read": 0,
        "v7_tone_drift_records_read": 0,
        "v7_model_results_read": 0,
        "untouched_v8_mechanic_records_read": 0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
