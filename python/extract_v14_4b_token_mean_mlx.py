#!/usr/bin/env python3
"""Extract the locked V14 4B hypothesis-mean features for unique NLI prompts."""

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

from extract_v10_features_mlx import NLI_SYSTEM_PROMPT, chat_prompt, nli_text
from extract_v11_scale_features_mlx import forward_to_layer
from extract_v13_token_local_mlx import hypothesis_from_text, prompt_tokens_and_hypothesis_span
from v10_protocol import file_sha256
from v14_protocol import load_records_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v14-4b-baseline-lock.json")
    parser.add_argument("--output-dir", default="outputs/v14-4b-baseline/features")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_unique_pairs(records: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], list[int]]:
    pairs: dict[tuple[str, str], int] = {}
    targets: list[int] = []
    for record in records:
        hypotheses = {value["determinant_id"]: value["statements"] for value in record["agent_input"]["state_hypotheses"]}
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            if target["temporal_status"] != "CURRENT":
                continue
            evidence_index = next(index for index, unit in enumerate(record["evidence_units"]) if (
                unit["start"] == target["evidence_span"]["start"]
                and unit["end"] == target["evidence_span"]["end"]
            ))
            pair = tuple(
                nli_text(record, determinant_index, evidence_index, hypothesis)
                for hypothesis in hypotheses[target["determinant_id"]]
            )
            value = 1 if target["current_value"] == "active" else 0
            if pair in pairs:
                if targets[pairs[pair]] != value:
                    raise RuntimeError("V14 duplicate local pair has conflicting targets")
            else:
                pairs[pair] = len(pairs)
                targets.append(value)
    return list(pairs), targets


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"V14 features already exist: {metadata_path}")
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"V14 baseline implementation changed: {path}")
    records = load_records_from_manifest(Path(lock["source"]["manifest"]))
    pairs, targets = build_unique_pairs(records)
    if len(pairs) != 756 or len(targets) != 756:
        raise RuntimeError("V14 unique local pair count differs from overlap audit")
    prompts = [prompt for pair in pairs for prompt in pair]
    if len(set(prompts)) != 1512:
        raise RuntimeError("V14 unique prompt count differs from overlap audit")

    specification = lock["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if (
        text_config["num_hidden_layers"] != specification["total_layers"]
        or text_config["hidden_size"] != specification["hidden_size"]
    ):
        raise RuntimeError("V14 loaded model architecture differs from lock")
    features = []
    prompt_lengths = []
    hypothesis_lengths = []
    hidden_dtypes: set[str] = set()
    for index, text in enumerate(prompts, start=1):
        hypothesis = hypothesis_from_text(text)
        prompt = chat_prompt(text, NLI_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_hypothesis_span(prompt, hypothesis, tokenizer)
        if len(tokens) > lock["max_sequence_length"]:
            raise RuntimeError(f"V14 prompt exceeds maximum: {len(tokens)}")
        hidden = forward_to_layer(model, mx.array([tokens]), specification["extraction_layer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        prompt_lengths.append(len(tokens))
        hypothesis_lengths.append(len(span))
        hidden_dtypes.add(str(hidden.dtype))
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(prompts)):
            print(f"v14 4B token mean: extracted {index}/{len(prompts)}", file=sys.stderr, flush=True)
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "v14-4b-token-mean-features.npz"
    np.savez_compressed(
        artifact_path,
        pair_prompts=np.asarray(pairs),
        pair_targets=np.asarray(targets, dtype=np.int8),
        hypothesis_mean_features=np.stack(features).astype(np.float32),
    )
    metadata = {
        "schema_version": 14,
        "experiment": "v14_4b_unique_pair_hypothesis_mean_extraction",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "model": specification,
        "frozen": True,
        "adapter_path": None,
        "unique_local_pairs": len(pairs),
        "unique_nli_prompts": len(prompts),
        "nli_prompt_text_sha256": canonical_sha256(prompts),
        "feature_dtype": "float32",
        "hidden_dtypes": sorted(hidden_dtypes),
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "minimum_hypothesis_tokens": min(hypothesis_lengths),
        "maximum_hypothesis_tokens": max(hypothesis_lengths),
        "truncated_prompts": 0,
        "feature_artifact": str(artifact_path),
        "feature_artifact_sha256": file_sha256(artifact_path),
        "data_access": lock["data_access"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
