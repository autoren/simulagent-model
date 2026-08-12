#!/usr/bin/env python3
"""Extract the locked 4B hypothesis-token representations for V13."""

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

from extract_v10_features_mlx import NLI_SYSTEM_PROMPT, chat_prompt
from extract_v11_scale_features_mlx import forward_to_layer
from v10_protocol import file_sha256


HYPOTHESIS_MARKER = "\nCurrent-state hypothesis: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v13-token-local-lock.json")
    parser.add_argument("--output-dir", default="outputs/v13-token-local/features")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hypothesis_from_text(text: str) -> str:
    if HYPOTHESIS_MARKER not in text:
        raise RuntimeError("V13 NLI text has no hypothesis marker")
    value = text.rsplit(HYPOTHESIS_MARKER, 1)[1]
    if not value or "\n" in value:
        raise RuntimeError("V13 NLI hypothesis is empty or not terminal")
    return value


def prompt_tokens_and_hypothesis_span(
    prompt: str, hypothesis: str, tokenizer: Any
) -> tuple[list[int], list[int]]:
    encoded = tokenizer._tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded["input_ids"]
    if tokens != tokenizer.encode(prompt):
        raise RuntimeError("V13 offset tokenizer ids differ from model tokenizer ids")
    marker = "Current-state hypothesis: "
    marker_start = prompt.rfind(marker)
    if marker_start < 0:
        raise RuntimeError("V13 rendered prompt has no hypothesis marker")
    start = marker_start + len(marker)
    end = start + len(hypothesis)
    if prompt[start:end] != hypothesis:
        raise RuntimeError("V13 hypothesis text differs after chat rendering")
    span = [index for index, (left, right) in enumerate(encoded["offset_mapping"]) if left < end and right > start]
    if not span:
        raise RuntimeError("V13 hypothesis maps to no tokens")
    return tokens, span


def verify_lock(lock: dict[str, Any], lock_path: Path) -> None:
    for path, expected in lock["implementation"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"V13 locked implementation changed: {path}")
    source_path = Path(lock["source_v10"]["feature_artifact"])
    if file_sha256(source_path) != lock["source_v10"]["feature_artifact_sha256"]:
        raise RuntimeError("V13 V10 reference artifact changed")
    if lock["model"]["model_key"] != "qwen35_4b" or lock["model"]["extraction_layer"] != 8:
        raise RuntimeError("V13 supports only the locked 4B layer-8 extraction")
    if file_sha256(lock_path) != lock.get("self_sha256", file_sha256(lock_path)):
        raise RuntimeError("V13 lock identity mismatch")


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"V13 features already exist: {metadata_path}")
    lock = json.loads(lock_path.read_text())
    verify_lock(lock, lock_path)
    with np.load(lock["source_v10"]["feature_artifact"], allow_pickle=False) as reference:
        nli_prompts = reference["nli_prompts"].tolist()
    if len(nli_prompts) != 6984:
        raise RuntimeError("V13 expected the exact 6,984 V10 NLI prompts")

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
        raise RuntimeError("V13 loaded architecture differs from its lock")

    last_features = []
    mean_features = []
    prompt_lengths = []
    hypothesis_lengths = []
    hidden_dtypes: set[str] = set()
    for index, text in enumerate(nli_prompts, start=1):
        hypothesis = hypothesis_from_text(text)
        prompt = chat_prompt(text, NLI_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_hypothesis_span(prompt, hypothesis, tokenizer)
        if len(tokens) > lock["max_sequence_length"]:
            raise RuntimeError(f"V13 prompt exceeds maximum: {len(tokens)}")
        hidden = forward_to_layer(
            model, mx.array([tokens]), specification["extraction_layer"]
        )[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        positions = mx.array(span)
        last_features.append(np.asarray(hidden32[span[-1]], dtype=np.float32))
        mean_features.append(np.asarray(mx.mean(hidden32[positions], axis=0), dtype=np.float32))
        prompt_lengths.append(len(tokens))
        hypothesis_lengths.append(len(span))
        hidden_dtypes.add(str(hidden.dtype))
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(nli_prompts)):
            print(f"v13 4B token-local: extracted {index}/{len(nli_prompts)}", file=sys.stderr, flush=True)
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "v13-token-local-features.npz"
    np.savez_compressed(
        artifact_path,
        hypothesis_last_features=np.stack(last_features).astype(np.float32),
        hypothesis_mean_features=np.stack(mean_features).astype(np.float32),
    )
    metadata = {
        "schema_version": 13,
        "experiment": "v13_4b_token_local_extraction",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "model": specification,
        "frozen": True,
        "adapter_path": None,
        "source_v10_feature_artifact_sha256": lock["source_v10"]["feature_artifact_sha256"],
        "nli_prompt_text_sha256": canonical_sha256(nli_prompts),
        "unique_nli_prompts": len(nli_prompts),
        "representations": lock["representations"],
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
