#!/usr/bin/env python3
"""Extract the locked V10 representations from one pinned V11 larger backbone."""

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

from extract_v10_features_mlx import (
    BASE_SYSTEM_PROMPT,
    NLI_SYSTEM_PROMPT,
    chat_prompt,
    prompt_tokens_and_span,
)
from v10_protocol import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v11-frozen-scale-lock.json")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def forward_to_layer(model: Any, inputs: mx.array, extraction_layer: int) -> mx.array:
    text_model = model.language_model.model
    hidden = text_model.embed_tokens(inputs)
    cache = [None] * len(text_model.layers)
    full_attention_mask = create_attention_mask(hidden, cache[text_model.fa_idx])
    state_space_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx])
    for layer_number, (layer, layer_cache) in enumerate(zip(text_model.layers, cache), start=1):
        mask = state_space_mask if layer.is_linear else full_attention_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)
        if layer_number == extraction_layer:
            return hidden
    raise RuntimeError(f"Model has fewer than {extraction_layer} layers")


def evidence_from_base_text(text: str) -> str:
    marker = "\nEvidence excerpt: "
    if marker not in text:
        raise RuntimeError("V11 reference base prompt has no evidence marker")
    return text.rsplit(marker, 1)[1]


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text())
    if args.model_key not in lock["models"]:
        raise RuntimeError(f"Unknown V11 model key: {args.model_key}")
    model_spec = lock["models"][args.model_key]
    output_dir = Path(args.output_dir or f"outputs/v11-frozen-scale/features/{args.model_key}")
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"V11 features already exist: {metadata_path}")
    reference_path = Path(lock["source_v10"]["feature_artifact"])
    if file_sha256(reference_path) != lock["source_v10"]["feature_artifact_sha256"]:
        raise RuntimeError("V10 reference feature artifact changed")
    with np.load(reference_path, allow_pickle=False) as reference:
        base_prompts = reference["base_prompts"].tolist()
        nli_prompts = reference["nli_prompts"].tolist()
    if len(base_prompts) != 3492 or len(nli_prompts) != 6984:
        raise RuntimeError("V11 reference prompt counts differ from V10")

    model, tokenizer, model_config = load(
        model_spec["model"],
        revision=model_spec["revision"],
        return_config=True,
    )
    model.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != model_spec["totalLayers"]:
        raise RuntimeError("V11 loaded model layer count differs from lock")
    if text_config["hidden_size"] != model_spec["hiddenSize"]:
        raise RuntimeError("V11 loaded model hidden size differs from lock")
    expected_layer = round(
        model_spec["totalLayers"]
        * lock["depth_rule"]["referenceExtractionLayer"]
        / lock["depth_rule"]["referenceLayers"]
    )
    if expected_layer != model_spec["extractionLayer"]:
        raise RuntimeError("V11 extraction layer violates the locked homologous-depth rule")

    mean_features: list[np.ndarray] = []
    span_features: list[np.ndarray] = []
    nli_features: list[np.ndarray] = []
    base_lengths: list[int] = []
    span_lengths: list[int] = []
    nli_lengths: list[int] = []
    hidden_dtypes: set[str] = set()
    total = len(base_prompts) + len(nli_prompts)
    completed = 0
    for text in base_prompts:
        evidence = evidence_from_base_text(text)
        prompt = chat_prompt(text, BASE_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_span(prompt, evidence, tokenizer)
        if len(tokens) > lock["protocol"]["maxSequenceLength"]:
            raise RuntimeError(f"V11 base prompt exceeds maximum: {len(tokens)}")
        hidden = forward_to_layer(model, mx.array([tokens]), model_spec["extractionLayer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        mean_features.append(np.asarray(mx.mean(hidden32, axis=0), dtype=np.float32))
        span_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        base_lengths.append(len(tokens))
        span_lengths.append(len(span))
        hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == total):
            print(f"v11 {args.model_key}: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()

    for text in nli_prompts:
        prompt = chat_prompt(text, NLI_SYSTEM_PROMPT, tokenizer)
        tokens = tokenizer.encode(prompt)
        if len(tokens) > lock["protocol"]["maxSequenceLength"]:
            raise RuntimeError(f"V11 NLI prompt exceeds maximum: {len(tokens)}")
        hidden = forward_to_layer(model, mx.array([tokens]), model_spec["extractionLayer"])[0]
        mx.eval(hidden)
        nli_features.append(np.asarray(hidden[-1].astype(mx.float32), dtype=np.float32))
        nli_lengths.append(len(tokens))
        hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == total):
            print(f"v11 {args.model_key}: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "v11-features.npz"
    np.savez_compressed(
        artifact_path,
        base_mean_features=np.stack(mean_features).astype(np.float32),
        base_span_features=np.stack(span_features).astype(np.float32),
        nli_final_features=np.stack(nli_features).astype(np.float32),
    )
    metadata = {
        "schema_version": 11,
        "experiment": "v11_frozen_scale_feature_extraction",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "model_key": args.model_key,
        "model": model_spec["model"],
        "revision": model_spec["revision"],
        "total_layers": model_spec["totalLayers"],
        "extraction_layer": model_spec["extractionLayer"],
        "relative_depth": model_spec["extractionLayer"] / model_spec["totalLayers"],
        "hidden_size": model_spec["hiddenSize"],
        "frozen": True,
        "adapter_path": None,
        "source_v10_feature_artifact_sha256": lock["source_v10"]["feature_artifact_sha256"],
        "base_prompt_text_sha256": canonical_sha256(base_prompts),
        "nli_prompt_text_sha256": canonical_sha256(nli_prompts),
        "unique_base_prompts": len(base_prompts),
        "unique_nli_prompts": len(nli_prompts),
        "feature_dtype": "float32",
        "hidden_dtypes": sorted(hidden_dtypes),
        "minimum_base_prompt_tokens": min(base_lengths),
        "maximum_base_prompt_tokens": max(base_lengths),
        "minimum_evidence_span_tokens": min(span_lengths),
        "maximum_evidence_span_tokens": max(span_lengths),
        "minimum_nli_prompt_tokens": min(nli_lengths),
        "maximum_nli_prompt_tokens": max(nli_lengths),
        "truncated_prompts": 0,
        "feature_artifact": str(artifact_path),
        "feature_artifact_sha256": file_sha256(artifact_path),
        "data_access": lock["data_access"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
