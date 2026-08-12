#!/usr/bin/env python3
"""Extract the sealed V17 final prompts exactly once with the frozen V15 backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from extract_v10_features_mlx import (
    BASE_SYSTEM_PROMPT, NLI_SYSTEM_PROMPT, chat_prompt, prompt_tokens_and_span,
)
from extract_v11_scale_features_mlx import evidence_from_base_text, forward_to_layer
from extract_v13_token_local_mlx import hypothesis_from_text, prompt_tokens_and_hypothesis_span
from v10_protocol import file_sha256
from v17_protocol import build_v17_prompts, canonical_sha256, load_v17_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v17-final-evaluation-lock.json")
    parser.add_argument("--output-dir", default="outputs/v17-final/features")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"V17 feature directory already exists; refusing a second extraction: {output_dir}")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["final_feature_extractions_permitted"] != 1:
        raise RuntimeError("V17 lock does not permit exactly one final feature extraction")
    for path, expected in lock["implementation"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"V17 locked implementation changed: {path}")
    records = load_v17_records(lock)
    arrays = build_v17_prompts(records)

    dev_metadata_path = Path(lock["source"]["v15_features"])
    if file_sha256(dev_metadata_path) != lock["source"]["v15_features_sha256"]:
        raise RuntimeError("V17 frozen V15 metadata changed")
    dev_metadata = json.loads(dev_metadata_path.read_text())
    dev_feature_path = Path(dev_metadata["feature_artifact"])
    if file_sha256(dev_feature_path) != lock["source"]["v15_feature_artifact_sha256"]:
        raise RuntimeError("V17 frozen V15 feature artifact changed")
    with np.load(dev_feature_path, allow_pickle=False) as dev:
        dev_base = set(dev["base_prompts"].tolist())
        dev_nli = set(dev["nli_prompts"].tolist())
    base_overlap = len(dev_base.intersection(arrays["base_prompts"]))
    nli_overlap = len(dev_nli.intersection(arrays["nli_prompts"]))
    if base_overlap or nli_overlap:
        raise RuntimeError(f"V17 final prompts overlap development: base={base_overlap}, nli={nli_overlap}")

    specification = lock["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != specification["total_layers"] or text_config["hidden_size"] != specification["hidden_size"]:
        raise RuntimeError("V17 loaded architecture differs from the seal")

    base_features = []
    nli_features = []
    base_lengths = []
    evidence_lengths = []
    nli_lengths = []
    hypothesis_lengths = []
    hidden_dtypes: set[str] = set()
    total = len(arrays["base_prompts"]) + len(arrays["nli_prompts"])
    completed = 0
    for text in arrays["base_prompts"]:
        evidence = evidence_from_base_text(text)
        prompt = chat_prompt(text, BASE_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_span(prompt, evidence, tokenizer)
        if len(tokens) > lock["max_sequence_length"]:
            raise RuntimeError("V17 base prompt exceeds maximum sequence length")
        hidden = forward_to_layer(model, mx.array([tokens]), specification["extraction_layer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        base_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        base_lengths.append(len(tokens)); evidence_lengths.append(len(span)); hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every and (completed % args.progress_every == 0 or completed == total):
            print(f"v17 final 4B: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()
    for text in arrays["nli_prompts"]:
        hypothesis = hypothesis_from_text(text)
        prompt = chat_prompt(text, NLI_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_hypothesis_span(prompt, hypothesis, tokenizer)
        if len(tokens) > lock["max_sequence_length"]:
            raise RuntimeError("V17 NLI prompt exceeds maximum sequence length")
        hidden = forward_to_layer(model, mx.array([tokens]), specification["extraction_layer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        nli_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        nli_lengths.append(len(tokens)); hypothesis_lengths.append(len(span)); hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every and (completed % args.progress_every == 0 or completed == total):
            print(f"v17 final 4B: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "v17-final-features.npz"
    np.savez_compressed(
        artifact_path,
        record_ids=np.asarray([record["id"] for record in records]),
        base_prompts=np.asarray(arrays["base_prompts"]), nli_prompts=np.asarray(arrays["nli_prompts"]),
        base_span_features=np.stack(base_features).astype(np.float32),
        nli_hypothesis_mean_features=np.stack(nli_features).astype(np.float32),
        pair_base_indices=np.asarray(arrays["pair_base_indices"], dtype=np.int32),
        pair_nli_indices=np.asarray(arrays["pair_nli_indices"], dtype=np.int32),
        pair_record_indices=np.asarray(arrays["pair_record_indices"], dtype=np.int32),
        determinant_indices=np.asarray(arrays["determinant_indices"], dtype=np.int8),
        evidence_indices=np.asarray(arrays["evidence_indices"], dtype=np.int8),
        match_targets=np.asarray(arrays["match_targets"], dtype=np.uint8),
        temporal_targets=np.asarray(arrays["temporal_targets"], dtype=np.int8),
        current_value_targets=np.asarray(arrays["current_value_targets"], dtype=np.int8),
        unique_base_match_targets=np.asarray(arrays["base_match"], dtype=np.uint8),
        unique_base_temporal_targets=np.asarray(arrays["base_temporal"], dtype=np.int8),
    )
    metadata = {
        "schema_version": 17, "experiment": "v17_sealed_final_feature_extraction",
        "evaluation_lock": str(lock_path), "evaluation_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"], "model": specification,
        "frozen": True, "adapter_path": None, "final_feature_extraction_number": 1,
        "records_read": len(records), "unique_base_prompts": len(arrays["base_prompts"]),
        "unique_nli_prompts": len(arrays["nli_prompts"]), "new_model_forward_passes": total,
        "development_base_prompt_overlaps": base_overlap, "development_nli_prompt_overlaps": nli_overlap,
        "base_prompt_text_sha256": canonical_sha256(arrays["base_prompts"]),
        "nli_prompt_text_sha256": canonical_sha256(arrays["nli_prompts"]),
        "feature_dtype": "float32", "hidden_dtypes": sorted(hidden_dtypes), "truncated_prompts": 0,
        "base_token_length": {"minimum": min(base_lengths), "maximum": max(base_lengths)},
        "evidence_token_length": {"minimum": min(evidence_lengths), "maximum": max(evidence_lengths)},
        "nli_token_length": {"minimum": min(nli_lengths), "maximum": max(nli_lengths)},
        "hypothesis_token_length": {"minimum": min(hypothesis_lengths), "maximum": max(hypothesis_lengths)},
        "feature_artifact": str(artifact_path), "feature_artifact_sha256": file_sha256(artifact_path),
        "data_access": {
            **lock["data_access_at_seal"],
            "final_v17_mechanic_records_read": len(records),
            "final_v17_model_scores_read": 0,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
