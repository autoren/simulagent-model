#!/usr/bin/env python3
"""Perform the single locked V24 candidate-conditioned feature extraction."""

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

from audit_v24_cross_encoder import read_pairs
from extract_v10_features_mlx import chat_prompt
from extract_v11_scale_features_mlx import forward_to_layer
from extract_v22r2_relational_features_mlx import canonical_sha256, token_spans
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v24_cross_encoder import cross_prompt_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v24-cross-encoder-lock.json")
    parser.add_argument("--output-dir", default="outputs/v24-cross-encoder/features")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "feature-extraction-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V24 feature extraction was already attempted")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["featureExtractionAttempts"] != 1:
        raise RuntimeError("V24 lock does not authorize exactly one feature extraction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V24 locked implementation changed: {path}")
    audit_path = PROJECT_ROOT / lock["source"]["pre_extraction_audit"]
    if file_sha256(audit_path) != lock["source"]["pre_extraction_audit_sha256"]:
        raise RuntimeError("V24 pre-extraction audit changed after lock")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v24_protocol_lock":
        raise RuntimeError("V24 audit did not authorize frozen model access")
    proposal_root = PROJECT_ROOT / lock["source"]["proposal_corpus"]
    pairs = sorted(read_pairs(proposal_root), key=lambda row: row["id"])
    if len(pairs) != audit["budget"]["planned_model_forwards"]:
        raise RuntimeError("V24 pair inventory changed after audit")
    for name, expected in lock["source"]["proposal_file_sha256"].items():
        if file_sha256(proposal_root / name) != expected:
            raise RuntimeError(f"V24 proposal file changed after lock: {name}")

    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 24,
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "attempt_number": 1,
        "status": "started_before_model_load",
    }, indent=2, sort_keys=True) + "\n")

    specification = lock["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if (
        text_config["num_hidden_layers"] != specification["totalLayers"]
        or text_config["hidden_size"] != specification["hiddenSize"]
    ):
        raise RuntimeError("V24 loaded model architecture differs from the lock")

    features: list[np.ndarray] = []
    prompt_lengths: list[int] = []
    span_lengths: list[int] = []
    hidden_dtypes: set[str] = set()
    prompt_hashes: list[str] = []
    for index, pair in enumerate(pairs):
        content, candidate_chars = cross_prompt_layout(pair)
        prompt_hashes.append(hashlib.sha256(content.encode()).hexdigest())
        prompt = chat_prompt(content, specification["systemPrompt"], tokenizer)
        tokens, spans = token_spans(
            prompt, content, {pair["id"]: candidate_chars}, tokenizer
        )
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError(
                f"V24 prompt exceeds locked maximum: {pair['id']} has {len(tokens)} tokens"
            )
        hidden = forward_to_layer(
            model, mx.array([tokens]), specification["extractionLayer"]
        )[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        span = spans[pair["id"]]
        features.append(np.asarray(
            mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32
        ))
        prompt_lengths.append(len(tokens))
        span_lengths.append(len(span))
        hidden_dtypes.add(str(hidden.dtype))
        completed = index + 1
        if args.progress_every and (
            completed % args.progress_every == 0 or completed == len(pairs)
        ):
            print(
                f"v24 4B cross encoder: extracted {completed}/{len(pairs)} pairs",
                file=sys.stderr, flush=True,
            )
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "v24-cross-features.npz"
    np.savez_compressed(
        artifact_path,
        pair_ids=np.asarray([row["id"] for row in pairs]),
        pair_features=np.stack(features).astype(np.float32),
    )
    metadata = {
        "schema_version": 24,
        "experiment": "v24_single_candidate_conditioned_frozen_feature_extraction",
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_extraction_number": 1,
        "model": specification,
        "frozen": True,
        "adapter_path": None,
        "pairs": len(pairs),
        "new_model_forward_passes": len(pairs),
        "prompt_payload_sha256": canonical_sha256(prompt_hashes),
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "minimum_candidate_span_tokens": min(span_lengths),
        "maximum_candidate_span_tokens": max(span_lengths),
        "hidden_dtypes": sorted(hidden_dtypes),
        "feature_dtype": "float32",
        "truncated_prompts": 0,
        "feature_artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
        "feature_artifact_sha256": file_sha256(artifact_path),
        "data_access": {
            "model_forward_passes": len(pairs),
            "model_predictions_read": 0,
            "linear_fits": 0,
            "hyperparameter_selections": 0,
            "adapter_training_runs": 0,
            "fresh_benchmark_records_read": 0,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({
        "status": "completed",
        "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        "metadata_sha256": file_sha256(metadata_path),
    })
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
