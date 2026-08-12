#!/usr/bin/env python3
"""Perform the single locked V25 truth-hypothesis feature extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

from audit_v25_truth_hypotheses import read_rows
from extract_v10_features_mlx import chat_prompt
from extract_v11_scale_features_mlx import forward_to_layer
from extract_v22r2_relational_features_mlx import canonical_sha256, token_spans
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v25_truth_hypotheses import truth_prompt_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v25-truth-hypotheses-lock.json")
    parser.add_argument("--output-dir", default="outputs/v25-truth-hypotheses/features")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "feature-extraction-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V25 feature extraction was already attempted")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["featureExtractionAttempts"] != 1:
        raise RuntimeError("V25 lock does not authorize exactly one extraction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V25 locked implementation changed: {path}")
    audit_path = PROJECT_ROOT / lock["source"]["pre_extraction_audit"]
    if file_sha256(audit_path) != lock["source"]["pre_extraction_audit_sha256"]:
        raise RuntimeError("V25 pre-extraction audit changed after lock")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v25_protocol_lock":
        raise RuntimeError("V25 audit did not authorize model access")
    corpus_root = PROJECT_ROOT / lock["source"]["corpus"]
    rows = sorted(read_rows(corpus_root), key=lambda row: row["id"])
    if len(rows) != audit["budget"]["planned_model_forwards"]:
        raise RuntimeError("V25 row inventory changed after audit")
    for name, expected in lock["source"]["corpus_file_sha256"].items():
        if file_sha256(corpus_root / name) != expected:
            raise RuntimeError(f"V25 corpus file changed after lock: {name}")

    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 25,
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
        raise RuntimeError("V25 loaded model architecture differs from the lock")

    features = []
    prompt_lengths = []
    span_lengths = []
    hidden_dtypes = set()
    prompt_hashes = []
    for index, row in enumerate(rows):
        content, assessment_chars = truth_prompt_layout(row)
        prompt_hashes.append(hashlib.sha256(content.encode()).hexdigest())
        prompt = chat_prompt(content, specification["systemPrompt"], tokenizer)
        tokens, spans = token_spans(
            prompt, content, {row["id"]: assessment_chars}, tokenizer
        )
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError(
                f"V25 prompt exceeds locked maximum: {row['id']} has {len(tokens)} tokens"
            )
        hidden = forward_to_layer(
            model, mx.array([tokens]), specification["extractionLayer"]
        )[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        span = spans[row["id"]]
        features.append(np.asarray(
            mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32
        ))
        prompt_lengths.append(len(tokens))
        span_lengths.append(len(span))
        hidden_dtypes.add(str(hidden.dtype))
        completed = index + 1
        if args.progress_every and (
            completed % args.progress_every == 0 or completed == len(rows)
        ):
            print(
                f"v25 4B truth hypotheses: extracted {completed}/{len(rows)} rows",
                file=sys.stderr, flush=True,
            )
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "v25-truth-features.npz"
    np.savez_compressed(
        artifact_path,
        row_ids=np.asarray([row["id"] for row in rows]),
        truth_features=np.stack(features).astype(np.float32),
    )
    metadata = {
        "schema_version": 25,
        "experiment": "v25_single_explicit_truth_hypothesis_feature_extraction",
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_extraction_number": 1,
        "model": specification,
        "frozen": True,
        "adapter_path": None,
        "rows": len(rows),
        "new_model_forward_passes": len(rows),
        "prompt_payload_sha256": canonical_sha256(prompt_hashes),
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "minimum_assessment_span_tokens": min(span_lengths),
        "maximum_assessment_span_tokens": max(span_lengths),
        "hidden_dtypes": sorted(hidden_dtypes),
        "feature_dtype": "float32",
        "truncated_prompts": 0,
        "feature_artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
        "feature_artifact_sha256": file_sha256(artifact_path),
        "data_access": {
            "model_forward_passes": len(rows),
            "linear_fits": 0,
            "match_head_fits": 0,
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
