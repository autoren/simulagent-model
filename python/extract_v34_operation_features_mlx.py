#!/usr/bin/env python3
"""Perform the single locked V34 operation-focused frozen-model extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

from audit_v32_factorized_semantics import read_rows
from evaluate_v30_signed_fact_language_mlx import dequantized_label_rows
from extract_v10_features_mlx import chat_prompt
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v34_operation import operation_prompt, target_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v34-operation-interface-lock.json")
    parser.add_argument("--output-dir", default="outputs/v34-operation-interface/features")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    lock_path, output_dir = (PROJECT_ROOT / args.lock).resolve(), (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "feature-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V34 feature extraction was already attempted")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    if lock["limits"]["featureExtractions"] != 1:
        raise RuntimeError("V34 lock does not authorize exactly one feature extraction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V34 locked implementation changed: {path}")
    for name, expected in lock["source"]["allowed_corpus_file_sha256"].items():
        if file_sha256(PROJECT_ROOT / config["sourceCorpus"] / name) != expected:
            raise RuntimeError(f"V34 allowed corpus changed: {name}")
    rows = sorted(read_rows(PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])), key=lambda row: row["id"])
    if len(rows) != lock["limits"]["backboneForwardPasses"]:
        raise RuntimeError("V34 population differs from locked forward-pass budget")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 34, "attempt_number": 1, "status": "started_before_model_load",
        "protocol_lock_sha256": file_sha256(lock_path), "evaluation_records_read": 0,
    }, indent=2, sort_keys=True) + "\n")
    spec = config["model"]
    model, tokenizer, model_config = load(spec["model"], revision=spec["revision"], return_config=True)
    model.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != spec["totalLayers"] or text_config["hidden_size"] != spec["hiddenSize"]:
        raise RuntimeError("V34 loaded model architecture differs from lock")
    encoded = [tokenizer.encode(token, add_special_tokens=False) for token in config["operationInterface"]["labelTokens"]]
    if any(len(values) != 1 for values in encoded):
        raise RuntimeError(f"V34 labels are not single tokens: {encoded}")
    token_ids = [values[0] for values in encoded]
    label_rows = dequantized_label_rows(model, token_ids)
    mx.eval(label_rows)
    hidden_values, logits_values, lengths, hashes = [], [], [], []
    for index, row in enumerate(rows, start=1):
        content = operation_prompt(row, config)
        prompt = chat_prompt(content, spec["systemPrompt"], tokenizer)
        tokens = tokenizer.encode(prompt)
        if len(tokens) > spec["maxSequenceLength"]:
            raise RuntimeError(f"V34 prompt exceeds maximum: {row['id']} ({len(tokens)})")
        hidden = model.language_model.model(mx.array([tokens]))[0, -1].astype(mx.float32)
        logits = hidden @ label_rows.T
        mx.eval(hidden, logits)
        hidden_values.append(np.asarray(hidden, dtype=np.float32))
        logits_values.append(np.asarray(logits, dtype=np.float32))
        lengths.append(len(tokens)); hashes.append(hashlib.sha256(content.encode()).hexdigest())
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"v34 operation features: {index}/{len(rows)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "features.npz"
    np.savez_compressed(
        artifact,
        record_ids=np.asarray([row["id"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        surface_names=np.asarray([row["oracle_metadata"]["surface_name"] for row in rows]),
        operation_targets=target_indices(rows, config),
        semantic_hidden=np.stack(hidden_values), native_label_logits=np.stack(logits_values),
    )
    metadata = {
        "schema_version": 34, "experiment": config["experiment"],
        "protocol_lock_sha256": file_sha256(lock_path), "feature_extraction_number": 1,
        "records": len(rows), "fit_records": sum(row["split"] == "factor_fit" for row in rows),
        "calibration_records": sum(row["split"] == "factor_calibration" for row in rows),
        "model": spec, "label_token_ids": dict(zip(config["operationInterface"]["labelTokens"], token_ids, strict=True)),
        "minimum_prompt_tokens": min(lengths), "maximum_prompt_tokens": max(lengths), "truncated_prompts": 0,
        "prompt_payload_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
        "feature_artifact": str(artifact.relative_to(PROJECT_ROOT)), "feature_artifact_sha256": file_sha256(artifact),
        "data_access": {"backbone_forward_passes": len(rows), "fit_records_read": sum(row["split"] == "factor_fit" for row in rows), "calibration_records_read": sum(row["split"] == "factor_calibration" for row in rows), "v32_evaluation_records_read": 0, "adapter_training_runs": 0},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "metadata_sha256": file_sha256(metadata_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
