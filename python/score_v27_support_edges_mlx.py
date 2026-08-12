#!/usr/bin/env python3
"""Score the single locked set of previously unscored V27 support edges."""

from __future__ import annotations

import argparse
import json
import sys

import mlx.core as mx
from mlx_lm import load

from audit_v27_support_map import read_rows
from evaluate_v26_native_decoder_mlx import dequantized_label_rows
from extract_v10_features_mlx import chat_prompt
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v26_native_decoder import decoder_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v27-support-map-lock.json")
    parser.add_argument("--output-dir", default="outputs/v27-support-map/edge-scores")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "edge-decoder-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V27 support-edge decoding was already attempted")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["supportEdgeDecoderAttempts"] != 1:
        raise RuntimeError("V27 lock does not authorize one edge-decoder attempt")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V27 locked implementation changed: {path}")
    audit_path = PROJECT_ROOT / lock["source"]["pre_decoder_audit"]
    if file_sha256(audit_path) != lock["source"]["pre_decoder_audit_sha256"]:
        raise RuntimeError("V27 pre-decoder audit changed after lock")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v27_protocol_lock":
        raise RuntimeError("V27 audit did not authorize decoder access")
    corpus_root = PROJECT_ROOT / lock["source"]["corpus"]
    rows = sorted(read_rows(corpus_root), key=lambda row: row["id"])
    if len(rows) != lock["limits"]["newModelForwardPasses"]:
        raise RuntimeError("V27 decoder row inventory changed")
    for name, expected in lock["source"]["corpus_file_sha256"].items():
        if file_sha256(corpus_root / name) != expected:
            raise RuntimeError(f"V27 corpus changed after lock: {name}")

    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 27, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_model_load",
    }, indent=2, sort_keys=True) + "\n")
    specification = lock["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != specification["totalLayers"] or text_config["hidden_size"] != specification["hiddenSize"]:
        raise RuntimeError("V27 loaded model architecture differs from lock")
    labels = lock["labels"]
    encoded = {label: tokenizer.encode(label, add_special_tokens=False) for label in labels}
    if any(len(tokens) != 1 for tokens in encoded.values()):
        raise RuntimeError(f"V27 decoder labels are not single tokens: {encoded}")
    token_ids = [encoded[label][0] for label in labels]
    label_rows = dequantized_label_rows(model, token_ids)
    mx.eval(label_rows)
    score_rows = []
    prompt_lengths = []
    for index, row in enumerate(rows, start=1):
        prompt = chat_prompt(decoder_prompt(row), specification["systemPrompt"], tokenizer)
        tokens = tokenizer.encode(prompt)
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError(f"V27 prompt exceeds locked maximum: {row['id']}")
        hidden = model.language_model.model(mx.array([tokens]))[0, -1]
        logits = hidden.astype(mx.float32) @ label_rows.T
        mx.eval(hidden, logits)
        values = [float(value) for value in logits.tolist()]
        score_rows.append({
            "id": row["id"], "source_pair_id": row["source_pair_id"],
            "scene_id": row["scene_id"], "evidence_id": row["evidence_id"],
            "candidate_id": row["candidate_id"], "split": row["split"], "role": "support",
            "fp32_direct_logits": {label: values[position] for position, label in enumerate(labels)},
        })
        prompt_lengths.append(len(tokens))
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"v27 support edges: scored {index}/{len(rows)} prompts", file=sys.stderr, flush=True)
        mx.clear_cache()
    output_dir.mkdir(parents=True, exist_ok=False)
    scores_path = output_dir / "support-edge-scores.jsonl"
    scores_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in score_rows
    ))
    metadata = {
        "schema_version": 27, "experiment": "v27_support_edge_native_decoder",
        "protocol_lock_sha256": file_sha256(lock_path),
        "rows": len(rows), "new_model_forward_passes": len(rows),
        "minimum_prompt_tokens": min(prompt_lengths), "maximum_prompt_tokens": max(prompt_lengths),
        "truncated_prompts": 0, "feature_or_head_fits": 0,
        "score_artifact": str(scores_path.relative_to(PROJECT_ROOT)),
        "score_artifact_sha256": file_sha256(scores_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "metadata_sha256": file_sha256(metadata_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
