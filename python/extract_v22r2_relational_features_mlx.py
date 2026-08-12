#!/usr/bin/env python3
"""Perform the single locked V22r2 frozen-representation extraction."""

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

from audit_v22r2_grounding import read_jsonl_directory
from extract_v10_features_mlx import chat_prompt
from extract_v11_scale_features_mlx import forward_to_layer
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT, scene_prompt_layout


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def token_spans(
    prompt: str, content: str, spans: dict[str, tuple[int, int]], tokenizer: Any,
) -> tuple[list[int], dict[str, list[int]]]:
    encoded = tokenizer._tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    tokens = encoded["input_ids"]
    if tokens != tokenizer.encode(prompt):
        raise RuntimeError("V22r2 offset tokenizer IDs differ from model tokenizer IDs")
    content_start = prompt.rfind(content)
    if content_start < 0:
        raise RuntimeError("V22r2 user content is absent from its chat prompt")
    result = {}
    offsets = encoded["offset_mapping"]
    for identifier, (left, right) in spans.items():
        start = content_start + left
        end = content_start + right
        indices = [
            index for index, (token_left, token_right) in enumerate(offsets)
            if token_left < end and token_right > start
        ]
        if not indices:
            raise RuntimeError(f"V22r2 target span maps to no tokens: {identifier}")
        result[identifier] = indices
    return tokens, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v22r2-relational-grounding-lock.json")
    parser.add_argument("--output-dir", default="outputs/v22r2-relational-grounding/features")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "feature-extraction-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V22r2 feature extraction was already attempted")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["featureExtractionAttempts"] != 1:
        raise RuntimeError("V22r2 lock does not authorize exactly one extraction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V22r2 locked implementation changed: {path}")
    audit_path = PROJECT_ROOT / lock["source"]["pre_extraction_audit"]
    if file_sha256(audit_path) != lock["source"]["pre_extraction_audit_sha256"]:
        raise RuntimeError("V22r2 audit changed after lock")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v22r2_protocol_lock":
        raise RuntimeError("V22r2 pre-extraction audit did not authorize model access")
    scenes = read_jsonl_directory(PROJECT_ROOT / lock["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    if len(scenes) != audit["surface_and_prompts"]["new_model_forward_passes"]:
        raise RuntimeError("V22r2 scene inventory changed after audit")

    attempt_path.write_text(json.dumps({
        "schema_version": "22r2",
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
        raise RuntimeError("V22r2 loaded model architecture differs from the lock")

    candidate_features = []
    evidence_features = []
    candidate_ids = []
    evidence_ids = []
    candidate_scene_indices = []
    evidence_scene_indices = []
    prompt_lengths = []
    span_lengths = []
    hidden_dtypes = set()
    prompt_hashes = []
    for scene_index, scene in enumerate(scenes):
        content, candidate_chars, evidence_chars = scene_prompt_layout(scene)
        prompt_hashes.append(hashlib.sha256(content.encode()).hexdigest())
        prompt = chat_prompt(content, specification["systemPrompt"], tokenizer)
        all_chars = {**candidate_chars, **evidence_chars}
        tokens, spans = token_spans(prompt, content, all_chars, tokenizer)
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError(
                f"V22r2 prompt exceeds locked maximum: {scene['id']} has {len(tokens)} tokens"
            )
        hidden = forward_to_layer(
            model, mx.array([tokens]), specification["extractionLayer"]
        )[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        for row in scene["agent_input"]["atom_candidates"]:
            span = spans[row["id"]]
            candidate_features.append(np.asarray(
                mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32
            ))
            candidate_ids.append(row["id"])
            candidate_scene_indices.append(scene_index)
            span_lengths.append(len(span))
        for row in scene["agent_input"]["evidence"]:
            span = spans[row["id"]]
            evidence_features.append(np.asarray(
                mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32
            ))
            evidence_ids.append(row["id"])
            evidence_scene_indices.append(scene_index)
            span_lengths.append(len(span))
        prompt_lengths.append(len(tokens))
        hidden_dtypes.add(str(hidden.dtype))
        completed = scene_index + 1
        if args.progress_every and (
            completed % args.progress_every == 0 or completed == len(scenes)
        ):
            print(
                f"v22r2 4B: extracted {completed}/{len(scenes)} scenes",
                file=sys.stderr, flush=True,
            )
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "v22r2-relational-features.npz"
    np.savez_compressed(
        artifact_path,
        scene_ids=np.asarray([row["id"] for row in scenes]),
        candidate_ids=np.asarray(candidate_ids),
        evidence_ids=np.asarray(evidence_ids),
        candidate_scene_indices=np.asarray(candidate_scene_indices, dtype=np.int16),
        evidence_scene_indices=np.asarray(evidence_scene_indices, dtype=np.int16),
        candidate_features=np.stack(candidate_features).astype(np.float32),
        evidence_features=np.stack(evidence_features).astype(np.float32),
    )
    metadata = {
        "schema_version": "22r2",
        "experiment": "v22r2_single_frozen_relational_feature_extraction",
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_extraction_number": 1,
        "model": specification,
        "frozen": True,
        "adapter_path": None,
        "scenes": len(scenes),
        "candidate_spans": len(candidate_features),
        "evidence_spans": len(evidence_features),
        "new_model_forward_passes": len(scenes),
        "prompt_payload_sha256": canonical_sha256(prompt_hashes),
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "minimum_target_span_tokens": min(span_lengths),
        "maximum_target_span_tokens": max(span_lengths),
        "hidden_dtypes": sorted(hidden_dtypes),
        "feature_dtype": "float32",
        "truncated_prompts": 0,
        "feature_artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
        "feature_artifact_sha256": file_sha256(artifact_path),
        "data_access": {
            "model_forward_passes": len(scenes),
            "model_predictions_read": 0,
            "linear_fits": 0,
            "adapter_training_runs": 0,
            "v21_final_records_read": 0,
            "v21_final_model_results_read": 0,
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
