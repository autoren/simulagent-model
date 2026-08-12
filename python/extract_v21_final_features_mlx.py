#!/usr/bin/env python3
"""Single frozen extraction for the sealed V21 final suite."""

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

from audit_v19_compatibility import read_scenes
from extract_v10_features_mlx import (
    BASE_SYSTEM_PROMPT, NLI_SYSTEM_PROMPT, chat_prompt, prompt_tokens_and_span,
)
from extract_v11_scale_features_mlx import forward_to_layer
from extract_v13_token_local_mlx import hypothesis_from_text, prompt_tokens_and_hypothesis_span
from extract_v19_grounding_features_mlx import build_prompt_mappings
from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", default="configs/v21-final-dataset-seal.json")
    parser.add_argument("--output-dir", default="outputs/v21-final/features")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    seal_path = PROJECT_ROOT / args.seal
    seal = json.loads(seal_path.read_text())
    output_dir = PROJECT_ROOT / args.output_dir
    if output_dir.exists():
        raise RuntimeError(f"V21 feature directory exists; retry forbidden: {output_dir}")
    if seal["limits"]["featureExtractionsPermitted"] != 1:
        raise RuntimeError("V21 seal does not authorize exactly one feature extraction")
    for path, expected in seal["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V21 locked implementation changed: {path}")
    audit_path = PROJECT_ROOT / seal["pre_extraction_audit"]
    if file_sha256(audit_path) != seal["pre_extraction_audit_sha256"]:
        raise RuntimeError("V21 pre-extraction audit changed after seal")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_single_v21_feature_extraction":
        raise RuntimeError("V21 pre-extraction audit blocks feature extraction")
    manifest_path = PROJECT_ROOT / seal["manifest"]
    if file_sha256(manifest_path) != seal["manifest_sha256"]:
        raise RuntimeError("V21 manifest changed after seal")
    scenes = read_scenes(manifest_path.parent)
    arrays = build_prompt_mappings(scenes)
    total = len(arrays["base_prompts"]) + len(arrays["nli_prompts"])
    if total != audit["prompt_inventory"]["new_model_forward_passes"]:
        raise RuntimeError("V21 prompt inventory changed after audit")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "attempt.json").write_text(json.dumps({
        "schema_version": 21,
        "feature_extraction_number": 1,
        "dataset_seal": args.seal,
        "dataset_seal_sha256": file_sha256(seal_path),
        "retry_authorized": False,
    }, indent=2, sort_keys=True) + "\n")
    specification = seal["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if (
        text_config["num_hidden_layers"] != specification["totalLayers"]
        or text_config["hidden_size"] != specification["hiddenSize"]
    ):
        raise RuntimeError("V21 loaded model architecture differs from lock")
    base_features = []
    nli_features = []
    prompt_lengths = []
    target_span_lengths = []
    hidden_dtypes = set()
    completed = 0
    for text in arrays["base_prompts"]:
        evidence = arrays["base_evidence"][text]
        prompt = chat_prompt(text, BASE_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_span(prompt, evidence, tokenizer)
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError("V21 base prompt exceeds locked maximum")
        hidden = forward_to_layer(model, mx.array([tokens]), specification["extractionLayer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        base_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        prompt_lengths.append(len(tokens))
        target_span_lengths.append(len(span))
        hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every and (completed % args.progress_every == 0 or completed == total):
            print(f"v21 4B: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()
    for text in arrays["nli_prompts"]:
        hypothesis = hypothesis_from_text(text)
        prompt = chat_prompt(text, NLI_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_hypothesis_span(prompt, hypothesis, tokenizer)
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError("V21 NLI prompt exceeds locked maximum")
        hidden = forward_to_layer(model, mx.array([tokens]), specification["extractionLayer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        nli_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        prompt_lengths.append(len(tokens))
        target_span_lengths.append(len(span))
        hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every and (completed % args.progress_every == 0 or completed == total):
            print(f"v21 4B: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()
    artifact_path = output_dir / "v21-final-grounding-features.npz"
    np.savez_compressed(
        artifact_path,
        scene_ids=np.asarray([value["id"] for value in scenes]),
        base_prompts=np.asarray(arrays["base_prompts"]),
        nli_prompts=np.asarray(arrays["nli_prompts"]),
        base_span_features=np.stack(base_features).astype(np.float32),
        nli_hypothesis_mean_features=np.stack(nli_features).astype(np.float32),
        pair_base_indices=np.asarray(arrays["pair_base_indices"], dtype=np.int32),
        pair_nli_indices=np.asarray(arrays["pair_nli_indices"], dtype=np.int32),
        pair_scene_indices=np.asarray(arrays["pair_scene_indices"], dtype=np.int32),
        determinant_indices=np.asarray(arrays["determinant_indices"], dtype=np.int8),
        evidence_indices=np.asarray(arrays["evidence_indices"], dtype=np.int8),
        match_targets=np.asarray(arrays["match_targets"], dtype=np.uint8),
        temporal_targets=np.asarray(arrays["temporal_targets"], dtype=np.int8),
        current_value_targets=np.asarray(arrays["current_value_targets"], dtype=np.int8),
    )
    metadata = {
        "schema_version": 21,
        "experiment": "v21_single_frozen_final_feature_extraction",
        "dataset_seal": args.seal,
        "dataset_seal_sha256": file_sha256(seal_path),
        "feature_extraction_number": 1,
        "model": specification,
        "frozen": True,
        "adapter_path": None,
        "scenes": len(scenes),
        "unique_base_prompts": len(base_features),
        "unique_nli_prompts": len(nli_features),
        "new_model_forward_passes": total,
        "base_prompt_text_sha256": canonical_sha256(arrays["base_prompts"]),
        "nli_prompt_text_sha256": canonical_sha256(arrays["nli_prompts"]),
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "minimum_target_span_tokens": min(target_span_lengths),
        "maximum_target_span_tokens": max(target_span_lengths),
        "hidden_dtypes": sorted(hidden_dtypes),
        "feature_dtype": "float32",
        "truncated_prompts": 0,
        "feature_artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
        "feature_artifact_sha256": file_sha256(artifact_path),
        "data_access": {
            "final_scenes_read": len(scenes),
            "final_labels_used_for_model_selection": 0,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "adapter_training_runs": 0,
            "new_linear_fits": 0,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
