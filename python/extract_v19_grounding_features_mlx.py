#!/usr/bin/env python3
"""Single locked extraction of deduplicated V19 grounding prompt features."""

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

from audit_v19_compatibility import prompt_inventory, read_scenes
from extract_v10_features_mlx import (
    BASE_SYSTEM_PROMPT, NLI_SYSTEM_PROMPT, base_text, chat_prompt, nli_text,
    prompt_tokens_and_span,
)
from extract_v11_scale_features_mlx import forward_to_layer
from extract_v13_token_local_mlx import hypothesis_from_text, prompt_tokens_and_hypothesis_span
from v10_protocol import RELATION_ORDER, TEMPORAL_ORDER, VALUE_ORDER, file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v19-frozen-integration-lock.json")
    parser.add_argument("--output-dir", default="outputs/v19-frozen-integration/features")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_prompt_mappings(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    base_prompts, nli_prompts, base_evidence = prompt_inventory(scenes)
    base_index = {value: index for index, value in enumerate(base_prompts)}
    nli_index = {value: index for index, value in enumerate(nli_prompts)}
    pair_base = []
    pair_nli = []
    pair_scenes = []
    determinant_indices = []
    evidence_indices = []
    match_targets = []
    temporal_targets = []
    current_targets = []
    for scene_index, scene in enumerate(scenes):
        hypotheses = {
            value["determinant_id"]: value["statements"]
            for value in scene["agent_input"]["state_hypotheses"]
        }
        for determinant_index, target in enumerate(scene["target"]["determinant_grounding"]):
            positive = next(
                index for index, value in enumerate(scene["evidence_units"])
                if value == target["evidence_span"]
            )
            for evidence_index in range(len(scene["evidence_units"])):
                base = base_text(scene, determinant_index, evidence_index)
                nli_values = [
                    nli_index[nli_text(scene, determinant_index, evidence_index, hypothesis)]
                    for hypothesis in hypotheses[target["determinant_id"]]
                ]
                pair_base.append(base_index[base])
                pair_nli.append(nli_values)
                pair_scenes.append(scene_index)
                determinant_indices.append(determinant_index)
                evidence_indices.append(evidence_index)
                match_targets.append(evidence_index == positive)
                temporal_targets.append(TEMPORAL_ORDER.index(target["temporal_status"]))
                current_targets.append(
                    VALUE_ORDER.index(target["current_value"])
                    if target["current_value"] is not None else -1
                )
    return {
        "base_prompts": base_prompts,
        "nli_prompts": nli_prompts,
        "base_evidence": base_evidence,
        "pair_base_indices": pair_base,
        "pair_nli_indices": pair_nli,
        "pair_scene_indices": pair_scenes,
        "determinant_indices": determinant_indices,
        "evidence_indices": evidence_indices,
        "match_targets": match_targets,
        "temporal_targets": temporal_targets,
        "current_value_targets": current_targets,
    }


def main() -> None:
    args = parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    metadata_path = output_dir / "metadata.json"
    if output_dir.exists():
        raise RuntimeError(f"V19 feature directory already exists; refusing another extraction: {output_dir}")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["featureExtractionsPermitted"] != 1:
        raise RuntimeError("V19 lock does not authorize exactly one feature extraction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V19 locked implementation changed: {path}")
    audit_path = PROJECT_ROOT / lock["source"]["pre_extraction_audit"]
    if file_sha256(audit_path) != lock["source"]["pre_extraction_audit_sha256"]:
        raise RuntimeError("V19 pre-extraction audit changed after lock")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_single_v19_feature_extraction":
        raise RuntimeError("V19 pre-extraction gate did not authorize extraction")
    dataset_dir = PROJECT_ROOT / lock["source"]["v19_dataset"]
    scenes = read_scenes(dataset_dir)
    arrays = build_prompt_mappings(scenes)
    total = len(arrays["base_prompts"]) + len(arrays["nli_prompts"])
    if total != audit["prompt_inventory"]["new_model_forward_passes"]:
        raise RuntimeError("V19 prompt inventory changed after audit")

    specification = lock["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != specification["totalLayers"] or text_config["hidden_size"] != specification["hiddenSize"]:
        raise RuntimeError("V19 loaded model architecture differs from lock")
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
            raise RuntimeError("V19 base prompt exceeds locked maximum")
        hidden = forward_to_layer(model, mx.array([tokens]), specification["extractionLayer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        base_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        prompt_lengths.append(len(tokens)); target_span_lengths.append(len(span)); hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every and (completed % args.progress_every == 0 or completed == total):
            print(f"v19 4B: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()
    for text in arrays["nli_prompts"]:
        hypothesis = hypothesis_from_text(text)
        prompt = chat_prompt(text, NLI_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_hypothesis_span(prompt, hypothesis, tokenizer)
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError("V19 NLI prompt exceeds locked maximum")
        hidden = forward_to_layer(model, mx.array([tokens]), specification["extractionLayer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        nli_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        prompt_lengths.append(len(tokens)); target_span_lengths.append(len(span)); hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every and (completed % args.progress_every == 0 or completed == total):
            print(f"v19 4B: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "v19-grounding-features.npz"
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
        "schema_version": 19,
        "experiment": "v19_single_frozen_grounding_feature_extraction",
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
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
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "adapter_training_runs": 0,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
