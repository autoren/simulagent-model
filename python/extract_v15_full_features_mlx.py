#!/usr/bin/env python3
"""Extract deduplicated V15 base-span and hypothesis-mean features."""

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

from extract_v10_features_mlx import (
    BASE_SYSTEM_PROMPT, NLI_SYSTEM_PROMPT, base_text, chat_prompt, nli_text, prompt_tokens_and_span,
)
from extract_v11_scale_features_mlx import evidence_from_base_text, forward_to_layer
from extract_v13_token_local_mlx import hypothesis_from_text, prompt_tokens_and_hypothesis_span
from v10_protocol import TEMPORAL_ORDER, file_sha256
from v14_protocol import load_records_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v15-full-pipeline-lock.json")
    parser.add_argument("--output-dir", default="outputs/v15-full-pipeline/features")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_prompts(records: list[dict[str, Any]]) -> dict[str, Any]:
    base_prompts: list[str] = []
    base_index: dict[str, int] = {}
    base_match: list[bool] = []
    base_temporal: list[int] = []
    nli_prompts: list[str] = []
    nli_index: dict[str, int] = {}
    pair_base = []
    pair_nli = []
    pair_records = []
    determinant_indices = []
    evidence_indices = []
    match_targets = []
    temporal_targets = []
    current_targets = []
    for record_index, record in enumerate(records):
        hypotheses = {value["determinant_id"]: value["statements"] for value in record["agent_input"]["state_hypotheses"]}
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            positive = next(index for index, unit in enumerate(record["evidence_units"]) if (
                unit["start"] == target["evidence_span"]["start"] and unit["end"] == target["evidence_span"]["end"]
            ))
            for evidence_index, _unit in enumerate(record["evidence_units"]):
                matched = evidence_index == positive
                base = base_text(record, determinant_index, evidence_index)
                temporal = TEMPORAL_ORDER.index(target["temporal_status"]) if matched else -1
                if base in base_index:
                    index = base_index[base]
                    if base_match[index] != matched or (matched and base_temporal[index] != temporal):
                        raise RuntimeError("V15 duplicate base prompt has conflicting targets")
                else:
                    index = len(base_prompts)
                    base_index[base] = index
                    base_prompts.append(base)
                    base_match.append(matched)
                    base_temporal.append(temporal)
                nli_values = []
                for hypothesis in hypotheses[target["determinant_id"]]:
                    text = nli_text(record, determinant_index, evidence_index, hypothesis)
                    if text not in nli_index:
                        nli_index[text] = len(nli_prompts)
                        nli_prompts.append(text)
                    nli_values.append(nli_index[text])
                pair_base.append(index)
                pair_nli.append(nli_values)
                pair_records.append(record_index)
                determinant_indices.append(determinant_index)
                evidence_indices.append(evidence_index)
                match_targets.append(matched)
                temporal_targets.append(temporal)
                current_targets.append(
                    (1 if target["current_value"] == "active" else 0)
                    if matched and target["current_value"] is not None else -1
                )
    if len(base_prompts) != 5022 or len(nli_prompts) != 10044 or len(pair_base) != 94500:
        raise RuntimeError("V15 prompt cardinality differs from preregistration")
    return {
        "base_prompts": base_prompts, "base_match": base_match, "base_temporal": base_temporal,
        "nli_prompts": nli_prompts, "pair_base_indices": pair_base, "pair_nli_indices": pair_nli,
        "pair_record_indices": pair_records, "determinant_indices": determinant_indices,
        "evidence_indices": evidence_indices, "match_targets": match_targets,
        "temporal_targets": temporal_targets, "current_value_targets": current_targets,
    }


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"V15 features already exist: {metadata_path}")
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"V15 locked implementation changed: {path}")
    records = load_records_from_manifest(Path(lock["source"]["manifest"]))
    arrays = build_prompts(records)
    reused_path = Path(lock["source"]["v14_feature_artifact"])
    if file_sha256(reused_path) != lock["source"]["v14_feature_artifact_sha256"]:
        raise RuntimeError("V15 reused V14 feature artifact changed")
    with np.load(reused_path, allow_pickle=False) as reused:
        reused_prompts = [value for pair in reused["pair_prompts"].tolist() for value in pair]
        reused_features = reused["hypothesis_mean_features"].astype(np.float32)
    reused_lookup = {text: reused_features[index] for index, text in enumerate(reused_prompts)}
    if len(reused_lookup) != 1512:
        raise RuntimeError("V15 expected 1,512 reusable V14 NLI features")

    specification = lock["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != specification["total_layers"] or text_config["hidden_size"] != specification["hidden_size"]:
        raise RuntimeError("V15 loaded architecture differs from lock")
    base_features = []
    nli_features = []
    base_lengths = []
    evidence_lengths = []
    nli_lengths = []
    hypothesis_lengths = []
    hidden_dtypes: set[str] = set()
    total_new = len(arrays["base_prompts"]) + len(arrays["nli_prompts"]) - len(reused_lookup)
    completed = 0
    for text in arrays["base_prompts"]:
        evidence = evidence_from_base_text(text)
        prompt = chat_prompt(text, BASE_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_span(prompt, evidence, tokenizer)
        if len(tokens) > lock["max_sequence_length"]:
            raise RuntimeError("V15 base prompt exceeds maximum")
        hidden = forward_to_layer(model, mx.array([tokens]), specification["extraction_layer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        base_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        base_lengths.append(len(tokens)); evidence_lengths.append(len(span)); hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every and (completed % args.progress_every == 0 or completed == total_new):
            print(f"v15 4B: extracted {completed}/{total_new}", file=sys.stderr, flush=True)
        mx.clear_cache()
    for text in arrays["nli_prompts"]:
        if text in reused_lookup:
            nli_features.append(reused_lookup[text])
            continue
        hypothesis = hypothesis_from_text(text)
        prompt = chat_prompt(text, NLI_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_hypothesis_span(prompt, hypothesis, tokenizer)
        if len(tokens) > lock["max_sequence_length"]:
            raise RuntimeError("V15 NLI prompt exceeds maximum")
        hidden = forward_to_layer(model, mx.array([tokens]), specification["extraction_layer"])[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        nli_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        nli_lengths.append(len(tokens)); hypothesis_lengths.append(len(span)); hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every and (completed % args.progress_every == 0 or completed == total_new):
            print(f"v15 4B: extracted {completed}/{total_new}", file=sys.stderr, flush=True)
        mx.clear_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "v15-full-features.npz"
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
        "schema_version": 15, "experiment": "v15_deduplicated_full_feature_extraction",
        "protocol_lock": str(lock_path), "protocol_lock_sha256": file_sha256(lock_path),
        "model": specification, "frozen": True, "adapter_path": None,
        "unique_base_prompts": len(arrays["base_prompts"]), "unique_nli_prompts": len(arrays["nli_prompts"]),
        "reused_v14_nli_prompts": len(reused_lookup), "new_model_forward_passes": total_new,
        "base_prompt_text_sha256": canonical_sha256(arrays["base_prompts"]),
        "nli_prompt_text_sha256": canonical_sha256(arrays["nli_prompts"]),
        "feature_dtype": "float32", "hidden_dtypes": sorted(hidden_dtypes), "truncated_prompts": 0,
        "feature_artifact": str(artifact_path), "feature_artifact_sha256": file_sha256(artifact_path),
        "data_access": lock["data_access"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
