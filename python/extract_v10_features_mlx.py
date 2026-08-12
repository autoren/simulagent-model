#!/usr/bin/env python3
"""Extract all locked frozen V10 mean, evidence-span, and NLI-final features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

from extract_v6_development_features_mlx import forward_layer_six
from v10_protocol import RELATION_ORDER, TEMPORAL_ORDER, VALUE_ORDER, file_sha256, load_locked_records


BASE_SYSTEM_PROMPT = " ".join((
    "Encode whether one evidence excerpt concerns the queried transition determinant.",
    "Preserve temporal status and current-state polarity without inferring a transition outcome.",
    "Do not generate an answer.",
))
NLI_SYSTEM_PROMPT = " ".join((
    "Compare one evidence excerpt with one candidate current-state hypothesis.",
    "Preserve whether the evidence entails, contradicts, or leaves the hypothesis unknown.",
    "Do not infer transition outcomes and do not generate an answer.",
))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v10-frozen-lock.json")
    parser.add_argument("--output-dir", default="outputs/v10-frozen/features")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def base_text(record: dict[str, Any], determinant_index: int, evidence_index: int) -> str:
    determinants = record["agent_input"]["transition_determinants"]
    return "\n".join((
        f"Candidate action: {record['agent_input']['candidate_action']}",
        "Listed determinants: " + json.dumps(determinants, sort_keys=True, separators=(",", ":")),
        "Queried determinant: " + json.dumps(determinants[determinant_index], sort_keys=True, separators=(",", ":")),
        f"Evidence excerpt: {record['evidence_units'][evidence_index]['text']}",
    ))


def nli_text(record: dict[str, Any], determinant_index: int, evidence_index: int, hypothesis: str) -> str:
    determinant = record["agent_input"]["transition_determinants"][determinant_index]
    return "\n".join((
        f"Candidate action: {record['agent_input']['candidate_action']}",
        "Queried determinant: " + json.dumps(determinant, sort_keys=True, separators=(",", ":")),
        f"Evidence excerpt: {record['evidence_units'][evidence_index]['text']}",
        f"Current-state hypothesis: {hypothesis}",
    ))


def chat_prompt(text: str, system_prompt: str, tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prompt_tokens_and_span(prompt: str, evidence: str, tokenizer: Any) -> tuple[list[int], list[int]]:
    encoded = tokenizer._tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded["input_ids"]
    if tokens != tokenizer.encode(prompt):
        raise RuntimeError("V10 offset tokenizer ids differ from model tokenizer ids")
    start = prompt.rfind(evidence)
    if start < 0:
        raise RuntimeError("V10 evidence excerpt is absent from its rendered prompt")
    end = start + len(evidence)
    span = [index for index, (left, right) in enumerate(encoded["offset_mapping"]) if left < end and right > start]
    if not span:
        raise RuntimeError("V10 evidence excerpt maps to no tokens")
    return tokens, span


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"V10 frozen features already exist: {metadata_path}")
    lock = json.loads(lock_path.read_text())
    if not lock["pre_model_audit"]["gates_passed"]:
        raise RuntimeError("V10 pre-model gates did not authorize extraction")
    records = load_locked_records(lock)

    base_prompts: list[str] = []
    base_evidence: list[str] = []
    base_index: dict[str, int] = {}
    nli_prompts: list[str] = []
    nli_index: dict[str, int] = {}
    pair_base_indices: list[int] = []
    pair_nli_indices: list[list[int]] = []
    pair_record_indices: list[int] = []
    determinant_indices: list[int] = []
    evidence_indices: list[int] = []
    match_targets: list[bool] = []
    temporal_targets: list[int] = []
    current_value_targets: list[int] = []
    relation_targets: list[list[int]] = []
    for record_index, record in enumerate(records):
        hypothesis_by_id = {value["determinant_id"]: value["statements"] for value in record["agent_input"]["state_hypotheses"]}
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            positive = next(index for index, unit in enumerate(record["evidence_units"]) if (
                unit["start"] == target["evidence_span"]["start"] and unit["end"] == target["evidence_span"]["end"]
            ))
            hypotheses = hypothesis_by_id[target["determinant_id"]]
            for evidence_index, unit in enumerate(record["evidence_units"]):
                text = base_text(record, determinant_index, evidence_index)
                if text not in base_index:
                    base_index[text] = len(base_prompts)
                    base_prompts.append(text)
                    base_evidence.append(unit["text"])
                nli_values = []
                for statement in hypotheses:
                    value = nli_text(record, determinant_index, evidence_index, statement)
                    if value not in nli_index:
                        nli_index[value] = len(nli_prompts)
                        nli_prompts.append(value)
                    nli_values.append(nli_index[value])
                matched = evidence_index == positive
                pair_base_indices.append(base_index[text])
                pair_nli_indices.append(nli_values)
                pair_record_indices.append(record_index)
                determinant_indices.append(determinant_index)
                evidence_indices.append(evidence_index)
                match_targets.append(matched)
                temporal_targets.append(TEMPORAL_ORDER.index(target["temporal_status"]) if matched else -1)
                current_value_targets.append(VALUE_ORDER.index(target["current_value"]) if matched and target["current_value"] else -1)
                relation_targets.append(
                    [RELATION_ORDER.index(value) for value in target["hypothesis_relations"]] if matched else [-1, -1]
                )

    model, tokenizer = load(lock["protocol"]["model"])
    model.eval()
    mean_features: list[np.ndarray] = []
    span_features: list[np.ndarray] = []
    base_lengths: list[int] = []
    span_lengths: list[int] = []
    hidden_dtypes: set[str] = set()
    total = len(base_prompts) + len(nli_prompts)
    completed = 0
    for text, evidence in zip(base_prompts, base_evidence):
        prompt = chat_prompt(text, BASE_SYSTEM_PROMPT, tokenizer)
        tokens, span = prompt_tokens_and_span(prompt, evidence, tokenizer)
        if len(tokens) > lock["protocol"]["maxSequenceLength"]:
            raise RuntimeError(f"V10 base prompt exceeds maximum: {len(tokens)}")
        hidden = forward_layer_six(model, mx.array([tokens]))[0]
        mx.eval(hidden)
        hidden32 = hidden.astype(mx.float32)
        mean_features.append(np.asarray(mx.mean(hidden32, axis=0), dtype=np.float32))
        span_features.append(np.asarray(mx.mean(hidden32[mx.array(span)], axis=0), dtype=np.float32))
        base_lengths.append(len(tokens))
        span_lengths.append(len(span))
        hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == total):
            print(f"v10 frozen: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()

    nli_features: list[np.ndarray] = []
    nli_lengths: list[int] = []
    for text in nli_prompts:
        prompt = chat_prompt(text, NLI_SYSTEM_PROMPT, tokenizer)
        tokens = tokenizer.encode(prompt)
        if len(tokens) > lock["protocol"]["maxSequenceLength"]:
            raise RuntimeError(f"V10 NLI prompt exceeds maximum: {len(tokens)}")
        hidden = forward_layer_six(model, mx.array([tokens]))[0]
        mx.eval(hidden)
        nli_features.append(np.asarray(hidden[-1].astype(mx.float32), dtype=np.float32))
        nli_lengths.append(len(tokens))
        hidden_dtypes.add(str(hidden.dtype))
        completed += 1
        if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == total):
            print(f"v10 frozen: extracted {completed}/{total}", file=sys.stderr, flush=True)
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "v10-features.npz"
    np.savez_compressed(
        artifact_path,
        record_ids=np.asarray([record["id"] for record in records]),
        base_prompts=np.asarray(base_prompts),
        nli_prompts=np.asarray(nli_prompts),
        base_mean_features=np.stack(mean_features).astype(np.float32),
        base_span_features=np.stack(span_features).astype(np.float32),
        nli_final_features=np.stack(nli_features).astype(np.float32),
        pair_base_indices=np.asarray(pair_base_indices, dtype=np.int32),
        pair_nli_indices=np.asarray(pair_nli_indices, dtype=np.int32),
        pair_record_indices=np.asarray(pair_record_indices, dtype=np.int32),
        determinant_indices=np.asarray(determinant_indices, dtype=np.int8),
        evidence_indices=np.asarray(evidence_indices, dtype=np.int8),
        match_targets=np.asarray(match_targets, dtype=np.uint8),
        temporal_targets=np.asarray(temporal_targets, dtype=np.int8),
        current_value_targets=np.asarray(current_value_targets, dtype=np.int8),
        relation_targets=np.asarray(relation_targets, dtype=np.int8),
    )
    metadata = {
        "schema_version": 10,
        "experiment": "v10_frozen_multi_representation_extraction",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "model": lock["protocol"]["model"],
        "layer": lock["protocol"]["layer"],
        "frozen": True,
        "adapter_path": None,
        "records": len(records),
        "pair_examples": len(pair_base_indices),
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
        "temporal_order": TEMPORAL_ORDER,
        "relation_order": RELATION_ORDER,
        "value_order": VALUE_ORDER,
        "data_access": lock["data_access"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
