#!/usr/bin/env python3
"""Extract locked frozen pair representations for V9r2 evidence grounding."""

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

from extract_v6_development_features_mlx import forward_layer_six


SYSTEM_PROMPT = " ".join((
    "Encode whether one natural-language evidence excerpt concerns a queried transition determinant.",
    "Preserve its current value constraints, negation, uncertainty, conflicts, and temporal status.",
    "Do not infer transition outcomes and do not generate an answer.",
))
VALUE_ORDER = ["inactive", "active", "active|inactive"]
TEMPORAL_ORDER = ["CURRENT", "UNKNOWN_CURRENT", "STALE_ONLY", "CONFLICTING_CURRENT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v9-frozen-grounding-lock.json")
    parser.add_argument("--output-dir", default="outputs/v9-frozen-grounding/features")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def grounding_text(record: dict[str, Any], determinant_index: int, evidence_index: int) -> str:
    determinants = record["agent_input"]["transition_determinants"]
    return "\n".join((
        f"Candidate action: {record['agent_input']['candidate_action']}",
        "Listed determinants: " + json.dumps(determinants, sort_keys=True, separators=(",", ":")),
        "Queried determinant: " + json.dumps(determinants[determinant_index], sort_keys=True, separators=(",", ":")),
        f"Evidence excerpt: {record['evidence_units'][evidence_index]['text']}",
    ))


def prompt_tokens(text: str, tokenizer: Any) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(prompt)


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"V9 frozen features already exist: {metadata_path}")
    lock = json.loads(lock_path.read_text())
    if not lock["pre_model_audit"]["gates_passed"]:
        raise RuntimeError("V9 pre-model gates did not authorize feature extraction")
    manifest_path = Path(lock["dataset_manifest"])
    if file_sha256(manifest_path) != lock["dataset_manifest_sha256"]:
        raise RuntimeError("V9 manifest changed after frozen-grounding lock")
    records: list[dict[str, Any]] = []
    for relative, expected in lock["dataset_artifact_sha256"].items():
        path = manifest_path.parent / relative
        if file_sha256(path) != expected:
            raise RuntimeError(f"V9 artifact changed after lock: {relative}")
        records.extend(read_jsonl(path))

    unique_texts: list[str] = []
    text_index: dict[str, int] = {}
    pair_feature_indices = []
    pair_record_indices = []
    determinant_indices = []
    evidence_indices = []
    match_targets = []
    value_targets = []
    temporal_targets = []
    for record_index, record in enumerate(records):
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            positive = next(index for index, unit in enumerate(record["evidence_units"]) if (
                unit["start"] == target["evidence_span"]["start"]
                and unit["end"] == target["evidence_span"]["end"]
            ))
            value_class = VALUE_ORDER.index("|".join(sorted(target["allowed_values"])))
            temporal_class = TEMPORAL_ORDER.index(target["temporal_status"])
            for evidence_index in range(len(record["evidence_units"])):
                text = grounding_text(record, determinant_index, evidence_index)
                if text not in text_index:
                    text_index[text] = len(unique_texts)
                    unique_texts.append(text)
                matched = evidence_index == positive
                pair_feature_indices.append(text_index[text])
                pair_record_indices.append(record_index)
                determinant_indices.append(determinant_index)
                evidence_indices.append(evidence_index)
                match_targets.append(matched)
                value_targets.append(value_class if matched else -1)
                temporal_targets.append(temporal_class if matched else -1)

    model, tokenizer = load(lock["protocol"]["model"])
    model.eval()
    features = []
    lengths = []
    hidden_dtypes: set[str] = set()
    for index, text in enumerate(unique_texts, start=1):
        tokens = prompt_tokens(text, tokenizer)
        if len(tokens) > lock["protocol"]["maxSequenceLength"]:
            raise RuntimeError(
                f"V9 prompt exceeds locked maximum: {len(tokens)} > {lock['protocol']['maxSequenceLength']}"
            )
        hidden = forward_layer_six(model, mx.array([tokens]))[0]
        mx.eval(hidden)
        features.append(np.asarray(mx.mean(hidden.astype(mx.float32), axis=0), dtype=np.float32))
        lengths.append(len(tokens))
        hidden_dtypes.add(str(hidden.dtype))
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(unique_texts)):
            print(f"v9 grounding: extracted {index}/{len(unique_texts)}", file=sys.stderr, flush=True)
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "grounding-features.npz"
    np.savez_compressed(
        artifact_path,
        record_ids=np.asarray([record["id"] for record in records]),
        unique_prompts=np.asarray(unique_texts),
        unique_features=np.stack(features).astype(np.float32),
        pair_feature_indices=np.asarray(pair_feature_indices, dtype=np.int32),
        pair_record_indices=np.asarray(pair_record_indices, dtype=np.int32),
        determinant_indices=np.asarray(determinant_indices, dtype=np.int8),
        evidence_indices=np.asarray(evidence_indices, dtype=np.int8),
        match_targets=np.asarray(match_targets, dtype=np.uint8),
        value_targets=np.asarray(value_targets, dtype=np.int8),
        temporal_targets=np.asarray(temporal_targets, dtype=np.int8),
    )
    metadata = {
        "schema_version": 9,
        "revision": 2,
        "experiment": "v9_frozen_grounding_feature_extraction",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "model": lock["protocol"]["model"],
        "feature": lock["protocol"]["feature"],
        "frozen": True,
        "adapter_path": None,
        "records": len(records),
        "pair_examples": len(pair_feature_indices),
        "unique_prompts": len(unique_texts),
        "feature_dtype": "float32",
        "hidden_dtypes": sorted(hidden_dtypes),
        "minimum_prompt_tokens": min(lengths),
        "maximum_prompt_tokens": max(lengths),
        "truncated_prompts": 0,
        "feature_artifact": str(artifact_path),
        "feature_artifact_sha256": file_sha256(artifact_path),
        "value_order": VALUE_ORDER,
        "temporal_order": TEMPORAL_ORDER,
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
            "final_v9_mechanic_records_read": 0,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
