#!/usr/bin/env python3
"""Extract locked component embeddings for the V8 action-conditioned head."""

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
    "Encode one component of a structured transition-reasoning problem.",
    "Preserve the action, determinant role, evidence status, value, and transition-table relationships.",
    "Do not generate an answer.",
))

STATUS_ORDER = [
    "RESOLVED_TRUE",
    "RESOLVED_FALSE",
    "UNRESOLVED_OUTCOME_SENSITIVE",
    "UNRESOLVED_OUTCOME_INVARIANT",
    "IRRELEVANT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v8-structured-head-lock.json")
    parser.add_argument("--output-dir", default="outputs/v8-structured-head/components")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def component_prompt(text: str, tokenizer: Any) -> list[int]:
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


def component_texts(record: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    value = record["agent_input"]
    schema = value["action_dependency_schema"]
    action = "ACTION " + json.dumps({
        "candidate": value["candidate_action"],
        "schema_action": schema["candidate_action"],
    }, sort_keys=True, separators=(",", ":"))
    table = "DEPENDENCY_TABLE " + json.dumps(schema, sort_keys=True, separators=(",", ":"))
    determinant_ids = {item["id"] for item in schema["transition_determinants"]}
    roles = []
    evidence = []
    for fact in value["evidence_ledger"]:
        roles.append("ROLE " + json.dumps({
            "id": fact["id"],
            "role": fact["role"],
            "listed_transition_determinant": fact["id"] in determinant_ids,
        }, sort_keys=True, separators=(",", ":")))
        evidence.append("EVIDENCE " + json.dumps(fact, sort_keys=True, separators=(",", ":")))
    return action, table, roles, evidence


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"V8 structured components already exist: {metadata_path}")
    lock = json.loads(lock_path.read_text())
    if not lock["stage3_result"]["gates_passed"]:
        raise RuntimeError("V8 Stage 3 did not authorize the structured head")
    manifest_path = Path(lock["dataset_manifest"])
    if file_sha256(manifest_path) != lock["dataset_manifest_sha256"]:
        raise RuntimeError("V8 manifest changed after structured-head lock")
    manifest = json.loads(manifest_path.read_text())
    dataset_root = manifest_path.parent
    records: list[dict[str, Any]] = []
    for split in ("train", "calibration"):
        relative = f"records/{split}.jsonl"
        path = dataset_root / relative
        if file_sha256(path) != lock["dataset_artifact_sha256"][relative]:
            raise RuntimeError(f"V8 {split} changed after structured-head lock")
        records.extend(read_jsonl(path))

    stage3_metadata_path = Path(lock["stage3_features"]["metadata"])
    stage3_metadata = json.loads(stage3_metadata_path.read_text())
    stage3_feature_path = Path(stage3_metadata["feature_artifact"])
    if file_sha256(stage3_feature_path) != lock["stage3_features"]["artifact_sha256"]:
        raise RuntimeError("V8 Stage 3 global features changed")
    with np.load(stage3_feature_path, allow_pickle=False) as values:
        if values["ids"].tolist() != [record["id"] for record in records]:
            raise RuntimeError("V8 records and Stage 3 feature order differ")

    all_texts: list[str] = []
    per_record: list[tuple[str, str, list[str], list[str]]] = []
    for record in records:
        parts = component_texts(record)
        per_record.append(parts)
        all_texts.extend([parts[0], parts[1], *parts[2], *parts[3]])
    vocabulary = list(dict.fromkeys(all_texts))
    index_by_text = {text: index for index, text in enumerate(vocabulary)}

    model, tokenizer = load(lock["method"]["model"])
    model.eval()
    embeddings: list[np.ndarray] = []
    lengths: list[int] = []
    hidden_dtypes: set[str] = set()
    truncated = 0
    for index, text in enumerate(vocabulary, start=1):
        tokens = component_prompt(text, tokenizer)
        if len(tokens) > lock["method"]["max_component_tokens"]:
            tokens = tokens[-lock["method"]["max_component_tokens"]:]
            truncated += 1
        hidden = forward_layer_six(model, mx.array([tokens]))[0]
        mx.eval(hidden)
        embeddings.append(np.asarray(mx.mean(hidden.astype(mx.float32), axis=0), dtype=np.float32))
        lengths.append(len(tokens))
        hidden_dtypes.add(str(hidden.dtype))
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(vocabulary)):
            print(f"v8 components: extracted {index}/{len(vocabulary)}", file=sys.stderr, flush=True)
        mx.clear_cache()

    action_indices = []
    table_indices = []
    role_indices = []
    evidence_indices = []
    status_targets = []
    status_index = {name: index for index, name in enumerate(STATUS_ORDER)}
    for record, (action, table, roles, evidence) in zip(records, per_record):
        if len(roles) != 7 or len(evidence) != 7 or len(record["target"]["determinant_ledger"]) != 7:
            raise RuntimeError("V8 structured head requires exactly seven evidence rows")
        action_indices.append(index_by_text[action])
        table_indices.append(index_by_text[table])
        role_indices.append([index_by_text[text] for text in roles])
        evidence_indices.append([index_by_text[text] for text in evidence])
        schema_ids = [item["id"] for item in record["agent_input"]["action_dependency_schema"]["transition_determinants"]]
        canonical_targets = record["target"]["determinant_ledger"]
        relevant_statuses = {
            model_id: status_index[target["status"]]
            for model_id, target in zip(schema_ids, canonical_targets[:len(schema_ids)])
        }
        status_targets.append([
            relevant_statuses.get(fact["id"], status_index["IRRELEVANT"])
            for fact in record["agent_input"]["evidence_ledger"]
        ])

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "components.npz"
    np.savez_compressed(
        artifact_path,
        vocabulary=np.asarray(vocabulary),
        embeddings=np.stack(embeddings).astype(np.float32),
        action_indices=np.asarray(action_indices, dtype=np.int32),
        table_indices=np.asarray(table_indices, dtype=np.int32),
        role_indices=np.asarray(role_indices, dtype=np.int32),
        evidence_indices=np.asarray(evidence_indices, dtype=np.int32),
        status_targets=np.asarray(status_targets, dtype=np.int32),
    )
    metadata = {
        "schema_version": 8,
        "experiment": "v8_structured_component_extraction",
        "structured_head_lock": str(lock_path),
        "structured_head_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "stage3_feature_artifact_sha256": lock["stage3_features"]["artifact_sha256"],
        "model": lock["method"]["model"],
        "layer": 6,
        "pooling": "mean",
        "records": len(records),
        "evidence_rows_per_record": 7,
        "unique_components": len(vocabulary),
        "artifact": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "feature_dtype": "float32",
        "hidden_dtypes": sorted(hidden_dtypes),
        "minimum_component_tokens": min(lengths),
        "maximum_component_tokens": max(lengths),
        "truncated_components": truncated,
        "status_order": STATUS_ORDER,
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
