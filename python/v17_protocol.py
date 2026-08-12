#!/usr/bin/env python3
"""Shared immutable-record and prompt utilities for V17."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from extract_v10_features_mlx import base_text, nli_text
from v10_protocol import TEMPORAL_ORDER, file_sha256, read_jsonl


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_v17_records(lock: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = Path(lock["manifest"])
    if file_sha256(manifest_path) != lock["manifest_sha256"]:
        raise RuntimeError("V17 sealed manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_sha256"] != lock["dataset_sha256"] or manifest["validation"]["errors"]:
        raise RuntimeError("V17 sealed dataset identity or validation changed")
    records: list[dict[str, Any]] = []
    for path_text, expected in lock["artifact_sha256"].items():
        path = Path(path_text)
        if file_sha256(path) != expected:
            raise RuntimeError(f"V17 sealed record artifact changed: {path}")
        records.extend(read_jsonl(path))
    if len(records) != lock["expected"]["records"]:
        raise RuntimeError("V17 sealed record cardinality changed")
    return records


def build_v17_prompts(records: list[dict[str, Any]]) -> dict[str, Any]:
    base_prompts: list[str] = []
    base_index: dict[str, int] = {}
    base_match: list[bool] = []
    base_temporal: list[int] = []
    nli_prompts: list[str] = []
    nli_index: dict[str, int] = {}
    pair_base: list[int] = []
    pair_nli: list[list[int]] = []
    pair_records: list[int] = []
    determinant_indices: list[int] = []
    evidence_indices: list[int] = []
    match_targets: list[bool] = []
    temporal_targets: list[int] = []
    current_targets: list[int] = []
    for record_index, record in enumerate(records):
        hypotheses = {
            value["determinant_id"]: value["statements"]
            for value in record["agent_input"]["state_hypotheses"]
        }
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            positive = next(index for index, unit in enumerate(record["evidence_units"]) if (
                unit["start"] == target["evidence_span"]["start"]
                and unit["end"] == target["evidence_span"]["end"]
            ))
            for evidence_index, _unit in enumerate(record["evidence_units"]):
                matched = evidence_index == positive
                base = base_text(record, determinant_index, evidence_index)
                temporal = TEMPORAL_ORDER.index(target["temporal_status"]) if matched else -1
                if base in base_index:
                    index = base_index[base]
                    if base_match[index] != matched or (matched and base_temporal[index] != temporal):
                        raise RuntimeError("V17 duplicate base prompt has conflicting targets")
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
    return {
        "base_prompts": base_prompts, "base_match": base_match, "base_temporal": base_temporal,
        "nli_prompts": nli_prompts, "pair_base_indices": pair_base, "pair_nli_indices": pair_nli,
        "pair_record_indices": pair_records, "determinant_indices": determinant_indices,
        "evidence_indices": evidence_indices, "match_targets": match_targets,
        "temporal_targets": temporal_targets, "current_value_targets": current_targets,
    }
