#!/usr/bin/env python3
"""Evaluate the locked deduplicated V15 frozen neuro-symbolic pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_v10_frozen import (
    build_pair_lookup, evaluate_ablation, evaluate_cell, gate_report, probe, save_pipeline,
)
from v10_protocol import RELATION_ORDER, VALUE_ORDER, file_sha256
from v14_protocol import load_records_from_manifest, primary_folds, zero_shot_operator_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v15-full-pipeline-lock.json")
    parser.add_argument("--features", default="outputs/v15-full-pipeline/features")
    parser.add_argument("--output-dir", default="outputs/v15-full-pipeline/evaluation")
    return parser.parse_args()


def unique_memberships(pair_base: np.ndarray, pair_records: np.ndarray, count: int) -> list[list[int]]:
    values = [set() for _ in range(count)]
    for base, record in zip(pair_base, pair_records):
        values[int(base)].add(int(record))
    return [sorted(value) for value in values]


def membership_mask(mask: np.ndarray, memberships: list[list[int]]) -> np.ndarray:
    return np.asarray([any(mask[index] for index in values) for values in memberships])


def unique_current_targets(
    pair_base: np.ndarray, current_targets: np.ndarray, count: int,
) -> np.ndarray:
    result = np.full(count, -1, dtype=np.int8)
    for base, target in zip(pair_base, current_targets):
        if target < 0:
            continue
        index = int(base)
        if result[index] >= 0 and result[index] != target:
            raise RuntimeError("V15 duplicate base prompt has conflicting current targets")
        result[index] = target
    return result


def nli_pairs_by_base(pair_base: np.ndarray, pair_nli: np.ndarray, count: int) -> np.ndarray:
    result = np.full((count, 2), -1, dtype=np.int32)
    for base, nli in zip(pair_base, pair_nli):
        index = int(base)
        if result[index, 0] >= 0 and not np.array_equal(result[index], nli):
            raise RuntimeError("V15 duplicate base prompt maps to conflicting NLI prompts")
        result[index] = nli
    if np.any(result < 0):
        raise RuntimeError("V15 base prompt lacks an NLI pair")
    return result


def group_scope(records: list[dict[str, Any]], fold: dict[str, Any]) -> np.ndarray:
    if fold["kind"] in {"context", "mechanic", "surface", "semantic_operator_diagnostic"}:
        return fold["evaluation"].copy()
    if fold["kind"] == "lexicon":
        return np.asarray([record["split"] == "evaluation" for record in records])
    held_operator = fold["name"].split(":")[1]
    return np.asarray([record["operator_family"] == held_operator for record in records])


def evaluate_one_fold(
    fold: dict[str, Any], fold_index: int, records: list[dict[str, Any]], arrays: dict[str, np.ndarray],
    memberships: list[list[int]], base_features: np.ndarray, polarity_features: np.ndarray,
    unique_current: np.ndarray, lock: dict[str, Any], output_dir: Path, prefix: str,
) -> dict[str, Any]:
    train_unique = membership_mask(fold["train"], memberships)
    base_match = arrays["unique_base_match_targets"].astype(bool)
    base_temporal = arrays["unique_base_temporal_targets"].astype(np.int8)
    positive_train = train_unique & base_match
    current_train = train_unique & (unique_current >= 0)
    match_model = probe(lock["c_value"], lock["seed"] + fold_index)
    temporal_model = probe(lock["c_value"], lock["seed"] + fold_index)
    polarity_model = probe(lock["c_value"], lock["seed"] + fold_index)
    match_model.fit(base_features[train_unique], base_match[train_unique])
    temporal_model.fit(base_features[positive_train], base_temporal[positive_train])
    polarity_model.fit(polarity_features[current_train], unique_current[current_train])
    unique_match_scores = match_model.decision_function(base_features).astype(np.float32)
    unique_temporal = temporal_model.predict(base_features).astype(np.int8)
    unique_current_predictions = polarity_model.predict(polarity_features).astype(np.int8)
    pair_base = arrays["pair_base_indices"].astype(np.int32)
    match_scores = unique_match_scores[pair_base]
    temporal_predictions = unique_temporal[pair_base]
    current_predictions = unique_current_predictions[pair_base]
    relation_predictions = np.asarray([
        [RELATION_ORDER.index("ENTAILED"), RELATION_ORDER.index("CONTRADICTED")]
        if value == VALUE_ORDER.index("active")
        else [RELATION_ORDER.index("CONTRADICTED"), RELATION_ORDER.index("ENTAILED")]
        for value in current_predictions
    ], dtype=np.int8)
    payload = {}
    save_pipeline("match", match_model, payload)
    save_pipeline("temporal", temporal_model, payload)
    save_pipeline("polarity", polarity_model, payload)
    artifact = output_dir / f"{prefix}-{fold['name'].replace(':', '-')}-heads.npz"
    np.savez_compressed(artifact, **payload)
    pair_records = arrays["pair_record_indices"].astype(np.int32)
    determinant_indices = arrays["determinant_indices"].astype(np.int8)
    evidence_indices = arrays["evidence_indices"].astype(np.int8)
    match_targets = arrays["match_targets"].astype(bool)
    pair_lookup = build_pair_lookup(pair_records, determinant_indices)
    overall = evaluate_cell(
        records, fold["evaluation"], pair_lookup, pair_records, evidence_indices,
        match_targets, match_scores, temporal_predictions, relation_predictions,
    )
    by_surface = {}
    for surface in sorted({record["state_lexicon_family"] for record in records}):
        mask = fold["evaluation"] & np.asarray([
            record["state_lexicon_family"] == surface for record in records
        ])
        if mask.any():
            by_surface[surface] = evaluate_cell(
                records, mask, pair_lookup, pair_records, evidence_indices,
                match_targets, match_scores, temporal_predictions, relation_predictions,
            )
    scope = group_scope(records, fold)
    group_metrics, _ = evaluate_ablation(
        records, scope, pair_lookup, evidence_indices, match_targets, match_scores,
        temporal_predictions, relation_predictions, False, False,
    )
    return {
        "kind": fold["kind"],
        "training_unique_base_prompts": int(train_unique.sum()),
        "training_unique_positive_prompts": int(positive_train.sum()),
        "training_unique_current_prompts": int(current_train.sum()),
        "head_artifact": str(artifact), "head_artifact_sha256": file_sha256(artifact),
        "overall": overall, "by_surface": by_surface,
        "group_scope": {
            "records": int(scope.sum()),
            "complete_intervention_groups": group_metrics["complete_intervention_groups"],
            "complete_intervention_group_accuracy": group_metrics["complete_intervention_group_accuracy"],
        },
    }


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    feature_root = Path(args.features)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"V15 result already exists: {result_path}")
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"V15 locked implementation changed: {path}")
    records = load_records_from_manifest(Path(lock["source"]["manifest"]))
    metadata = json.loads((feature_root / "metadata.json").read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V15 features do not share the lock")
    feature_path = Path(metadata["feature_artifact"])
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V15 feature artifact changed")
    with np.load(feature_path, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    if arrays["record_ids"].tolist() != [record["id"] for record in records]:
        raise RuntimeError("V15 record order differs")
    pair_base = arrays["pair_base_indices"].astype(np.int32)
    pair_records = arrays["pair_record_indices"].astype(np.int32)
    memberships = unique_memberships(pair_base, pair_records, 5022)
    unique_current = unique_current_targets(pair_base, arrays["current_value_targets"].astype(np.int8), 5022)
    nli_by_base = nli_pairs_by_base(pair_base, arrays["pair_nli_indices"].astype(np.int32), 5022)
    nli = arrays["nli_hypothesis_mean_features"].astype(np.float32)
    polarity_features = nli[nli_by_base[:, 0]] - nli[nli_by_base[:, 1]]
    base_features = arrays["base_span_features"].astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_results = {}
    primary = primary_folds(records)
    for index, fold in enumerate(primary):
        primary_results[fold["name"]] = evaluate_one_fold(
            fold, index, records, arrays, memberships, base_features, polarity_features,
            unique_current, lock, output_dir, "primary",
        )
    diagnostics = {}
    for index, fold in enumerate(zero_shot_operator_folds(records), start=len(primary)):
        diagnostics[fold["name"]] = evaluate_one_fold(
            fold, index, records, arrays, memberships, base_features, polarity_features,
            unique_current, lock, output_dir, "diagnostic",
        )
    transfer = {name: value for name, value in primary_results.items() if name != "context"}
    gates = gate_report(transfer, lock["gates"])
    decision = (
        "full_operator_supported_frozen_pipeline_passes_authorize_separate_final_mechanic_protocol"
        if gates["passed"] else "full_pipeline_transfer_fails_decompose_before_any_adaptation"
    )
    result = {
        "schema_version": 15, "experiment": "v15_operator_supported_frozen_full_pipeline",
        "protocol_lock": str(lock_path), "protocol_lock_sha256": file_sha256(lock_path),
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "primary_folds": primary_results, "zero_shot_operator_diagnostics": diagnostics,
        "primary_transfer_gates": gates, "context_gating": False,
        "decision": decision, "lora_authorized": False, "final_mechanic_authorized": False,
        "separate_final_mechanic_protocol_authorized": gates["passed"],
        "data_access": lock["data_access"],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": decision, "primary_transfer_gates": gates,
        "context": primary_results["context"]["overall"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
