#!/usr/bin/env python3
"""Evaluate the locked V13 token-local relation readouts."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from evaluate_v12_joint_readout import (
    conditional_probe,
    gate_report,
    metrics,
    primary_probe,
    save_pipeline,
)
from v10_protocol import file_sha256, folds, load_locked_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v13-token-local-lock.json")
    parser.add_argument("--features", default="outputs/v13-token-local/features")
    parser.add_argument("--output-dir", default="outputs/v13-token-local/evaluation")
    return parser.parse_args()


def features_for_head(
    name: str,
    last_active: np.ndarray,
    last_inactive: np.ndarray,
    mean_active: np.ndarray,
    mean_inactive: np.ndarray,
) -> np.ndarray:
    if name == "hypothesis_last_linear":
        return last_active - last_inactive
    if name == "hypothesis_mean_linear":
        return mean_active - mean_inactive
    if name == "hypothesis_token_joint_mlp":
        return np.concatenate((
            last_active - last_inactive,
            mean_active - mean_inactive,
            (last_active + last_inactive) * 0.5,
        ), axis=1)
    raise ValueError(f"unknown V13 head: {name}")


def evaluate_head(
    name: str,
    specification: dict[str, Any],
    records: list[dict[str, Any]],
    all_folds: list[dict[str, Any]],
    eligible_records: np.ndarray,
    targets: np.ndarray,
    features: np.ndarray,
    swapped: np.ndarray,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    surface_by_record = np.asarray([record["state_lexicon_family"] for record in records])
    results = {}
    for fold_index, fold in enumerate(all_folds):
        training = fold["train"][eligible_records]
        evaluation = fold["evaluation"][eligible_records]
        model = (
            primary_probe(specification, seed + fold_index)
            if name.endswith("_linear")
            else conditional_probe(specification, seed + fold_index)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features[training], targets[training])
        scores = model.predict_proba(features)[:, 1].astype(np.float32)
        swapped_scores = model.predict_proba(swapped)[:, 1].astype(np.float32)
        artifact = output_dir / f"{name}-{fold['name'].replace(':', '-')}-head.npz"
        save_pipeline(artifact, "signed_difference_linear" if name.endswith("_linear") else "joint_mlp", model)
        by_surface = {}
        for surface in sorted(set(surface_by_record.tolist())):
            mask = evaluation & (surface_by_record[eligible_records] == surface)
            if mask.any():
                by_surface[surface] = metrics(targets[mask], scores[mask], swapped_scores[mask])
        results[fold["name"]] = {
            "kind": fold["kind"],
            "training_examples": int(training.sum()),
            "evaluation_examples": int(evaluation.sum()),
            "convergence_warnings": sum(issubclass(item.category, ConvergenceWarning) for item in caught),
            "head_artifact": str(artifact),
            "head_artifact_sha256": file_sha256(artifact),
            "overall": metrics(targets[evaluation], scores[evaluation], swapped_scores[evaluation]),
            "by_surface": by_surface,
        }
    return results


def select_decision(heads: dict[str, Any]) -> str:
    if heads["hypothesis_last_linear"]["gates"]["passed"]:
        return "token_local_linear_relation_accessible_repair_temporal_with_hypothesis_last"
    if heads["hypothesis_mean_linear"]["gates"]["passed"]:
        return "token_local_linear_relation_accessible_repair_temporal_with_hypothesis_mean"
    if heads.get("hypothesis_token_joint_mlp", {}).get("gates", {}).get("passed"):
        return "token_local_nonlinear_relation_accessible_repair_temporal"
    return "token_local_frozen_readout_insufficient_stop_probes_redesign_supervision"


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    feature_root = Path(args.features)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"V13 result already exists: {result_path}")
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"V13 locked implementation changed: {path}")
    v10_lock = json.loads(Path(lock["source_v10"]["frozen_lock"]).read_text())
    records = load_locked_records(v10_lock)
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V13 features do not share the locked protocol")
    feature_path = Path(metadata["feature_artifact"])
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V13 feature artifact changed")

    with np.load(lock["source_v10"]["feature_artifact"], allow_pickle=False) as reference:
        pair_records = reference["pair_record_indices"].astype(np.int32)
        pair_nli = reference["pair_nli_indices"].astype(np.int32)
        match_targets = reference["match_targets"].astype(bool)
        current_targets = reference["current_value_targets"].astype(np.int8)
        record_ids = reference["record_ids"].tolist()
    if record_ids != [record["id"] for record in records]:
        raise RuntimeError("V13 records and feature indices differ")
    eligible = np.flatnonzero(match_targets & (current_targets >= 0))
    eligible_records = pair_records[eligible]
    eligible_nli = pair_nli[eligible]
    targets = current_targets[eligible]
    if np.bincount(targets, minlength=2).tolist() != [3690, 3690]:
        raise RuntimeError("V13 eligible targets differ from V12")

    with np.load(feature_path, allow_pickle=False) as values:
        last = values["hypothesis_last_features"].astype(np.float32)
        mean = values["hypothesis_mean_features"].astype(np.float32)
    if last.shape != (6984, lock["model"]["hidden_size"]) or mean.shape != last.shape:
        raise RuntimeError("V13 feature shapes differ from lock")
    last_active, last_inactive = last[eligible_nli[:, 0]], last[eligible_nli[:, 1]]
    mean_active, mean_inactive = mean[eligible_nli[:, 0]], mean[eligible_nli[:, 1]]
    all_folds = folds(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    heads = {}
    for specification in lock["linear_heads"]:
        name = specification["name"]
        values = features_for_head(name, last_active, last_inactive, mean_active, mean_inactive)
        swapped = features_for_head(name, last_inactive, last_active, mean_inactive, mean_active)
        fold_results = evaluate_head(
            name, specification, records, all_folds, eligible_records, targets,
            values, swapped, output_dir, lock["seed"],
        )
        heads[name] = {"folds": fold_results, "gates": gate_report(fold_results, lock["gates"])}

    conditional_ran = not any(value["gates"]["passed"] for value in heads.values())
    if conditional_ran:
        specification = lock["conditional_head"]
        name = specification["name"]
        values = features_for_head(name, last_active, last_inactive, mean_active, mean_inactive)
        swapped = features_for_head(name, last_inactive, last_active, mean_inactive, mean_active)
        fold_results = evaluate_head(
            name, specification, records, all_folds, eligible_records, targets,
            values, swapped, output_dir, lock["seed"],
        )
        heads[name] = {"folds": fold_results, "gates": gate_report(fold_results, lock["gates"])}

    result = {
        "schema_version": 13,
        "experiment": "v13_4b_token_local_relation_diagnostic",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "eligible_examples": int(targets.size),
        "class_counts": np.bincount(targets, minlength=2).tolist(),
        "heads": heads,
        "conditional_head_ran": conditional_ran,
        "decision": select_decision(heads),
        "lora_authorized": False,
        "final_mechanic_authorized": False,
        "data_access": lock["data_access"],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "conditional_head_ran": conditional_ran,
        "gates": {name: value["gates"] for name, value in heads.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
