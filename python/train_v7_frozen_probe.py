#!/usr/bin/env python3
"""Train the single preregistered V7 linear probe on development only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from binary_metrics import evaluate_binary, fit_threshold
from evaluate_v5_challenge_mlx import safe_metrics, surface_invariance
from train_frozen_linear_probe import error_concentration, grouped_bootstrap, make_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v7-protocol-lock.json")
    parser.add_argument("--features", default="outputs/v7-frozen-probe/features")
    parser.add_argument("--output-dir", default="outputs/v7-frozen-probe/probe")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(root: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(root / f"{split}.npz", allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def rows_for(values: dict[str, np.ndarray], scores: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "id": str(record_id),
            "split_group": str(group),
            "surface_pair_id": str(pair_id),
            "surface_variant": str(surface),
            "evidence_pair_id": str(intervention),
            "evidence_intervention_kind": str(intervention_kind),
            "evidence_variant": str(evidence_variant),
            "mechanic": str(mechanic),
            "action_template": str(action_template),
            "gold_ambiguous": bool(gold),
            "score": float(score),
        }
        for (
            record_id,
            group,
            pair_id,
            surface,
            intervention,
            intervention_kind,
            evidence_variant,
            mechanic,
            action_template,
            gold,
            score,
        ) in zip(
            values["ids"],
            values["groups"],
            values["surface_pair_ids"],
            values["surface_variants"],
            values["evidence_intervention_ids"],
            values["evidence_intervention_kinds"],
            values["evidence_variants"],
            values["mechanics"],
            values["action_templates"],
            values["gold_ambiguous"],
            scores,
        )
    ]


def breakdown(rows: list[dict[str, Any]], key: str, threshold: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: safe_metrics(selected, threshold)
        for name, selected in sorted(groups.items())
    }


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text())
    implementation = lock["implementation"]["trainer"]
    if file_sha256(Path(implementation["path"])) != implementation["sha256"]:
        raise RuntimeError("V7 trainer changed after protocol lock")
    feature_root = Path(args.features)
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V7 features were not extracted under this protocol lock")
    if any(metadata[key] != 0 for key in (
        "untouched_mechanic_records_read",
        "prior_holdout_records_read",
        "v3_test_records_read",
    )):
        raise RuntimeError("V7 development extraction accessed a closed partition")
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"V7 probe result already exists: {result_path}")

    train = load_split(feature_root, "train")
    calibration = load_split(feature_root, "calibration")
    feature = lock["method"]["feature"]
    if train[feature].dtype != np.float32 or calibration[feature].dtype != np.float32:
        raise RuntimeError("V7 features are not float32")
    probe = make_probe(lock["method"]["c_value"], lock["method"]["seed"])
    train_gold = train["gold_ambiguous"].astype(bool)
    triplet_weights = np.full(len(train_gold), 1.0 / 3.0, dtype=np.float32)
    probe.fit(train[feature], train_gold, classifier__sample_weight=triplet_weights)
    train_scores = probe.decision_function(train[feature])
    calibration_scores = probe.decision_function(calibration[feature])
    if train_scores.dtype != np.float32 or calibration_scores.dtype != np.float32:
        raise RuntimeError("V7 decision scores are not float32")
    train_rows = rows_for(train, train_scores)
    calibration_rows = rows_for(calibration, calibration_scores)
    canonical = [row for row in calibration_rows if row["surface_variant"] == "canonical"]
    threshold_fit = fit_threshold(
        [row["gold_ambiguous"] for row in canonical],
        [row["score"] for row in canonical],
    )
    threshold = threshold_fit["threshold"]
    classifier = probe.named_steps["classifier"]
    scaler = probe.named_steps["standardize"]
    if classifier.coef_.dtype != np.float32:
        raise RuntimeError("V7 probe coefficients are not float32")

    output_dir.mkdir(parents=True, exist_ok=True)
    probe_path = output_dir / "selected-probe.npz"
    np.savez_compressed(
        probe_path,
        coefficient=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )
    (output_dir / "calibration.scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in calibration_rows)
    )
    canonical_gold = np.asarray([row["gold_ambiguous"] for row in canonical], dtype=bool)
    canonical_scores = np.asarray([row["score"] for row in canonical], dtype=float)
    canonical_groups = np.asarray([row["split_group"] for row in canonical])
    calibration_canonical = evaluate_binary(
        canonical_gold.tolist(), canonical_scores.tolist(), threshold
    )
    result = {
        "schema_version": 7,
        "experiment": "v7_fixed_frozen_probe_development",
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_metadata_sha256": file_sha256(metadata_path),
        "feature_artifact_sha256": {
            "train": file_sha256(feature_root / "train.npz"),
            "calibration": file_sha256(feature_root / "calibration.npz"),
        },
        "probe_training_number": 1,
        "model": lock["method"]["model"],
        "feature": feature,
        "c_value": lock["method"]["c_value"],
        "seed": lock["method"]["seed"],
        "threshold": threshold,
        "threshold_selection": "canonical_calibration_only",
        "invariance_training_target": "same binary label across complete surface triplets",
        "training": {
            "records": len(train_gold),
            "base_records": len(set(train["surface_pair_ids"].tolist())),
            "metrics": evaluate_binary(
                train_gold.tolist(), train_scores.astype(float).tolist(), threshold
            ),
            "error_concentration": error_concentration(train_rows, threshold),
        },
        "calibration_canonical": calibration_canonical,
        "calibration_canonical_grouped_bootstrap": grouped_bootstrap(
            canonical_gold,
            canonical_scores,
            canonical_groups,
            threshold,
            lock["method"]["bootstrap_samples"],
            lock["method"]["bootstrap_seed"],
        ),
        "calibration_by_surface": breakdown(calibration_rows, "surface_variant", threshold),
        "calibration_by_evidence": breakdown(canonical, "evidence_variant", threshold),
        "calibration_by_mechanic": breakdown(canonical, "mechanic", threshold),
        "calibration_by_action_template": breakdown(canonical, "action_template", threshold),
        "calibration_surface_invariance": surface_invariance(calibration_rows, threshold),
        "calibration_gate": {
            "minimum": lock["gates"]["minimumCalibrationCanonicalBalancedAccuracy"],
            "value": calibration_canonical["balanced_accuracy"],
            "passed": calibration_canonical["balanced_accuracy"]
            >= lock["gates"]["minimumCalibrationCanonicalBalancedAccuracy"],
        },
        "numerics": {
            "feature_dtype": str(train[feature].dtype),
            "coefficient_dtype": str(classifier.coef_.dtype),
            "decision_score_dtype": str(calibration_scores.dtype),
        },
        "probe_artifact": str(probe_path),
        "probe_artifact_sha256": file_sha256(probe_path),
        "untouched_mechanic_records_read": 0,
        "prior_holdout_records_read": 0,
        "v3_test_records_read": 0,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
