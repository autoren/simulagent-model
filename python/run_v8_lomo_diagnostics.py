#!/usr/bin/env python3
"""Run preregistered V8 leave-one-mechanic-out mean and pair-difference probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from binary_metrics import evaluate_binary, fit_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v8-development-lock.json")
    parser.add_argument("--features", default="outputs/v8-frozen-diagnostics/features")
    parser.add_argument("--output", default="outputs/v8-frozen-diagnostics/lomo-result.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classifier(c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=4000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def pair_indices(data: dict[str, np.ndarray], mask: np.ndarray) -> list[tuple[int, int]]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index in np.flatnonzero(mask):
        if data["intervention_kinds"][index] != "oracle_label_flip":
            continue
        key = (str(data["intervention_group_ids"][index]), str(data["surface_variants"][index]))
        groups.setdefault(key, []).append(int(index))
    result: list[tuple[int, int]] = []
    for key, indices in groups.items():
        ambiguous = [index for index in indices if bool(data["gold_ambiguous"][index])]
        identifiable = [index for index in indices if not bool(data["gold_ambiguous"][index])]
        if len(ambiguous) != 1 or len(identifiable) != 1:
            raise RuntimeError(f"Malformed V8 pair {key}: {indices}")
        result.append((ambiguous[0], identifiable[0]))
    return sorted(result)


def direction_metrics(
    pairs: list[tuple[int, int]],
    scores: np.ndarray,
    resolved_values: np.ndarray,
) -> dict[str, Any]:
    margins = np.asarray([scores[ambiguous] - scores[identifiable] for ambiguous, identifiable in pairs])
    values = np.asarray([bool(resolved_values[ambiguous]) for ambiguous, _ in pairs])
    return {
        "pairs": len(pairs),
        "accuracy": float(np.mean(margins > 0)),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "resolved_true_accuracy": float(np.mean(margins[values] > 0)),
        "resolved_false_accuracy": float(np.mean(margins[~values] > 0)),
        "resolved_true_pairs": int(values.sum()),
        "resolved_false_pairs": int((~values).sum()),
    }


def pair_probe_metrics(
    pairs: list[tuple[int, int]],
    features: np.ndarray,
    model: Any,
    resolved_values: np.ndarray,
) -> dict[str, Any]:
    deltas = np.stack([features[ambiguous] - features[identifiable] for ambiguous, identifiable in pairs])
    scores = model.decision_function(deltas)
    values = np.asarray([bool(resolved_values[ambiguous]) for ambiguous, _ in pairs])
    return {
        "pairs": len(pairs),
        "accuracy": float(np.mean(scores > 0)),
        "mean_margin": float(np.mean(scores)),
        "median_margin": float(np.median(scores)),
        "resolved_true_accuracy": float(np.mean(scores[values] > 0)),
        "resolved_false_accuracy": float(np.mean(scores[~values] > 0)),
        "resolved_true_pairs": int(values.sum()),
        "resolved_false_pairs": int((~values).sum()),
    }


def gate_report(folds: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    pair_values = [
        metrics["pair_difference"][surface]["accuracy"]
        for metrics in folds.values()
        for surface in metrics["pair_difference"]
    ]
    pointwise_values = [
        metrics["pointwise_score_direction"][surface]["accuracy"]
        for metrics in folds.values()
        for surface in metrics["pointwise_score_direction"]
    ]
    checks = [
        {
            "name": "minimum_fold_surface_pair_difference_accuracy",
            "value": min(pair_values),
            "minimum": gates["minimumEveryFoldPairDifferenceAccuracy"],
        },
        {
            "name": "mean_fold_surface_pair_difference_accuracy",
            "value": float(np.mean(pair_values)),
            "minimum": gates["minimumMeanPairDifferenceAccuracy"],
        },
        {
            "name": "minimum_fold_surface_pointwise_score_direction",
            "value": min(pointwise_values),
            "minimum": gates["minimumEveryFoldPointwiseScoreDirection"],
        },
    ]
    for check in checks:
        check["passed"] = check["value"] >= check["minimum"]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    feature_root = Path(args.features)
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError(f"V8 diagnostic result already exists: {output_path}")
    lock = json.loads(lock_path.read_text())
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    feature_path = Path(metadata["feature_artifact"])
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V8 features do not share the active protocol lock")
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V8 feature artifact changed after extraction")
    with np.load(feature_path, allow_pickle=False) as values:
        data = {key: values[key] for key in values.files}
    features = data[lock["method"]["feature"]].astype(np.float32)
    mechanics = list(lock["mechanics"])
    surfaces = ["canonical", "entity_renamed", "paraphrased"]
    folds: dict[str, Any] = {}
    for fold_number, heldout in enumerate(mechanics):
        other = data["mechanics"] != heldout
        train_mask = other & (data["splits"] == "train")
        calibration_mask = other & (data["splits"] == "calibration") & (data["surface_variants"] == "canonical")
        heldout_mask = data["mechanics"] == heldout

        pointwise = classifier(lock["method"]["c_value"], lock["method"]["seed"] + fold_number)
        pointwise.fit(features[train_mask], data["gold_ambiguous"][train_mask].astype(bool))
        calibration_scores = pointwise.decision_function(features[calibration_mask])
        threshold_report = fit_threshold(
            data["gold_ambiguous"][calibration_mask].astype(bool).tolist(),
            calibration_scores.tolist(),
        )
        threshold = threshold_report["threshold"]
        all_scores = pointwise.decision_function(features)

        training_pairs = pair_indices(data, train_mask)
        training_deltas = np.stack([
            features[ambiguous] - features[identifiable]
            for ambiguous, identifiable in training_pairs
        ]).astype(np.float32)
        signed_deltas = np.concatenate([training_deltas, -training_deltas], axis=0)
        signed_labels = np.concatenate([
            np.ones(len(training_deltas), dtype=bool),
            np.zeros(len(training_deltas), dtype=bool),
        ])
        pair_model = classifier(lock["method"]["c_value"], lock["method"]["seed"] + fold_number)
        pair_model.fit(signed_deltas, signed_labels)

        by_surface: dict[str, Any] = {}
        point_directions: dict[str, Any] = {}
        pair_differences: dict[str, Any] = {}
        for surface in surfaces:
            surface_mask = heldout_mask & (data["surface_variants"] == surface)
            gold = data["gold_ambiguous"][surface_mask].astype(bool).tolist()
            scores = all_scores[surface_mask].tolist()
            by_surface[surface] = evaluate_binary(gold, scores, threshold)
            pairs = pair_indices(data, surface_mask)
            point_directions[surface] = direction_metrics(
                pairs,
                all_scores,
                data["primary_resolved_values"],
            )
            pair_differences[surface] = pair_probe_metrics(
                pairs,
                features,
                pair_model,
                data["primary_resolved_values"],
            )
        folds[heldout] = {
            "training_records": int(train_mask.sum()),
            "training_pairs": len(training_pairs),
            "calibration_records": int(calibration_mask.sum()),
            "threshold": threshold,
            "threshold_calibration": threshold_report,
            "pointwise_by_surface": by_surface,
            "pointwise_score_direction": point_directions,
            "pair_difference": pair_differences,
        }

    gates = gate_report(folds, lock["gates"])
    result = {
        "schema_version": 8,
        "experiment": "v8_frozen_mean_lomo_diagnostics",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "model": lock["method"]["model"],
        "feature": lock["method"]["feature"],
        "mechanics": mechanics,
        "folds": folds,
        "gates": gates,
        "decision": "advance_to_action_conditioned_head" if gates["passed"] else "mean_pooling_insufficient_use_token_aware_head",
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
