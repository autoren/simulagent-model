#!/usr/bin/env python3
"""Select and evaluate a regularized float32 probe over frozen Qwen features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binary_metrics import evaluate_binary, fit_threshold


DEFAULT_C_VALUES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=3505)
    return parser.parse_args()


def load_split(root: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(root / f"{split}.npz", allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def feature_keys(values: dict[str, np.ndarray]) -> list[str]:
    return sorted(key for key in values if key.startswith("layer_"))


def make_probe(c_value: float, seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                    solver="saga",
                ),
            ),
        ]
    )


def error_concentration(
    rows: list[dict[str, Any]], threshold: float
) -> dict[str, int]:
    groups: dict[str, list[bool]] = {}
    for row in rows:
        error = (row["score"] > threshold) != row["gold_ambiguous"]
        groups.setdefault(row["split_group"], []).append(error)
    return {
        "errors": sum(sum(errors) for errors in groups.values()),
        "context_groups_with_errors": sum(any(errors) for errors in groups.values()),
        "context_groups": len(groups),
        "completely_misclassified_context_groups": sum(all(errors) for errors in groups.values()),
    }


def candidate_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, str]:
    report = candidate["calibration"]
    return (
        report["balanced_accuracy"],
        report["ambiguity"]["f1"],
        report["roc_auc"],
        -candidate["c_value"],
        candidate["feature"],
    )


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values), quantile))


def grouped_bootstrap(
    gold: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    grouped_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    balanced = []
    auc = []
    attempts = 0
    while len(balanced) < samples and attempts < samples * 10:
        attempts += 1
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([grouped_indices[group] for group in sampled])
        sampled_gold = gold[indices].astype(bool).tolist()
        if len(set(sampled_gold)) < 2:
            continue
        report = evaluate_binary(sampled_gold, scores[indices].tolist(), threshold)
        balanced.append(report["balanced_accuracy"])
        auc.append(report["roc_auc"])
    if len(balanced) < samples:
        raise RuntimeError(f"Only obtained {len(balanced)} valid grouped bootstrap samples")
    return {
        "unit": "context_group",
        "context_groups": len(unique_groups),
        "samples": samples,
        "balanced_accuracy_95_percentile_interval": [
            percentile(balanced, 0.025),
            percentile(balanced, 0.975),
        ],
        "roc_auc_95_percentile_interval": [percentile(auc, 0.025), percentile(auc, 0.975)],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.features)
    metadata = json.loads((root / "metadata.json").read_text())
    train = load_split(root, "train")
    calibration = load_split(root, "calibration")
    candidates = []
    fitted_probes: dict[tuple[str, float], Pipeline] = {}
    train_gold = train["gold_ambiguous"].astype(bool)
    calibration_gold = calibration["gold_ambiguous"].astype(bool).tolist()
    for feature in feature_keys(train):
        for c_value in DEFAULT_C_VALUES:
            probe = make_probe(c_value, args.seed)
            probe.fit(train[feature], train_gold)
            raw_scores = probe.decision_function(calibration[feature])
            if raw_scores.dtype != np.float32:
                raise RuntimeError(f"Probe decision scores are not float32: {raw_scores.dtype}")
            scores = raw_scores.astype(float).tolist()
            fitted = fit_threshold(calibration_gold, scores)
            candidate = {
                "feature": feature,
                "c_value": c_value,
                "threshold": fitted["threshold"],
                "calibration": fitted,
            }
            candidates.append(candidate)
            fitted_probes[(feature, c_value)] = probe

    selected = max(candidates, key=candidate_key)
    selected_probe = fitted_probes[(selected["feature"], selected["c_value"])]
    validation = load_split(root, "validation")
    validation_gold = validation["gold_ambiguous"].astype(bool)
    raw_validation_scores = selected_probe.decision_function(validation[selected["feature"]])
    if raw_validation_scores.dtype != np.float32:
        raise RuntimeError(
            f"Selected probe decision scores are not float32: {raw_validation_scores.dtype}"
        )
    validation_scores = raw_validation_scores.astype(float)
    validation_report = evaluate_binary(
        validation_gold.tolist(), validation_scores.tolist(), selected["threshold"]
    )
    bootstrap = grouped_bootstrap(
        validation_gold,
        validation_scores,
        validation["groups"],
        selected["threshold"],
        args.bootstrap_samples,
        args.bootstrap_seed + args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier = selected_probe.named_steps["classifier"]
    scaler = selected_probe.named_steps["standardize"]
    np.savez_compressed(
        output_dir / "selected-probe.npz",
        coefficient=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )
    score_rows = [
        {
            "id": str(record_id),
            "split_group": str(group),
            "gold_ambiguous": bool(gold),
            "score": float(score),
        }
        for record_id, group, gold, score in zip(
            validation["ids"], validation["groups"], validation_gold, validation_scores
        )
    ]
    (output_dir / "validation.scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows)
    )
    result = {
        "experiment": "v5_frozen_linear_probe",
        "seed": args.seed,
        "model": metadata["model"],
        "frozen": True,
        "adapter_path": None,
        "input_variant": metadata["input_variant"],
        "training_split": "train",
        "selection_split": "calibration",
        "evaluation_split": "validation",
        "test_records_read": 0,
        "candidate_count": len(candidates),
        "numerics": {
            "feature_dtype": str(train[selected["feature"]].dtype),
            "coefficient_dtype": str(classifier.coef_.dtype),
            "decision_score_dtype": str(raw_validation_scores.dtype),
        },
        "selected": selected,
        "validation": validation_report,
        "validation_error_concentration": error_concentration(
            score_rows, selected["threshold"]
        ),
        "validation_grouped_bootstrap": bootstrap,
        "calibration_candidates": candidates,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
