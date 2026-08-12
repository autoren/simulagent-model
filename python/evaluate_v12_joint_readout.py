#!/usr/bin/env python3
"""Evaluate locked joint active/inactive readouts on existing frozen features."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v10_protocol import file_sha256, folds, load_locked_records


MODEL_ORDER = ["qwen35_0_8b", "qwen35_4b", "qwen35_9b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v12-joint-readout-lock.json")
    parser.add_argument("--output-dir", default="outputs/v12-joint-readout/evaluation")
    return parser.parse_args()


def primary_probe(specification: dict[str, Any], seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=specification["cValue"],
            class_weight=specification["classWeight"],
            max_iter=specification["maxIterations"],
            random_state=seed,
            solver=specification["solver"],
        ),
    )


def conditional_probe(specification: dict[str, Any], seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(specification["hiddenUnits"],),
            activation=specification["activation"],
            alpha=specification["alpha"],
            batch_size=specification["batchSize"],
            learning_rate_init=specification["learningRateInit"],
            max_iter=specification["maxIterations"],
            random_state=seed,
            solver=specification["solver"],
            tol=specification["tolerance"],
            n_iter_no_change=specification["iterationsWithoutChange"],
            early_stopping=False,
            shuffle=True,
        ),
    )


def model_features(name: str, active: np.ndarray, inactive: np.ndarray) -> np.ndarray:
    difference = active - inactive
    if name == "signed_difference_linear":
        return difference
    if name == "joint_mlp":
        return np.concatenate(((active + inactive) * 0.5, difference), axis=1)
    raise ValueError(f"unknown V12 head: {name}")


def probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    return model.predict_proba(values)[:, 1].astype(np.float32)


def metrics(targets: np.ndarray, scores: np.ndarray, swapped_scores: np.ndarray) -> dict[str, Any]:
    predictions = scores >= 0.5
    swapped_predictions = swapped_scores >= 0.5
    return {
        "examples": int(targets.size),
        "class_counts": [int(np.sum(targets == 0)), int(np.sum(targets == 1))],
        "accuracy": float(np.mean(predictions == targets)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "swap_complement_accuracy": float(np.mean(predictions != swapped_predictions)),
    }


def save_pipeline(path: Path, name: str, model: Any) -> None:
    scaler = model.named_steps["standardscaler"]
    estimator = model.steps[-1][1]
    payload: dict[str, Any] = {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "classes": estimator.classes_,
    }
    if name == "signed_difference_linear":
        payload.update({"coef": estimator.coef_, "intercept": estimator.intercept_})
    else:
        payload.update({
            "n_iter": np.asarray([estimator.n_iter_], dtype=np.int32),
            "loss": np.asarray([estimator.loss_], dtype=np.float64),
            **{f"coef_{index}": value for index, value in enumerate(estimator.coefs_)},
            **{f"intercept_{index}": value for index, value in enumerate(estimator.intercepts_)},
        })
    np.savez_compressed(path, **payload)


def gate_report(fold_results: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    overall = [value["overall"]["accuracy"] for value in fold_results.values()]
    surfaces = [
        cell["accuracy"]
        for value in fold_results.values()
        for cell in value["by_surface"].values()
    ]
    checks = [
        {
            "name": "minimum_fold_accuracy",
            "value": float(min(overall)),
            "minimum": gates["minimumEveryFoldAccuracy"],
        },
        {
            "name": "minimum_surface_accuracy",
            "value": float(min(surfaces)),
            "minimum": gates["minimumEverySurfaceAccuracy"],
        },
    ]
    for check in checks:
        check["passed"] = check["value"] >= check["minimum"]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def smallest_passing(reports: dict[str, Any]) -> str | None:
    return next((model for model in MODEL_ORDER if reports.get(model, {}).get("gates", {}).get("passed")), None)


def decision(primary: dict[str, Any], conditional: dict[str, Any] | None) -> str:
    passing = smallest_passing(primary)
    if passing is not None:
        return f"joint_linear_relation_accessible_repair_temporal_with_{passing}"
    passing = smallest_passing(conditional or {})
    if passing is not None:
        return f"joint_nonlinear_relation_accessible_repair_temporal_with_{passing}"
    return "frozen_final_state_joint_readout_insufficient_extract_token_span_interactions"


def evaluate_head(
    head_name: str,
    specification: dict[str, Any],
    records: list[dict[str, Any]],
    all_folds: list[dict[str, Any]],
    pair_records: np.ndarray,
    eligible_pair_indices: np.ndarray,
    targets: np.ndarray,
    active: np.ndarray,
    inactive: np.ndarray,
    output_dir: Path,
    model_key: str,
    seed: int,
) -> dict[str, Any]:
    features = model_features(head_name, active, inactive)
    swapped_features = model_features(head_name, inactive, active)
    eligible_records = pair_records[eligible_pair_indices]
    surface_by_record = np.asarray([record["state_lexicon_family"] for record in records])
    results = {}
    for fold_index, fold in enumerate(all_folds):
        training = fold["train"][eligible_records]
        evaluation = fold["evaluation"][eligible_records]
        if head_name == "signed_difference_linear":
            model = primary_probe(specification, seed + fold_index)
        else:
            model = conditional_probe(specification, seed + fold_index)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features[training], targets[training])
        convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        scores = probabilities(model, features)
        swapped_scores = probabilities(model, swapped_features)
        fold_slug = fold["name"].replace(":", "-")
        artifact = output_dir / model_key / f"{head_name}-{fold_slug}-head.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        save_pipeline(artifact, head_name, model)
        overall = metrics(targets[evaluation], scores[evaluation], swapped_scores[evaluation])
        by_surface = {}
        for surface in sorted(set(surface_by_record.tolist())):
            mask = evaluation & (surface_by_record[eligible_records] == surface)
            if mask.any():
                by_surface[surface] = metrics(targets[mask], scores[mask], swapped_scores[mask])
        results[fold["name"]] = {
            "kind": fold["kind"],
            "training_examples": int(training.sum()),
            "evaluation_examples": int(evaluation.sum()),
            "convergence_warnings": convergence_warnings,
            "head_artifact": str(artifact),
            "head_artifact_sha256": file_sha256(artifact),
            "overall": overall,
            "by_surface": by_surface,
        }
    return results


def verify_sources(lock_path: Path, lock: dict[str, Any]) -> None:
    if lock["model_order"] != MODEL_ORDER:
        raise RuntimeError("V12 model order differs from implementation")
    for path, expected in lock["implementation"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"V12 locked implementation changed: {path}")
    for source in (lock["source_v10"], lock["source_v11"]):
        for key, value in source.items():
            if key.endswith("_sha256"):
                path_key = key.removesuffix("_sha256")
                if path_key in source and file_sha256(Path(source[path_key])) != value:
                    raise RuntimeError(f"V12 source changed: {source[path_key]}")
    for specification in lock["models"].values():
        if file_sha256(Path(specification["feature_artifact"])) != specification["feature_artifact_sha256"]:
            raise RuntimeError("V12 feature artifact changed")
        if file_sha256(Path(specification["feature_metadata"])) != specification["feature_metadata_sha256"]:
            raise RuntimeError("V12 feature metadata changed")


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"V12 result already exists: {result_path}")
    lock = json.loads(lock_path.read_text())
    verify_sources(lock_path, lock)
    records = load_locked_records(json.loads(Path(lock["source_v10"]["frozen_lock"]).read_text()))
    all_folds = folds(records)

    with np.load(lock["source_v10"]["feature_artifact"], allow_pickle=False) as source:
        pair_records = source["pair_record_indices"].astype(np.int32)
        pair_nli = source["pair_nli_indices"].astype(np.int32)
        match_targets = source["match_targets"].astype(bool)
        current_targets = source["current_value_targets"].astype(np.int8)
        record_ids = source["record_ids"].tolist()
    if record_ids != [record["id"] for record in records]:
        raise RuntimeError("V12 feature indices and locked records differ")
    eligible_pair_indices = np.flatnonzero(match_targets & (current_targets >= 0))
    eligible_nli = pair_nli[eligible_pair_indices]
    targets = current_targets[eligible_pair_indices]
    if np.bincount(targets, minlength=2).tolist() != [int(targets.size / 2), int(targets.size / 2)]:
        raise RuntimeError("V12 expected exactly balanced eligible examples")

    output_dir.mkdir(parents=True, exist_ok=True)
    head_results: dict[str, dict[str, Any]] = {lock["primary_head"]["name"]: {}}
    for model_key in lock["model_order"]:
        feature_path = lock["models"][model_key]["feature_artifact"]
        with np.load(feature_path, allow_pickle=False) as values:
            nli = values["nli_final_features"].astype(np.float32)
        active = nli[eligible_nli[:, 0]]
        inactive = nli[eligible_nli[:, 1]]
        folds_result = evaluate_head(
            lock["primary_head"]["name"], lock["primary_head"], records, all_folds,
            pair_records, eligible_pair_indices, targets, active, inactive,
            output_dir, model_key, lock["seed"],
        )
        head_results[lock["primary_head"]["name"]][model_key] = {
            "folds": folds_result,
            "gates": gate_report(folds_result, lock["gates"]),
        }

    primary_results = head_results[lock["primary_head"]["name"]]
    conditional_ran = smallest_passing(primary_results) is None
    if conditional_ran:
        conditional_name = lock["conditional_head"]["name"]
        head_results[conditional_name] = {}
        for model_key in lock["model_order"]:
            feature_path = lock["models"][model_key]["feature_artifact"]
            with np.load(feature_path, allow_pickle=False) as values:
                nli = values["nli_final_features"].astype(np.float32)
            active = nli[eligible_nli[:, 0]]
            inactive = nli[eligible_nli[:, 1]]
            folds_result = evaluate_head(
                conditional_name, lock["conditional_head"], records, all_folds,
                pair_records, eligible_pair_indices, targets, active, inactive,
                output_dir, model_key, lock["seed"],
            )
            head_results[conditional_name][model_key] = {
                "folds": folds_result,
                "gates": gate_report(folds_result, lock["gates"]),
            }

    conditional_results = head_results.get(lock["conditional_head"]["name"])
    result = {
        "schema_version": 12,
        "experiment": "v12_frozen_joint_hypothesis_readout",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "eligible_examples": int(targets.size),
        "class_counts": np.bincount(targets, minlength=2).tolist(),
        "model_order": lock["model_order"],
        "primary_head": lock["primary_head"]["name"],
        "conditional_head": lock["conditional_head"]["name"],
        "conditional_head_ran": conditional_ran,
        "heads": head_results,
        "decision": decision(primary_results, conditional_results),
        "lora_authorized": False,
        "final_mechanic_authorized": False,
        "new_feature_extractions": 0,
        "data_access": lock["data_access"],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "conditional_head_ran": conditional_ran,
        "gates": {
            head: {model: value["gates"] for model, value in models.items()}
            for head, models in head_results.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
