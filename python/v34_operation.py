"""Prompt, ridge selection, and isolated operation metrics for V34."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v32_language import compile_truth


def operation_prompt(row: dict[str, Any], config: dict[str, Any]) -> str:
    interface = config["operationInterface"]
    options = "\n".join(
        f"{token}: {interface['definitions'][operation]}"
        for token, operation in zip(
            interface["labelTokens"], interface["classes"], strict=True
        )
    )
    return (
        f"Evidence statement: {row['agent_input']['evidence_text']}\n"
        "Classify the outer semantic operation applied to the embedded literal. The embedded "
        "literal may itself be positive or negative; ignore that lexical sign. Distinguish a "
        "denial of the literal from a denial of denying the literal.\n"
        f"Options:\n{options}\nAnswer:"
    )


def target_indices(rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> np.ndarray:
    classes = config["operationInterface"]["classes"]
    return np.asarray([
        classes.index(row["target"]["factorization"]["outer_operation"])
        for row in rows
    ], dtype=np.int64)


def surface_groups(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["oracle_metadata"]["surface_name"] for row in rows])


def make_ridge(alpha: float):
    return make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=float(alpha), class_weight="balanced")
    )


def cross_validate_ridge(
    features: np.ndarray, targets: np.ndarray, groups: np.ndarray,
    alphas: Sequence[float],
) -> list[dict[str, Any]]:
    reports = []
    unique = sorted(set(groups.tolist()))
    if len(unique) < 2:
        raise ValueError("V34 requires at least two fit surface groups")
    for alpha in alphas:
        folds = []
        for group in unique:
            held = groups == group
            model = make_ridge(alpha)
            model.fit(features[~held], targets[~held])
            folds.append({
                "held_out_surface_name": group,
                "records": int(np.sum(held)),
                "operation_accuracy": float(np.mean(model.predict(features[held]) == targets[held])),
            })
        reports.append({
            "alpha": float(alpha), "folds": folds,
            "mean_group_cv_operation_accuracy": float(np.mean([row["operation_accuracy"] for row in folds])),
            "minimum_group_cv_operation_accuracy": float(min(row["operation_accuracy"] for row in folds)),
        })
    return reports


def select_alpha(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return sorted(reports, key=lambda row: (
        -row["mean_group_cv_operation_accuracy"],
        -row["minimum_group_cv_operation_accuracy"],
        -row["alpha"],
    ))[0]


def select_prompt_method(methods: dict[str, dict[str, Any]]) -> str:
    eligible = ("semanticHiddenRidge", "nativeLogitRidge")
    return sorted(eligible, key=lambda name: (
        -methods[name]["selected_cv"]["mean_group_cv_operation_accuracy"],
        -methods[name]["selected_cv"]["minimum_group_cv_operation_accuracy"],
        -methods[name]["selected_cv"]["alpha"], name,
    ))[0]


def score_operations(
    rows: Sequence[dict[str, Any]], predictions: Sequence[int], config: dict[str, Any],
) -> dict[str, Any]:
    classes = config["operationInterface"]["classes"]
    targets = target_indices(rows, config)
    predicted = np.asarray(predictions, dtype=np.int64)
    if predicted.shape != targets.shape:
        raise ValueError("V34 predictions do not cover the requested population")
    by_operation = {}
    compiled = []
    for index, operation in enumerate(classes):
        mask = targets == index
        by_operation[operation] = {
            "records": int(np.sum(mask)),
            "operation_accuracy": float(np.mean(predicted[mask] == targets[mask])),
        }
    for row, prediction in zip(rows, predicted, strict=True):
        sign = row["target"]["factorization"]["lexical_sign"]
        compiled.append(
            compile_truth(sign, classes[int(prediction)], config["v32_config"])
            == row["target"]["truth_status"]
        )
    return {
        "records": len(rows),
        "operation_accuracy": float(np.mean(predicted == targets)),
        "worst_operation_accuracy": min(row["operation_accuracy"] for row in by_operation.values()),
        "oracle_sign_compiled_truth_accuracy": float(np.mean(compiled)),
        "by_operation": by_operation,
    }


def qualification(
    selected: dict[str, Any], legacy: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    gates = config["qualification"]
    gain = selected["operation_accuracy"] - legacy["operation_accuracy"]
    checks = {
        "calibration_operation_accuracy": selected["operation_accuracy"] >= gates["minimumCalibrationOperationAccuracy"],
        "worst_operation_accuracy": selected["worst_operation_accuracy"] >= gates["minimumWorstOperationCalibrationAccuracy"],
        "oracle_sign_compiled_truth_accuracy": selected["oracle_sign_compiled_truth_accuracy"] >= gates["minimumCalibrationOracleSignCompiledTruthAccuracy"],
        "gain_over_legacy": gain >= gates["minimumCalibrationGainOverLegacyHiddenRidge"],
    }
    return {"passed": all(checks.values()), "checks": checks, "calibration_operation_gain_over_legacy": gain}
