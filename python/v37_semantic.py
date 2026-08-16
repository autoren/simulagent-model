"""Fit-only selection and validation metrics for V37 semantic invariance."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v32_language import compile_truth


LEARNED_METHODS = (
    "candidate_hidden_shared_ridge",
    "candidate_margin_vector_ridge",
    "direct_hidden_ridge",
)
ALL_METHODS = (
    "candidate_hidden_shared_ridge",
    "candidate_margin_vector_ridge",
    "candidate_native_margin",
    "direct_hidden_ridge",
)


def make_ridge(alpha: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=float(alpha), class_weight="balanced"),
    )


def binary_positive_scores(model: Any, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(features), dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("Expected a binary candidate-compatibility decision function")
    return scores


def fit_predict_method(
    method: str,
    alpha: float | None,
    train: dict[str, np.ndarray],
    train_targets: np.ndarray,
    evaluate: dict[str, np.ndarray],
) -> tuple[np.ndarray, Any | None]:
    classes = train["candidate_hidden"].shape[1]
    if evaluate["candidate_hidden"].shape[1] != classes:
        raise ValueError("Candidate class dimension changed")
    if method == "candidate_native_margin":
        return np.argmax(evaluate["candidate_margin"], axis=1).astype(np.int64), None
    if alpha is None:
        raise ValueError(f"V37 learned method lacks alpha: {method}")
    if method == "candidate_hidden_shared_ridge":
        x_train = train["candidate_hidden"].reshape(-1, train["candidate_hidden"].shape[-1])
        y_train = (np.arange(classes)[None, :] == train_targets[:, None]).reshape(-1).astype(np.int64)
        model = make_ridge(alpha)
        model.fit(x_train, y_train)
        scores = binary_positive_scores(
            model, evaluate["candidate_hidden"].reshape(-1, evaluate["candidate_hidden"].shape[-1])
        ).reshape(len(evaluate["candidate_hidden"]), classes)
        return np.argmax(scores, axis=1).astype(np.int64), model
    if method == "candidate_margin_vector_ridge":
        model = make_ridge(alpha)
        model.fit(train["candidate_margin"], train_targets)
        return np.asarray(model.predict(evaluate["candidate_margin"]), dtype=np.int64), model
    if method == "direct_hidden_ridge":
        model = make_ridge(alpha)
        model.fit(train["direct_hidden"], train_targets)
        return np.asarray(model.predict(evaluate["direct_hidden"]), dtype=np.int64), model
    raise ValueError(f"Unknown V37 method: {method}")


def grouped_folds(groups: Sequence[str], fold_count: int, seed: int) -> np.ndarray:
    unique = sorted(
        set(groups),
        key=lambda group: hashlib.sha256(f"v37-fold|{seed}|{group}".encode()).hexdigest(),
    )
    if len(unique) < fold_count:
        raise ValueError("Too few V37 selection groups")
    mapping = {group: index % fold_count for index, group in enumerate(unique)}
    return np.asarray([mapping[group] for group in groups], dtype=np.int64)


def cross_validate_component(
    bundle: dict[str, np.ndarray],
    targets: np.ndarray,
    groups: Sequence[str],
    alphas: Sequence[float],
    fold_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    folds = grouped_folds(groups, fold_count, seed)
    reports: list[dict[str, Any]] = []
    candidates = [(method, float(alpha)) for method in LEARNED_METHODS for alpha in alphas]
    candidates.append(("candidate_native_margin", None))
    for method, alpha in candidates:
        fold_reports = []
        for fold in range(fold_count):
            held = folds == fold
            train_bundle = {key: value[~held] for key, value in bundle.items()}
            held_bundle = {key: value[held] for key, value in bundle.items()}
            predicted, _ = fit_predict_method(method, alpha, train_bundle, targets[~held], held_bundle)
            fold_reports.append({
                "fold": fold,
                "records": int(np.sum(held)),
                "accuracy": float(np.mean(predicted == targets[held])),
            })
        reports.append({
            "method": method,
            "alpha": alpha,
            "folds": fold_reports,
            "mean_group_cv_accuracy": float(np.mean([row["accuracy"] for row in fold_reports])),
            "worst_group_cv_accuracy": float(min(row["accuracy"] for row in fold_reports)),
        })
    return reports


def select_method(reports: Sequence[dict[str, Any]], method_order: Sequence[str]) -> dict[str, Any]:
    order = {method: index for index, method in enumerate(method_order)}
    return sorted(
        reports,
        key=lambda row: (
            -row["mean_group_cv_accuracy"],
            -row["worst_group_cv_accuracy"],
            -(row["alpha"] if row["alpha"] is not None else 0.0),
            order[row["method"]],
        ),
    )[0]


def semantic_predictions(
    sign_indices: np.ndarray,
    operation_indices: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, str]]:
    signs = config["interfaces"]["lexicalSignClasses"]
    operations = config["interfaces"]["outerOperationClasses"]
    return [
        {"lexical_sign": signs[int(sign)], "outer_operation": operations[int(operation)]}
        for sign, operation in zip(sign_indices, operation_indices, strict=True)
    ]


def score_semantics(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, str]],
    v32_config: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("V37 predictions do not cover validation")
    details: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_placement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        target = row["target"]
        target_factors = target["factorization"]
        sign_correct = prediction["lexical_sign"] == target_factors["lexical_sign"]
        operation_correct = prediction["outer_operation"] == target_factors["outer_operation"]
        truth = compile_truth(
            prediction["lexical_sign"], prediction["outer_operation"], v32_config
        )
        truth_correct = truth == target["truth_status"]
        detail = {
            "id": row["id"],
            "lexical_sign_correct": sign_correct,
            "outer_operation_correct": operation_correct,
            "compiled_truth_correct": truth_correct,
            "negative_composition": target_factors["lexical_sign"] == "negative"
            and target_factors["outer_operation"] in ("deny", "double_deny", "contrast_select"),
        }
        details.append(detail)
        by_family[row["oracle_metadata"]["surface_family"]].append(detail)
        by_operation[target_factors["outer_operation"]].append(detail)
        by_placement[row["oracle_metadata"]["distractor_placement"]].append(detail)

    mean = lambda values: float(np.mean(values)) if values else 0.0
    family_metrics = {
        name: {
            "records": len(values),
            "compiled_truth_accuracy": mean([row["compiled_truth_correct"] for row in values]),
        }
        for name, values in sorted(by_family.items())
    }
    operation_metrics = {
        name: {
            "records": len(values),
            "outer_operation_accuracy": mean([row["outer_operation_correct"] for row in values]),
        }
        for name, values in sorted(by_operation.items())
    }
    placement_metrics = {
        name: {
            "records": len(values),
            "lexical_sign_accuracy": mean([row["lexical_sign_correct"] for row in values]),
            "outer_operation_accuracy": mean([row["outer_operation_correct"] for row in values]),
            "compiled_truth_accuracy": mean([row["compiled_truth_correct"] for row in values]),
        }
        for name, values in sorted(by_placement.items())
    }
    detail_by_id = {row["id"]: row for row in details}
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        for pair in row["oracle_metadata"]["pairs"]:
            pairs[(pair["kind"], pair["id"])].append(row["id"])
    pairs_by_kind: dict[str, list[bool]] = defaultdict(list)
    for (kind, _), identifiers in pairs.items():
        pairs_by_kind[kind].append(
            all(detail_by_id[identifier]["compiled_truth_correct"] for identifier in identifiers)
        )
    pair_metrics = {
        kind: {"pairs": len(values), "pair_exact_accuracy": mean(values)}
        for kind, values in sorted(pairs_by_kind.items())
    }
    negative = [row for row in details if row["negative_composition"]]
    distractor = [
        row for placement, values in by_placement.items() if placement != "none" for row in values
    ]
    return {
        "records": len(details),
        "lexical_sign_accuracy": mean([row["lexical_sign_correct"] for row in details]),
        "outer_operation_accuracy": mean([row["outer_operation_correct"] for row in details]),
        "compiled_truth_accuracy": mean([row["compiled_truth_correct"] for row in details]),
        "worst_operation_accuracy": min(value["outer_operation_accuracy"] for value in operation_metrics.values()),
        "worst_surface_family_truth_accuracy": min(value["compiled_truth_accuracy"] for value in family_metrics.values()),
        "distractor_truth_accuracy": mean([row["compiled_truth_correct"] for row in distractor]),
        "negative_composition_truth_accuracy": mean([row["compiled_truth_correct"] for row in negative]),
        "by_surface_family": family_metrics,
        "by_operation": operation_metrics,
        "by_distractor_placement": placement_metrics,
        "by_pair_kind": pair_metrics,
    }


def qualification(
    selected: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["qualification"]
    gain = selected["compiled_truth_accuracy"] - baseline["compiled_truth_accuracy"]
    checks = {
        "lexical_sign": selected["lexical_sign_accuracy"] >= gates["minimumValidationLexicalSignAccuracy"],
        "outer_operation": selected["outer_operation_accuracy"] >= gates["minimumValidationOuterOperationAccuracy"],
        "compiled_truth": selected["compiled_truth_accuracy"] >= gates["minimumValidationCompiledTruthAccuracy"],
        "worst_operation": selected["worst_operation_accuracy"] >= gates["minimumWorstOperationAccuracy"],
        "worst_surface_family_truth": selected["worst_surface_family_truth_accuracy"] >= gates["minimumWorstSurfaceFamilyTruthAccuracy"],
        "distractor_truth": selected["distractor_truth_accuracy"] >= gates["minimumDistractorTruthAccuracy"],
        "negative_composition_truth": selected["negative_composition_truth_accuracy"] >= gates["minimumNegativeCompositionTruthAccuracy"],
        "truth_gain_over_frozen_v36": gain >= gates["minimumTruthGainOverFrozenV36Interface"],
    }
    checks.update({
        f"pair_{kind}": value["pair_exact_accuracy"] >= gates["minimumPairExactAccuracy"]
        for kind, value in selected["by_pair_kind"].items()
    })
    if all(checks.values()):
        decision = "semantic_invariance_qualified_preregister_fresh_confirmation_only"
    elif checks["truth_gain_over_frozen_v36"]:
        decision = "semantic_invariance_improved_continue_isolated_development"
    else:
        decision = "semantic_invariance_no_material_gain_pivot_parser_or_grounder"
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "compiled_truth_gain_over_frozen_v36": gain,
        "decision": decision,
    }
