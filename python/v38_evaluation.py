"""Focus selection and compiled semantic metrics for V38."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from v35_binding import make_ridge
from v32_language import compile_truth


METHODS = (
    "candidate_span_hidden_shared_ridge",
    "candidate_span_native_margin",
    "deterministic_discourse_parser",
)


def predict_hidden(
    train_hidden: np.ndarray, train_mask: np.ndarray, train_targets: np.ndarray,
    eval_hidden: np.ndarray, eval_mask: np.ndarray, alpha: float,
) -> tuple[np.ndarray, Any]:
    classes = train_hidden.shape[1]
    flat_x = train_hidden.reshape(-1, train_hidden.shape[-1])
    flat_valid = train_mask.reshape(-1)
    flat_y = (np.arange(classes)[None, :] == train_targets[:, None]).reshape(-1).astype(np.int64)
    model = make_ridge(alpha)
    model.fit(flat_x[flat_valid], flat_y[flat_valid])
    scores = np.asarray(model.decision_function(eval_hidden.reshape(-1, eval_hidden.shape[-1]))).reshape(len(eval_hidden), classes)
    scores = np.where(eval_mask, scores, -np.inf)
    return np.argmax(scores, axis=1).astype(np.int64), model


def predict_method(method, alpha, train, train_targets, evaluate):
    if method == "candidate_span_hidden_shared_ridge":
        return predict_hidden(
            train["hidden"], train["mask"], train_targets,
            evaluate["hidden"], evaluate["mask"], float(alpha),
        )[0]
    if method == "candidate_span_native_margin":
        scores = np.where(evaluate["mask"], evaluate["margin"], -np.inf)
        return np.argmax(scores, axis=1).astype(np.int64)
    if method == "deterministic_discourse_parser":
        return evaluate["deterministic"].astype(np.int64)
    raise ValueError(f"Unknown V38 focus method: {method}")


def cross_validate(bundle, targets, groups, alphas):
    unique = sorted(set(groups))
    reports = []
    choices = [("candidate_span_hidden_shared_ridge", alpha) for alpha in alphas]
    choices += [("candidate_span_native_margin", None), ("deterministic_discourse_parser", None)]
    groups = np.asarray(groups)
    for method, alpha in choices:
        folds = []
        for group in unique:
            held = groups == group
            train = {key: value[~held] for key, value in bundle.items()}
            evaluate = {key: value[held] for key, value in bundle.items()}
            prediction = predict_method(method, alpha, train, targets[~held], evaluate)
            folds.append({"held_out_surface_family": group, "records": int(np.sum(held)), "focus_accuracy": float(np.mean(prediction == targets[held]))})
        reports.append({
            "method": method, "alpha": alpha, "folds": folds,
            "mean_group_cv_focus_accuracy": float(np.mean([row["focus_accuracy"] for row in folds])),
            "worst_group_cv_focus_accuracy": float(min(row["focus_accuracy"] for row in folds)),
        })
    return reports


def select_method(reports, method_order):
    order = {method: index for index, method in enumerate(method_order)}
    return sorted(reports, key=lambda row: (
        -row["mean_group_cv_focus_accuracy"], -row["worst_group_cv_focus_accuracy"],
        -(row["alpha"] or 0.0), order[row["method"]],
    ))[0]


def score(rows, selected_indices, candidates, operation_predictions, v32_config):
    details = []
    groups = {name: defaultdict(list) for name in ("surface", "order", "decoy", "orientation")}
    for row, index, row_candidates, operation in zip(rows, selected_indices, candidates, operation_predictions, strict=True):
        selected = row_candidates[int(index)]
        target = row["target"]
        focus_correct = int(index) == target["focus_candidate_index"]
        sign_correct = selected.sign == target["focus"]["lexical_sign"]
        operation_correct = operation == target["outer_operation"]
        truth_correct = compile_truth(selected.sign, operation, v32_config) == target["truth_status"]
        detail = {"focus_correct": focus_correct, "sign_correct": sign_correct, "operation_correct": operation_correct, "truth_correct": truth_correct}
        details.append(detail)
        metadata = row["oracle_metadata"]
        groups["surface"][metadata["surface_family"]].append(detail)
        groups["order"][metadata["focus_order"]].append(detail)
        groups["decoy"][metadata["decoy_kind"]].append(detail)
        groups["orientation"][metadata["orientation"]].append(detail)
    mean = lambda values, key: float(np.mean([row[key] for row in values]))
    summarize = lambda grouped: {
        name: {"records": len(values), "focus_accuracy": mean(values, "focus_correct"), "lexical_sign_accuracy": mean(values, "sign_correct"), "compiled_truth_accuracy": mean(values, "truth_correct")}
        for name, values in sorted(grouped.items())
    }
    by_surface, by_order, by_decoy, by_orientation = (summarize(groups[name]) for name in ("surface", "order", "decoy", "orientation"))
    return {
        "records": len(details),
        "focus_accuracy": mean(details, "focus_correct"),
        "lexical_sign_accuracy": mean(details, "sign_correct"),
        "outer_operation_accuracy": mean(details, "operation_correct"),
        "compiled_truth_accuracy": mean(details, "truth_correct"),
        "focus_first_accuracy": by_order["focus_first"]["focus_accuracy"],
        "focus_second_accuracy": by_order["focus_second"]["focus_accuracy"],
        "exact_opposite_decoy_accuracy": by_decoy["exact_opposite"]["focus_accuracy"],
        "different_atom_decoy_accuracy": by_decoy["different_grounded_atom"]["focus_accuracy"],
        "worst_surface_family_accuracy": min(row["focus_accuracy"] for row in by_surface.values()),
        "by_surface_family": by_surface, "by_focus_order": by_order,
        "by_decoy_kind": by_decoy, "by_orientation": by_orientation,
    }


def qualification(metrics, config):
    gates = config["gates"]
    checks = {
        "focus": metrics["focus_accuracy"] >= gates["minimumValidationFocusAccuracy"],
        "lexical_sign": metrics["lexical_sign_accuracy"] >= gates["minimumValidationLexicalSignAccuracy"],
        "focus_first": metrics["focus_first_accuracy"] >= gates["minimumFocusFirstAccuracy"],
        "focus_second": metrics["focus_second_accuracy"] >= gates["minimumFocusSecondAccuracy"],
        "exact_opposite_decoy": metrics["exact_opposite_decoy_accuracy"] >= gates["minimumExactOppositeDecoyAccuracy"],
        "different_atom_decoy": metrics["different_atom_decoy_accuracy"] >= gates["minimumDifferentAtomDecoyAccuracy"],
        "worst_surface_family": metrics["worst_surface_family_accuracy"] >= gates["minimumWorstSurfaceFamilyAccuracy"],
        "compiled_truth": metrics["compiled_truth_accuracy"] >= gates["minimumCompiledTruthWithFrozenOperation"],
    }
    return {"passed": all(checks.values()), "checks": checks}
