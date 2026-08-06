#!/usr/bin/env python3
"""Deterministic binary classification metrics and threshold fitting."""

from __future__ import annotations

from math import inf
from typing import Any, Sequence


def evaluate_binary(
    gold_ambiguous: Sequence[bool], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    if len(gold_ambiguous) != len(scores) or not scores:
        raise ValueError("Gold labels and non-empty scores must have equal lengths.")
    predicted = [score > threshold for score in scores]
    tp = sum(gold and guess for gold, guess in zip(gold_ambiguous, predicted))
    tn = sum((not gold) and (not guess) for gold, guess in zip(gold_ambiguous, predicted))
    fp = sum((not gold) and guess for gold, guess in zip(gold_ambiguous, predicted))
    fn = sum(gold and (not guess) for gold, guess in zip(gold_ambiguous, predicted))
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "examples": len(scores),
        "threshold": threshold,
        "accuracy": safe_divide(tp + tn, len(scores)),
        "balanced_accuracy": (recall + specificity) / 2,
        "roc_auc": roc_auc(gold_ambiguous, scores),
        "ambiguity": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "prediction_distribution": {
            "identifiable": predicted.count(False),
            "ambiguous": predicted.count(True),
        },
    }


def fit_threshold(gold_ambiguous: Sequence[bool], scores: Sequence[float]) -> dict[str, Any]:
    unique = sorted(set(scores))
    if not unique:
        raise ValueError("Cannot fit a threshold without scores.")
    span = max(1.0, abs(unique[0]), abs(unique[-1])) * 1e-9
    candidates = [unique[0] - span]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    candidates.append(unique[-1] + span)
    reports = [evaluate_binary(gold_ambiguous, scores, threshold) for threshold in candidates]
    return max(
        reports,
        key=lambda report: (
            report["balanced_accuracy"],
            report["ambiguity"]["f1"],
            report["accuracy"],
            -abs(report["threshold"]),
            -report["threshold"],
        ),
    )


def roc_auc(gold_ambiguous: Sequence[bool], scores: Sequence[float]) -> float:
    positive = [score for gold, score in zip(gold_ambiguous, scores) if gold]
    negative = [score for gold, score in zip(gold_ambiguous, scores) if not gold]
    if not positive or not negative:
        return 0.5
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            if positive_score > negative_score:
                wins += 1
            elif positive_score == negative_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def nonconstant(report: dict[str, Any]) -> bool:
    distribution = report["prediction_distribution"]
    return distribution["identifiable"] > 0 and distribution["ambiguous"] > 0


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
