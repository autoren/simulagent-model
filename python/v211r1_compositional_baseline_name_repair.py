from __future__ import annotations

from typing import Any

from v211_deterministic_residual_baselines import (
    _predict_one,
    audit_scores,
    score_predictions,
)


def predict_evaluation(evaluation_surfaces: list[dict[str, Any]], learned: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = []
    for row in evaluation_surfaces:
        raw = _predict_one(row, learned["RAW_LEXICAL"], context_contrast=False)
        compositional = _predict_one(row, learned["COMPOSITIONAL_RESPONSE_SPAN"], context_contrast=True)
        consensus = raw if raw == compositional and raw != "ABSTAIN" else "ABSTAIN"
        values = {
            "RAW_LEXICAL": raw,
            "COMPOSITIONAL_RESPONSE_SPAN": compositional,
            "ABSTENTION_FIRST_CONSENSUS": consensus,
            "ABSTAIN_ALWAYS": "ABSTAIN",
        }
        for baseline, prediction in values.items():
            predictions.append({"record_id": row["record_id"], "baseline": baseline, "prediction": prediction, "accepted": prediction != "ABSTAIN"})
    return predictions


def repair_diagnostics(parent_predictions: list[dict[str, Any]], repaired_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_parent = [
        {**row, "baseline": "COMPOSITIONAL_RESPONSE_SPAN" if row["baseline"] == "CONTEXT_CONTRAST" else row["baseline"]}
        for row in parent_predictions
    ]
    value_keys = ("record_id", "prediction", "accepted")
    parent_values = sorted(tuple(row[key] for key in value_keys) for row in parent_predictions)
    repaired_values = sorted(tuple(row[key] for key in value_keys) for row in repaired_predictions)
    return {
        "normalized_parent_matches_repaired_exactly": normalized_parent == repaired_predictions,
        "prediction_values_match_as_multiset": parent_values == repaired_values,
        "changed_prediction_value_count": sum(left != right for left, right in zip(parent_values, repaired_values)),
        "old_key_count": sum(row["baseline"] == "CONTEXT_CONTRAST" for row in parent_predictions),
        "new_key_count": sum(row["baseline"] == "COMPOSITIONAL_RESPONSE_SPAN" for row in repaired_predictions),
        "repaired_prediction_count": len(repaired_predictions),
    }


__all__ = ["audit_scores", "predict_evaluation", "repair_diagnostics", "score_predictions"]
