#!/usr/bin/env python3
"""Evaluate possible-outcome predictions for the v2 agent task."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_dataset import canonical, read_dataset
from evaluate_predictions import valid_schema


Record = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/v2/records/agent/test.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def normalize(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def valid_epistemic_schema(value: dict[str, Any]) -> bool:
    if set(value) != {"identifiable", "possible_outcomes"}:
        return False
    if not isinstance(value["identifiable"], bool):
        return False
    outcomes = value["possible_outcomes"]
    if not isinstance(outcomes, list) or not outcomes:
        return False
    if not all(isinstance(outcome, dict) and valid_schema(outcome) for outcome in outcomes):
        return False
    unique_outcomes = {canonical(outcome) for outcome in outcomes}
    if len(unique_outcomes) != len(outcomes):
        return False
    return value["identifiable"] == (len(unique_outcomes) == 1)


def evaluate_epistemic(gold_rows: list[Record], prediction_rows: list[Record]) -> dict[str, Any]:
    predictions = {row["id"]: row for row in prediction_rows}
    counts: defaultdict[str, int] = defaultdict(int)
    outcome_counts: defaultdict[str, int] = defaultdict(int)
    identifiability_confusion: defaultdict[str, int] = defaultdict(int)
    predicted_outcome_counts: defaultdict[int, int] = defaultdict(int)
    strata = {
        "identifiable": defaultdict(int),
        "ambiguous": defaultdict(int),
    }
    for record in gold_rows:
        counts["total"] += 1
        stratum = "identifiable" if record["target"]["identifiable"] else "ambiguous"
        strata[stratum]["total"] += 1
        expected = {canonical(value) for value in record["target"]["possible_outcomes"]}
        row = predictions.get(record["id"])
        if row is None:
            outcome_counts["fn"] += len(expected)
            continue
        counts["covered"] += 1
        strata[stratum]["covered"] += 1
        prediction = normalize(row.get("prediction"))
        if prediction is None:
            outcome_counts["fn"] += len(expected)
            continue
        counts["json_valid"] += 1
        strata[stratum]["json_valid"] += 1
        if not valid_epistemic_schema(prediction):
            outcome_counts["fn"] += len(expected)
            continue
        counts["schema_valid"] += 1
        strata[stratum]["schema_valid"] += 1
        predicted_label = "identifiable" if prediction["identifiable"] else "ambiguous"
        identifiability_confusion[f"gold_{stratum}_pred_{predicted_label}"] += 1
        predicted_outcome_counts[len(prediction["possible_outcomes"])] += 1
        if prediction["identifiable"] == record["target"]["identifiable"]:
            counts["identifiable_correct"] += 1
            strata[stratum]["identifiable_correct"] += 1
        predicted = {canonical(value) for value in prediction["possible_outcomes"]}
        outcome_counts["tp"] += len(predicted & expected)
        outcome_counts["fp"] += len(predicted - expected)
        outcome_counts["fn"] += len(expected - predicted)
        if predicted == expected:
            counts["outcome_set_exact"] += 1
            strata[stratum]["outcome_set_exact"] += 1
        if prediction == record["target"]:
            counts["exact"] += 1
            strata[stratum]["exact"] += 1

    total = counts["total"]
    tp, fp, fn = outcome_counts["tp"], outcome_counts["fp"], outcome_counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "examples": total,
        "coverage": counts["covered"] / total,
        "json_valid_rate": counts["json_valid"] / total,
        "schema_valid_rate": counts["schema_valid"] / total,
        "identifiability_accuracy": counts["identifiable_correct"] / total,
        "identifiability_confusion": {
            key: identifiability_confusion[key]
            for key in (
                "gold_identifiable_pred_identifiable",
                "gold_identifiable_pred_ambiguous",
                "gold_ambiguous_pred_identifiable",
                "gold_ambiguous_pred_ambiguous",
            )
        },
        "predicted_outcome_count_distribution": {
            str(key): value for key, value in sorted(predicted_outcome_counts.items())
        },
        "outcome_set_exact_match": counts["outcome_set_exact"] / total,
        "exact_match": counts["exact"] / total,
        "outcome_set_micro": {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "by_identifiability": {
            name: {
                "examples": values["total"],
                "coverage": values["covered"] / values["total"] if values["total"] else 0.0,
                "json_valid_rate": values["json_valid"] / values["total"] if values["total"] else 0.0,
                "schema_valid_rate": values["schema_valid"] / values["total"] if values["total"] else 0.0,
                "identifiability_accuracy": values["identifiable_correct"] / values["total"] if values["total"] else 0.0,
                "outcome_set_exact_match": values["outcome_set_exact"] / values["total"] if values["total"] else 0.0,
                "exact_match": values["exact"] / values["total"] if values["total"] else 0.0,
            }
            for name, values in strata.items()
        },
    }


def main() -> None:
    args = parse_args()
    gold = read_dataset(Path(args.gold).parent)
    if Path(args.gold).name != "test.jsonl":
        gold = read_jsonl(Path(args.gold))
    else:
        gold = [record for record in gold if record["split"] == "test"]
    predictions = read_jsonl(Path(args.predictions))
    report = evaluate_epistemic(gold, predictions)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
