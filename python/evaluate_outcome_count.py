#!/usr/bin/env python3
"""Evaluate compact outcome-count predictions against v2 epistemic records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def valid_count_schema(value: dict[str, Any]) -> bool:
    count = value.get("outcome_count")
    return (
        set(value) == {"outcome_count"}
        and isinstance(count, int)
        and not isinstance(count, bool)
        and 1 <= count <= 5
    )


def normalize_count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 1 <= value <= 5 else None
    if isinstance(value, dict):
        return value["outcome_count"] if valid_count_schema(value) else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if text in {"1", "2", "3", "4", "5"}:
        return int(text)
    parsed = normalize(text)
    return parsed["outcome_count"] if parsed and valid_count_schema(parsed) else None


def evaluate_counts(gold_rows: list[Record], prediction_rows: list[Record]) -> dict[str, Any]:
    predictions = {record["id"]: record for record in prediction_rows}
    counters: defaultdict[str, int] = defaultdict(int)
    gold_distribution: Counter[int] = Counter()
    predicted_distribution: Counter[int] = Counter()
    confusion: Counter[str] = Counter()
    by_gold: dict[str, defaultdict[str, int]] = {
        str(count): defaultdict(int) for count in range(1, 6)
    }
    absolute_error = 0
    for record in gold_rows:
        gold_count = len(record["target"]["possible_outcomes"])
        gold_distribution[gold_count] += 1
        values = by_gold[str(gold_count)]
        values["examples"] += 1
        row = predictions.get(record["id"])
        if row is None:
            continue
        counters["covered"] += 1
        values["covered"] += 1
        predicted_count = normalize_count(row.get("prediction"))
        if predicted_count is None:
            continue
        counters["valid_prediction"] += 1
        values["valid_prediction"] += 1
        predicted_distribution[predicted_count] += 1
        confusion[f"gold_{gold_count}_pred_{predicted_count}"] += 1
        absolute_error += abs(predicted_count - gold_count)
        if (predicted_count == 1) == (gold_count == 1):
            counters["identifiability_correct"] += 1
        if gold_count > 1 and predicted_count > 1:
            counters["ambiguity_tp"] += 1
        elif gold_count == 1 and predicted_count > 1:
            counters["ambiguity_fp"] += 1
        elif gold_count > 1 and predicted_count == 1:
            counters["ambiguity_fn"] += 1
        else:
            counters["ambiguity_tn"] += 1
        if predicted_count == gold_count:
            counters["exact"] += 1
            values["exact"] += 1

    total = len(gold_rows)
    valid_predictions = counters["valid_prediction"]
    observed_gold_counts = [
        values for values in by_gold.values() if values["examples"] > 0
    ]
    ambiguous_examples = sum(
        values["examples"] for count, values in by_gold.items() if count != "1"
    )
    ambiguous_exact = sum(
        values["exact"] for count, values in by_gold.items() if count != "1"
    )
    ambiguity_tp = counters["ambiguity_tp"]
    ambiguity_fp = counters["ambiguity_fp"]
    ambiguity_fn = counters["ambiguity_fn"]
    ambiguity_tn = counters["ambiguity_tn"]
    ambiguity_precision = (
        ambiguity_tp / (ambiguity_tp + ambiguity_fp)
        if ambiguity_tp + ambiguity_fp
        else 0.0
    )
    ambiguity_recall = (
        ambiguity_tp / (ambiguity_tp + ambiguity_fn)
        if ambiguity_tp + ambiguity_fn
        else 0.0
    )
    identifiable_recall = (
        ambiguity_tn / (ambiguity_tn + ambiguity_fp)
        if ambiguity_tn + ambiguity_fp
        else 0.0
    )
    return {
        "examples": total,
        "coverage": counters["covered"] / total if total else 0.0,
        "valid_prediction_rate": valid_predictions / total if total else 0.0,
        "accuracy": counters["exact"] / total if total else 0.0,
        "macro_accuracy_by_observed_gold_count": (
            sum(values["exact"] / values["examples"] for values in observed_gold_counts)
            / len(observed_gold_counts)
            if observed_gold_counts
            else 0.0
        ),
        "ambiguous_count_accuracy": (
            ambiguous_exact / ambiguous_examples if ambiguous_examples else 0.0
        ),
        "identifiability_accuracy": (
            counters["identifiability_correct"] / total if total else 0.0
        ),
        "balanced_identifiability_accuracy": (identifiable_recall + ambiguity_recall) / 2,
        "ambiguity_detection": {
            "precision": ambiguity_precision,
            "recall": ambiguity_recall,
            "f1": (
                2 * ambiguity_precision * ambiguity_recall
                / (ambiguity_precision + ambiguity_recall)
                if ambiguity_precision + ambiguity_recall
                else 0.0
            ),
            "tp": ambiguity_tp,
            "fp": ambiguity_fp,
            "fn": ambiguity_fn,
            "tn": ambiguity_tn,
        },
        "accuracy_given_valid_prediction": (
            counters["exact"] / valid_predictions if valid_predictions else 0.0
        ),
        "mean_absolute_error_given_valid_prediction": (
            absolute_error / valid_predictions if valid_predictions else 0.0
        ),
        "gold_count_distribution": {
            str(key): value for key, value in sorted(gold_distribution.items())
        },
        "predicted_count_distribution": {
            str(key): value for key, value in sorted(predicted_distribution.items())
        },
        "confusion": dict(sorted(confusion.items())),
        "by_gold_count": {
            count: {
                "examples": values["examples"],
                "coverage": values["covered"] / values["examples"] if values["examples"] else 0.0,
                "accuracy": values["exact"] / values["examples"] if values["examples"] else 0.0,
            }
            for count, values in by_gold.items()
        },
    }


def main() -> None:
    args = parse_args()
    report = evaluate_counts(read_jsonl(Path(args.gold)), read_jsonl(Path(args.predictions)))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
