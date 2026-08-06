#!/usr/bin/env python3
"""Evaluate canonical Simulagent transition predictions against oracle records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SET_FIELDS = (
    "inventory_added",
    "inventory_removed",
    "visible_actions_added",
    "visible_actions_removed",
    "blocked_actions_added",
    "blocked_actions_removed",
    "hidden_actions_revealed",
    "hidden_actions_concealed",
)
SCALAR_FIELDS = (
    "success",
    "next_location",
    "reachable_room_delta",
    "environment_changed",
)
REQUIRED_FIELDS = set(SCALAR_FIELDS + ("flags_changed",) + SET_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/pilot/records/test.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--stratify",
        action="append",
        choices=("scenario_id", "scenario_family", "split_group", "action_type", "target_change"),
        default=[],
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_prediction(value: Any) -> dict[str, Any] | None:
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


def safe_set(value: Any) -> set[str]:
    return {str(item) for item in value} if isinstance(value, list) else set()


def main() -> None:
    args = parse_args()
    gold_rows = read_jsonl(Path(args.gold))
    prediction_rows = read_jsonl(Path(args.predictions))
    report = evaluate_records(gold_rows, prediction_rows)
    if args.stratify:
        predictions = {row["id"]: row for row in prediction_rows}
        report["stratified"] = {
            dimension: stratified_report(gold_rows, predictions, dimension)
            for dimension in args.stratify
        }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def evaluate_records(
    gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gold = {row["id"]: row for row in gold_rows}
    predictions = {row["id"]: row for row in prediction_rows}
    counts: defaultdict[str, int] = defaultdict(int)
    set_counts = {field: defaultdict(int) for field in SET_FIELDS}
    field_exact = defaultdict(int)

    for record_id, record in gold.items():
        counts["total"] += 1
        row = predictions.get(record_id)
        prediction = normalize_prediction(row.get("prediction")) if row else None
        if prediction is None:
            continue
        counts["json_valid"] += 1
        target = record["target"]
        schema_valid = valid_schema(prediction)
        if schema_valid:
            counts["schema_valid"] += 1
        if schema_valid and prediction == target:
            counts["exact"] += 1
        for field in SCALAR_FIELDS + ("flags_changed",):
            if prediction.get(field) == target.get(field):
                field_exact[field] += 1
        for field in SET_FIELDS:
            predicted = safe_set(prediction.get(field))
            expected = safe_set(target.get(field))
            set_counts[field]["tp"] += len(predicted & expected)
            set_counts[field]["fp"] += len(predicted - expected)
            set_counts[field]["fn"] += len(expected - predicted)
            if isinstance(prediction.get(field), list) and predicted == expected:
                field_exact[field] += 1

    total = counts["total"]
    report: dict[str, Any] = {
        "examples": total,
        "coverage": len(set(gold) & set(predictions)) / total if total else 0,
        "json_valid_rate": counts["json_valid"] / total if total else 0,
        "schema_valid_rate": counts["schema_valid"] / total if total else 0,
        "exact_match": counts["exact"] / total if total else 0,
        "exact_match_given_schema_valid": (
            counts["exact"] / counts["schema_valid"] if counts["schema_valid"] else 0
        ),
        "field_exact": {
            field: field_exact[field] / total if total else 0
            for field in SCALAR_FIELDS + ("flags_changed",) + SET_FIELDS
        },
        "set_metrics": {
            field: precision_recall(values) for field, values in set_counts.items()
        },
    }
    report["macro_field_exact"] = sum(report["field_exact"].values()) / len(
        report["field_exact"]
    )
    return report


def stratified_report(
    gold_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gold_rows:
        groups[str(stratum_value(record, dimension))].append(record)
    return {
        name: evaluate_records(rows, [predictions[row["id"]] for row in rows if row["id"] in predictions])
        for name, rows in sorted(groups.items())
    }


def stratum_value(record: dict[str, Any], dimension: str) -> str:
    if dimension == "action_type":
        return str(record["action"]["type"])
    if dimension == "target_change":
        return "changed" if target_has_state_change(record["target"]) else "unchanged"
    return str(record[dimension])


def target_has_state_change(target: dict[str, Any]) -> bool:
    return bool(
        target["flags_changed"]
        or target["inventory_added"]
        or target["inventory_removed"]
        or target["visible_actions_added"]
        or target["visible_actions_removed"]
        or target["blocked_actions_added"]
        or target["blocked_actions_removed"]
        or target["hidden_actions_revealed"]
        or target["hidden_actions_concealed"]
        or target["reachable_room_delta"] != 0
        or target["environment_changed"]
    )


def precision_recall(values: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = values["tp"], values["fp"], values["fn"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "positive_support": tp + fn,
    }


def valid_schema(value: dict[str, Any]) -> bool:
    if set(value) != REQUIRED_FIELDS:
        return False
    if not isinstance(value["success"], bool):
        return False
    if not isinstance(value["environment_changed"], bool):
        return False
    if not isinstance(value["next_location"], str):
        return False
    if isinstance(value["reachable_room_delta"], bool) or not isinstance(
        value["reachable_room_delta"], int
    ):
        return False
    if not isinstance(value["flags_changed"], dict) or not all(
        isinstance(key, str) and isinstance(flag, bool)
        for key, flag in value["flags_changed"].items()
    ):
        return False
    return all(
        isinstance(value[field], list)
        and all(isinstance(item, str) for item in value[field])
        and len(value[field]) == len(set(value[field]))
        for field in SET_FIELDS
    )


if __name__ == "__main__":
    main()
