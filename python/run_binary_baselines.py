#!/usr/bin/env python3
"""Run calibrated token Naive Bayes shortcut ablations for V4."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from binary_metrics import evaluate_binary, fit_threshold


Record = dict[str, Any]
TOKEN_PATTERN = re.compile(r"[a-z0-9_:+.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/v4/records")
    parser.add_argument("--output-dir", default="outputs/baselines/v4-binary")
    parser.add_argument("--markdown", default="docs/v4-binary-baselines.md")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def input_variant(record: Record, variant: str) -> Record:
    value = copy.deepcopy(record["agent_input"])
    observation = value["observation"]
    if variant in {"no_turn", "no_state_scalars", "no_state_scalars_or_history"}:
        observation.pop("turn", None)
    if variant in {"no_pressure_signal", "no_state_scalars", "no_state_scalars_or_history"}:
        observation.pop("pressure", None)
        observation.pop("signal", None)
    if variant in {"no_history", "no_state_scalars_or_history"}:
        value.pop("recent_history", None)
        observation.pop("memories", None)
    return value


def tokenize(value: Any) -> list[str]:
    return TOKEN_PATTERN.findall(json.dumps(value, sort_keys=True).lower())


def train_token_nb(train: list[Record], variant: str) -> dict[str, Any]:
    counts: dict[bool, Counter[str]] = {False: Counter(), True: Counter()}
    totals = {False: 0, True: 0}
    vocabulary: set[str] = set()
    for record in train:
        ambiguous = not record["target"]["identifiable"]
        tokens = tokenize(input_variant(record, variant))
        counts[ambiguous].update(tokens)
        totals[ambiguous] += len(tokens)
        vocabulary.update(tokens)
    return {"counts": counts, "totals": totals, "vocabulary": vocabulary}


def score_token_nb(model: dict[str, Any], records: list[Record], variant: str) -> list[Record]:
    smoothing = 1.0
    vocabulary = model["vocabulary"]
    denominator = {
        label: model["totals"][label] + smoothing * len(vocabulary) for label in (False, True)
    }
    rows = []
    for record in records:
        tokens = Counter(tokenize(input_variant(record, variant)))
        scores = {}
        for label in (False, True):
            scores[label] = sum(
                count
                * math.log(
                    (model["counts"][label][token] + smoothing) / denominator[label]
                )
                for token, count in tokens.items()
            )
        rows.append(
            {
                "id": record["id"],
                "gold_ambiguous": not record["target"]["identifiable"],
                "score": scores[True] - scores[False],
            }
        )
    return rows


def metric(rows: list[Record], threshold: float) -> dict[str, Any]:
    return evaluate_binary(
        [row["gold_ambiguous"] for row in rows],
        [row["score"] for row in rows],
        threshold,
    )


def render_markdown(summary: dict[str, Any]) -> str:
    labels = {
        "full": "Full visible input",
        "no_turn": "Remove turn",
        "no_pressure_signal": "Remove pressure + signal",
        "no_history": "Remove history + memories",
        "no_state_scalars": "Remove turn + pressure + signal",
        "no_state_scalars_or_history": "Remove scalars + history",
    }
    lines = [
        "# V4 binary token-baseline ablations",
        "",
        "Thresholds are fitted once on the context-disjoint calibration fold. Validation is not",
        "used for fitting or variant selection. V3 test remains closed.",
        "",
        "| Input | Calibration balanced | Validation balanced | Validation F1 | Validation AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, label in labels.items():
        result = summary[name]
        lines.append(
            f"| {label} | {result['calibration']['balanced_accuracy']:.2%} | "
            f"{result['validation']['balanced_accuracy']:.2%} | "
            f"{result['validation']['ambiguity']['f1']:.2%} | "
            f"{result['validation']['roc_auc']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset)
    train = read_jsonl(root / "train.jsonl")
    calibration = read_jsonl(root / "calibration.jsonl")
    validation = read_jsonl(root / "validation.jsonl")
    variants = [
        "full",
        "no_turn",
        "no_pressure_signal",
        "no_history",
        "no_state_scalars",
        "no_state_scalars_or_history",
    ]
    output = Path(args.output_dir)
    summary = {}
    for variant in variants:
        model = train_token_nb(train, variant)
        calibration_rows = score_token_nb(model, calibration, variant)
        validation_rows = score_token_nb(model, validation, variant)
        fitted = fit_threshold(
            [row["gold_ambiguous"] for row in calibration_rows],
            [row["score"] for row in calibration_rows],
        )
        threshold = fitted["threshold"]
        result = {
            "variant": variant,
            "threshold_selection_split": "calibration",
            "threshold": threshold,
            "calibration": fitted,
            "validation": metric(validation_rows, threshold),
        }
        summary[variant] = result
        write_jsonl(output / variant / "calibration.scores.jsonl", calibration_rows)
        write_jsonl(output / variant / "validation.scores.jsonl", validation_rows)
        (output / variant / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
