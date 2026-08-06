#!/usr/bin/env python3
"""Run deterministic baselines for the compact outcome-count task."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from audit_dataset import read_dataset
from evaluate_outcome_count import evaluate_counts
from run_baselines import feature_weights, weighted_jaccard


Record = dict[str, Any]
TOKEN_PATTERN = re.compile(r"[a-z_]+|\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/v2/records/agent")
    parser.add_argument("--output-dir", default="outputs/baselines/v2-outcome-count")
    parser.add_argument("--markdown", default="docs/v2-outcome-count-baselines.md")
    parser.add_argument("--title", default="Dataset v2 outcome-count baselines")
    parser.add_argument("--evaluation-split", choices=("valid", "test"), default="test")
    return parser.parse_args()


def prediction(record: Record, count: int, baseline: str) -> Record:
    return {
        "id": record["id"],
        "prediction": {"outcome_count": count},
        "baseline": baseline,
    }


def always_one(_train: list[Record], test: list[Record]) -> list[Record]:
    return [prediction(record, 1, "always_one") for record in test]


def action_majority(train: list[Record], test: list[Record]) -> list[Record]:
    counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    global_counts: Counter[int] = Counter()
    for record in train:
        action = record["agent_input"]["candidate_action"]["key"]
        outcome_count = len(record["target"]["possible_outcomes"])
        counts[action][outcome_count] += 1
        global_counts[outcome_count] += 1
    global_mode = choose(global_counts)
    modes = {action: choose(values) for action, values in counts.items()}
    return [
        prediction(
            record,
            modes.get(record["agent_input"]["candidate_action"]["key"], global_mode),
            "action_majority",
        )
        for record in test
    ]


def nearest_neighbor(train: list[Record], test: list[Record]) -> list[Record]:
    index: defaultdict[str, list[tuple[Record, dict[str, float]]]] = defaultdict(list)
    for record in sorted(train, key=lambda value: value["id"]):
        action = record["agent_input"]["candidate_action"]["key"]
        index[action].append((record, feature_weights(record)))
    predictions = []
    for record in test:
        action = record["agent_input"]["candidate_action"]["key"]
        query = feature_weights(record)
        best: Record | None = None
        score = -1.0
        for candidate, features in index[action]:
            similarity = weighted_jaccard(query, features)
            if similarity > score:
                best, score = candidate, similarity
        if best is None:
            raise RuntimeError(f"No neighbor for {record['id']}")
        row = prediction(record, len(best["target"]["possible_outcomes"]), "nearest_neighbor")
        row.update({"source_id": best["id"], "similarity": score})
        predictions.append(row)
    return predictions


def token_naive_bayes(train: list[Record], test: list[Record]) -> list[Record]:
    token_counts: dict[bool, Counter[str]] = {False: Counter(), True: Counter()}
    token_totals = {False: 0, True: 0}
    vocabulary: set[str] = set()
    ambiguous_counts: Counter[int] = Counter()
    for record in train:
        ambiguous = not record["target"]["identifiable"]
        tokens = tokenize(record["agent_input"])
        token_counts[ambiguous].update(tokens)
        token_totals[ambiguous] += len(tokens)
        vocabulary.update(tokens)
        if ambiguous:
            ambiguous_counts[len(record["target"]["possible_outcomes"])] += 1
    ambiguous_mode = choose(ambiguous_counts)
    smoothing = 1.0
    denominator = {
        label: token_totals[label] + smoothing * len(vocabulary) for label in (False, True)
    }
    predictions = []
    for record in test:
        tokens = Counter(tokenize(record["agent_input"]))
        scores = {}
        for label in (False, True):
            scores[label] = sum(
                count
                * math.log(
                    (token_counts[label][token] + smoothing) / denominator[label]
                )
                for token, count in tokens.items()
            )
        ambiguous = scores[True] > scores[False]
        predictions.append(
            prediction(
                record,
                ambiguous_mode if ambiguous else 1,
                "token_naive_bayes",
            )
        )
    return predictions


def tokenize(value: Any) -> list[str]:
    return TOKEN_PATTERN.findall(json.dumps(value, sort_keys=True).lower())


def choose(values: Counter[int]) -> int:
    return min(values, key=lambda value: (-values[value], value))


def write_jsonl(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def render_markdown(
    summary: dict[str, Any], evaluation_count: int, title: str, evaluation_split: str
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Prompt-disjoint {evaluation_split} set: {evaluation_count:,} unique agent prompts.",
        "",
        "| Baseline | Exact count | Macro count | Ambiguous exact | Balanced ID | Ambiguity F1 | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (
        ("always_one", "Always one"),
        ("action_majority", "Action count majority"),
        ("nearest_neighbor", "Nearest neighbour"),
        ("token_naive_bayes", "Token Naive Bayes"),
    ):
        report = summary[name]
        lines.append(
            f"| {label} | {report['accuracy']:.2%} | "
            f"{report['macro_accuracy_by_observed_gold_count']:.2%} | "
            f"{report['ambiguous_count_accuracy']:.2%} | "
            f"{report['balanced_identifiability_accuracy']:.2%} | "
            f"{report['ambiguity_detection']['f1']:.2%} | "
            f"{report['mean_absolute_error_given_valid_prediction']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = read_dataset(Path(args.dataset))
    train = [record for record in records if record["split"] == "train"]
    evaluation = [record for record in records if record["split"] == args.evaluation_split]
    generators: dict[str, Callable[[list[Record], list[Record]], list[Record]]] = {
        "always_one": always_one,
        "action_majority": action_majority,
        "nearest_neighbor": nearest_neighbor,
        "token_naive_bayes": token_naive_bayes,
    }
    output_dir = Path(args.output_dir)
    summary = {}
    for name, generator in generators.items():
        predictions = generator(train, evaluation)
        write_jsonl(output_dir / f"{name}.jsonl", predictions)
        report = evaluate_counts(evaluation, predictions)
        (output_dir / f"{name}.metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary[name] = report
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        render_markdown(summary, len(evaluation), args.title, args.evaluation_split),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                name: {
                    "accuracy": report["accuracy"],
                    "mean_absolute_error": report["mean_absolute_error_given_valid_prediction"],
                }
                for name, report in summary.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
