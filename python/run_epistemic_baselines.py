#!/usr/bin/env python3
"""Run deterministic baselines on the prompt-disjoint v2 agent task."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from audit_dataset import canonical, read_dataset
from evaluate_epistemic import evaluate_epistemic
from evaluate_predictions import SET_FIELDS
from run_baselines import feature_weights, weighted_jaccard


Record = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/v2/records/agent")
    parser.add_argument("--output-dir", default="outputs/baselines/v2-agent")
    parser.add_argument("--markdown", default="docs/v2-agent-baselines.md")
    return parser.parse_args()


def no_change_target(record: Record) -> dict[str, Any]:
    transition: dict[str, Any] = {field: [] for field in SET_FIELDS}
    transition.update(
        {
            "success": True,
            "next_location": record["agent_input"]["observation"]["location"],
            "reachable_room_delta": 0,
            "environment_changed": False,
            "flags_changed": {},
        }
    )
    return {"identifiable": True, "possible_outcomes": [transition]}


def no_change(train: list[Record], test: list[Record]) -> list[Record]:
    del train
    return [
        {"id": record["id"], "prediction": no_change_target(record), "baseline": "no_change"}
        for record in test
    ]


def action_majority(train: list[Record], test: list[Record]) -> list[Record]:
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for record in train:
        key = record["agent_input"]["candidate_action"]["key"]
        target = canonical(record["target"])
        counts[key][target] += 1
        global_counts[target] += 1
    global_target = choose(global_counts)
    majorities = {key: choose(values) for key, values in counts.items()}
    return [
        {
            "id": record["id"],
            "prediction": json.loads(
                majorities.get(record["agent_input"]["candidate_action"]["key"], global_target)
            ),
            "baseline": "action_majority",
        }
        for record in test
    ]


def nearest_neighbor(train: list[Record], test: list[Record]) -> list[Record]:
    index: defaultdict[str, list[tuple[Record, dict[str, float]]]] = defaultdict(list)
    for record in sorted(train, key=lambda value: value["id"]):
        key = record["agent_input"]["candidate_action"]["key"]
        index[key].append((record, feature_weights(record)))
    predictions = []
    for record in test:
        key = record["agent_input"]["candidate_action"]["key"]
        query = feature_weights(record)
        best: Record | None = None
        score = -1.0
        for candidate, features in index[key]:
            similarity = weighted_jaccard(query, features)
            if similarity > score:
                best, score = candidate, similarity
        if best is None:
            raise RuntimeError(f"No neighbor for {record['id']}")
        predictions.append(
            {
                "id": record["id"],
                "prediction": best["target"],
                "baseline": "nearest_neighbor",
                "source_id": best["id"],
                "similarity": score,
            }
        )
    return predictions


def choose(values: Counter[str]) -> str:
    return min(values, key=lambda value: (-values[value], value))


def write_jsonl(path: Path, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def render_markdown(summary: dict[str, Any], test_count: int) -> str:
    lines = [
        "# Dataset v2 agent baselines",
        "",
        f"Prompt-disjoint test set: {test_count:,} unique agent prompts.",
        "",
        "| Baseline | Exact target | Identifiability | Outcome-set exact | Outcome F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (
        ("no_change", "No change"),
        ("action_majority", "Action majority"),
        ("nearest_neighbor", "Nearest neighbour"),
    ):
        values = summary[name]
        lines.append(
            f"| {label} | {values['exact_match']:.2%} | "
            f"{values['identifiability_accuracy']:.2%} | "
            f"{values['outcome_set_exact_match']:.2%} | "
            f"{values['outcome_set_micro']['f1']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = read_dataset(Path(args.dataset))
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    generators: dict[str, Callable[[list[Record], list[Record]], list[Record]]] = {
        "no_change": no_change,
        "action_majority": action_majority,
        "nearest_neighbor": nearest_neighbor,
    }
    output_dir = Path(args.output_dir)
    summary = {}
    for name, generator in generators.items():
        predictions = generator(train, test)
        write_jsonl(output_dir / f"{name}.jsonl", predictions)
        metrics = evaluate_epistemic(test, predictions)
        (output_dir / f"{name}.metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary[name] = metrics
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(summary, len(test)), encoding="utf-8")
    print(json.dumps({name: {key: values[key] for key in (
        "exact_match", "identifiability_accuracy", "outcome_set_exact_match"
    )} for name, values in summary.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
