#!/usr/bin/env python3
"""Generate and evaluate deterministic transition-prediction baselines."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_dataset import canonical, read_dataset
from evaluate_predictions import SET_FIELDS, evaluate_records, stratified_report


Record = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/full")
    parser.add_argument("--output-dir", default="outputs/baselines/full")
    parser.add_argument("--markdown", default="docs/baseline-results.md")
    parser.add_argument("--track", choices=("agent", "privileged"), default="agent")
    return parser.parse_args()


def empty_transition(record: Record, success: bool) -> dict[str, Any]:
    target: dict[str, Any] = {field: [] for field in SET_FIELDS}
    target.update(
        {
            "success": success,
            "next_location": record["agent_input"]["observation"]["location"],
            "reachable_room_delta": 0,
            "environment_changed": False,
            "flags_changed": {},
        }
    )
    return target


def no_change_predictions(train: list[Record], test: list[Record]) -> list[Record]:
    successes = Counter(bool(row["target"]["success"]) for row in train)
    majority_success = successes[True] >= successes[False]
    return [
        {
            "id": row["id"],
            "prediction": empty_transition(row, majority_success),
            "baseline": "no_change",
        }
        for row in test
    ]


def action_majority_predictions(train: list[Record], test: list[Record]) -> list[Record]:
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in train:
        target = canonical(row["target"])
        key = row["agent_input"]["candidate_action"]["key"]
        counts[key][target] += 1
        global_counts[target] += 1
    global_target = choose_majority(global_counts)
    majority = {key: choose_majority(values) for key, values in counts.items()}
    return [
        {
            "id": row["id"],
            "prediction": json.loads(
                majority.get(row["agent_input"]["candidate_action"]["key"], global_target)
            ),
            "baseline": "action_majority",
        }
        for row in test
    ]


def prompt_lookup_predictions(
    train: list[Record], test: list[Record], track: str = "agent"
) -> list[Record]:
    prompt_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    action_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in train:
        target = canonical(row["target"])
        prompt_counts[canonical(model_input(row, track))][target] += 1
        action_counts[row["agent_input"]["candidate_action"]["key"]][target] += 1
        global_counts[target] += 1
    prompt_majority = {key: choose_majority(values) for key, values in prompt_counts.items()}
    action_majority = {key: choose_majority(values) for key, values in action_counts.items()}
    global_target = choose_majority(global_counts)
    predictions = []
    for row in test:
        prompt = canonical(model_input(row, track))
        action = row["agent_input"]["candidate_action"]["key"]
        target = prompt_majority.get(prompt)
        predictions.append(
            {
                "id": row["id"],
                "prediction": json.loads(
                    target or action_majority.get(action, global_target)
                ),
                "baseline": "prompt_lookup",
                "lookup_hit": target is not None,
            }
        )
    return predictions


def choose_majority(values: Counter[str]) -> str:
    return min(values, key=lambda value: (-values[value], value))


def nearest_neighbor_predictions(
    train: list[Record], test: list[Record], track: str = "agent"
) -> list[Record]:
    by_action: defaultdict[str, list[tuple[Record, dict[str, float]]]] = defaultdict(list)
    by_type: defaultdict[str, list[tuple[Record, dict[str, float]]]] = defaultdict(list)
    for row in sorted(train, key=lambda item: item["id"]):
        encoded = (row, feature_weights(row, track))
        by_action[row["agent_input"]["candidate_action"]["key"]].append(encoded)
        by_type[row["action"]["type"]].append(encoded)

    predictions = []
    for row in test:
        key = row["agent_input"]["candidate_action"]["key"]
        candidates = by_action.get(key) or by_type[row["action"]["type"]]
        query = feature_weights(row, track)
        best_row: Record | None = None
        best_score = -1.0
        for candidate, features in candidates:
            score = weighted_jaccard(query, features)
            if score > best_score:
                best_row = candidate
                best_score = score
        if best_row is None:
            raise RuntimeError(f"No nearest-neighbor candidate for {row['id']}")
        predictions.append(
            {
                "id": row["id"],
                "prediction": best_row["target"],
                "baseline": "nearest_neighbor",
                "source_id": best_row["id"],
                "similarity": best_score,
            }
        )
    return predictions


def feature_weights(record: Record, track: str = "agent") -> dict[str, float]:
    value = record["agent_input"]
    observation = value["observation"]
    features: dict[str, float] = {
        f"action:{value['candidate_action']['key']}": 8.0,
        f"location:{observation['location']}": 5.0,
        f"turn:{observation['turn']}": 2.0,
        f"pressure:{observation['pressure']}": 1.5,
        f"signal:{observation['signal']}": 1.5,
    }
    for item in observation["inventory"]:
        features[f"inventory:{item}"] = 3.0
    for item in observation["visibleObjects"]:
        features[f"object:{item['id']}:{item['portable']}"] = 2.0
    for item in observation["exits"]:
        features[f"exit:{item['direction']}:{item['roomName']}:{item['blocked']}"] = 2.0
    for item in value["available_actions"]:
        features[f"available:{item['key']}"] = 1.0
    for field in ("beliefs", "memories"):
        for token in text_tokens(" ".join(observation[field])):
            features[f"{field}:{token}"] = 0.25
    if track == "privileged":
        privileged = record["privileged_input"]
        world = privileged["privileged_world_state"]
        for key, state in world["flags"].items():
            features[f"flag:{key}:{state}"] = 3.0
        for room, runtime in world["rooms"].items():
            features[f"room:{room}:visited:{runtime['visited']}"] = 1.5
            for object_id, object_state in runtime["objects"].items():
                features[
                    f"runtime:{room}:{object_id}:{object_state['visible']}:{object_state['taken']}"
                ] = 1.0
        for key, rule in privileged.get("transition_rules", {}).items():
            features[f"rule:{key}:{rule}"] = 4.0
    return features


def model_input(record: Record, track: str) -> dict[str, Any]:
    return record["privileged_input"] if track == "privileged" else record["agent_input"]


def text_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def weighted_jaccard(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    union = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    intersection = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return intersection / union if union else 1.0


def write_jsonl(path: Path, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = read_dataset(Path(args.dataset))
    train = [row for row in records if row["split"] == "train"]
    test = [row for row in records if row["split"] == "test"]
    output_dir = Path(args.output_dir)
    generators = {
        "no_change": no_change_predictions,
        "action_majority": action_majority_predictions,
        "prompt_lookup": lambda train_rows, test_rows: prompt_lookup_predictions(
            train_rows, test_rows, args.track
        ),
        "nearest_neighbor": lambda train_rows, test_rows: nearest_neighbor_predictions(
            train_rows, test_rows, args.track
        ),
    }
    summary: dict[str, Any] = {}
    for name, generator in generators.items():
        predictions = generator(train, test)
        write_jsonl(output_dir / f"{name}.jsonl", predictions)
        prediction_map = {row["id"]: row for row in predictions}
        metrics = evaluate_records(test, predictions)
        metrics["stratified"] = {
            dimension: stratified_report(test, prediction_map, dimension)
            for dimension in (
                "target_change",
                "action_type",
                "scenario_family",
                "scenario_id",
            )
        }
        (output_dir / f"{name}.metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary[name] = {
            "exact_match": metrics["exact_match"],
            "macro_field_exact": metrics["macro_field_exact"],
            "changed_exact_match": metrics["stratified"]["target_change"]["changed"][
                "exact_match"
            ],
            "unchanged_exact_match": metrics["stratified"]["target_change"][
                "unchanged"
            ]["exact_match"],
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    lookup_rows = [json.loads(line) for line in (output_dir / "prompt_lookup.jsonl").read_text().splitlines()]
    lookup_hit_rate = sum(bool(row["lookup_hit"]) for row in lookup_rows) / len(lookup_rows)
    markdown.write_text(
        render_markdown(summary, len(train), len(test), lookup_hit_rate, args.track),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def render_markdown(
    summary: dict[str, Any],
    train_count: int,
    test_count: int,
    lookup_hit_rate: float,
    track: str,
) -> str:
    labels = {
        "no_change": "No change",
        "action_majority": "Action majority",
        "prompt_lookup": "Exact-prompt lookup",
        "nearest_neighbor": "Nearest neighbour",
    }
    lines = [
        "# Deterministic baseline results",
        "",
        f"Track: `{track}`. Evaluated on all {test_count:,} test transitions using {train_count:,} training transitions.",
        "",
        "| Baseline | Exact match | Changed targets | Unchanged targets | Macro field accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("no_change", "action_majority", "prompt_lookup", "nearest_neighbor"):
        values = summary[name]
        lines.append(
            f"| {labels[name]} | {values['exact_match']:.2%} | "
            f"{values['changed_exact_match']:.2%} | {values['unchanged_exact_match']:.2%} | "
            f"{values['macro_field_exact']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Exact-prompt training coverage of test: {lookup_hit_rate:.2%}.",
            "When this coverage is high, exact-prompt lookup measures input duplication rather than",
            "generalization. The no-change result shows why changed and unchanged targets must always be",
            "reported separately.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
