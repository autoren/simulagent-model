#!/usr/bin/env python3
"""Audit the epistemic-agent and Markov-privileged v2 tracks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_dataset import canonical, read_dataset


Record = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/v2")
    parser.add_argument("--output", default="outputs/audit/v2-audit.json")
    parser.add_argument("--markdown", default="docs/v2-audit.md")
    return parser.parse_args()


def split_counts(records: list[Record]) -> dict[str, int]:
    return dict(sorted(Counter(record["split"] for record in records).items()))


def cross_split_count(records: list[Record], input_key: str) -> int:
    splits: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        splits[canonical(record[input_key])].add(record["split"])
    return sum(len(values) > 1 for values in splits.values())


def test_seen_rate(records: list[Record], input_key: str) -> float:
    train = {
        canonical(record[input_key]) for record in records if record["split"] == "train"
    }
    test = [record for record in records if record["split"] == "test"]
    return sum(canonical(record[input_key]) in train for record in test) / len(test)


def audit_agent(records: list[Record]) -> dict[str, Any]:
    ambiguous = [record for record in records if not record["target"]["identifiable"]]
    varying_fields: Counter[str] = Counter()
    outcome_counts: Counter[int] = Counter()
    for record in records:
        outcomes = record["target"]["possible_outcomes"]
        outcome_counts[len(outcomes)] += 1
        if len(outcomes) > 1:
            for field in outcomes[0]:
                if len({canonical(outcome[field]) for outcome in outcomes}) > 1:
                    varying_fields[field] += 1
    split_identifiability = {}
    for split in ("train", "valid", "test"):
        split_records = [record for record in records if record["split"] == split]
        identifiable = sum(record["target"]["identifiable"] for record in split_records)
        split_identifiability[split] = {
            "records": len(split_records),
            "identifiable": identifiable,
            "ambiguous": len(split_records) - identifiable,
            "ambiguous_rate": (len(split_records) - identifiable) / len(split_records),
        }
    return {
        "records": len(records),
        "counts": split_counts(records),
        "unique_prompts": len({canonical(record["agent_input"]) for record in records}),
        "prompt_cross_split_overlaps": cross_split_count(records, "agent_input"),
        "test_prompt_seen_in_train_rate": test_seen_rate(records, "agent_input"),
        "identifiable_records": len(records) - len(ambiguous),
        "identifiable_rate": (len(records) - len(ambiguous)) / len(records),
        "ambiguous_records": len(ambiguous),
        "identifiability_by_split": split_identifiability,
        "possible_outcome_count_distribution": {
            str(key): value for key, value in sorted(outcome_counts.items())
        },
        "fields_varying_within_ambiguous_prompts": dict(varying_fields.most_common()),
        "source_record_count": sum(record["source_record_count"] for record in records),
    }


def audit_privileged(records: list[Record]) -> dict[str, Any]:
    prompts: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        prompts[canonical(record["privileged_input"])].add(canonical(record["target"]))
    return {
        "records": len(records),
        "counts": split_counts(records),
        "unique_prompts": len(prompts),
        "contradictory_prompts": sum(len(targets) > 1 for targets in prompts.values()),
        "prompt_cross_split_overlaps": cross_split_count(records, "privileged_input"),
        "test_prompt_seen_in_train_rate": test_seen_rate(records, "privileged_input"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    agent = report["agent"]
    privileged = report["privileged"]
    lines = [
        "# Dataset v2 audit",
        "",
        "## Gate results",
        "",
        "| Check | Result |",
        "| --- | ---: |",
        f"| Agent unique prompts | {agent['unique_prompts']:,} / {agent['records']:,} |",
        f"| Agent exact prompts crossing splits | {agent['prompt_cross_split_overlaps']} |",
        f"| Agent test prompts seen in training | {agent['test_prompt_seen_in_train_rate']:.2%} |",
        f"| Agent identifiable prompts | {agent['identifiable_rate']:.2%} |",
        f"| Privileged contradictory prompts | {privileged['contradictory_prompts']} |",
        f"| Privileged exact prompts crossing splits | {privileged['prompt_cross_split_overlaps']} |",
        f"| Privileged test prompts seen in training | {privileged['test_prompt_seen_in_train_rate']:.2%} |",
        "",
        "Agent ambiguity is now represented as a target property rather than contradictory rows.",
        "The privileged track contains explicit transition rules and is empirically Markov-complete",
        "for the current target schema. Both tracks use prompt-disjoint context splits.",
        "",
        "## Agent identifiability by split",
        "",
        "| Split | Prompts | Identifiable | Ambiguous | Ambiguous rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, values in agent["identifiability_by_split"].items():
        lines.append(
            f"| {split} | {values['records']:,} | {values['identifiable']:,} | "
            f"{values['ambiguous']:,} | {values['ambiguous_rate']:.2%} |"
        )
    lines.extend(
        [
        "",
        "## Agent possible-outcome counts",
        "",
        "| Outcomes | Prompts |",
        "| ---: | ---: |",
        ]
    )
    for count, records in agent["possible_outcome_count_distribution"].items():
        lines.append(f"| {count} | {records:,} |")
    lines.extend(
        [
            "",
            "## Fields varying in ambiguous prompts",
            "",
            "| Field | Prompt groups |",
            "| --- | ---: |",
        ]
    )
    for field, count in agent["fields_varying_within_ambiguous_prompts"].items():
        lines.append(f"| {field} | {count:,} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset) / "records"
    agent = read_dataset(root / "agent")
    privileged = read_dataset(root / "privileged")
    report = {"agent": audit_agent(agent), "privileged": audit_privileged(privileged)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
