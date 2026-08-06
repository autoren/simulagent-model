#!/usr/bin/env python3
"""Audit the context-safe stratified v3 epistemic calibration dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_dataset import canonical, read_dataset


Record = dict[str, Any]
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/v3")
    parser.add_argument("--output", default="outputs/audit/v3-audit.json")
    parser.add_argument("--markdown", default="docs/v3-audit.md")
    return parser.parse_args()


def distributions(records: list[Record], selector: Any) -> dict[str, dict[str, int]]:
    return {
        split: dict(sorted(Counter(selector(record) for record in records if record["split"] == split).items()))
        for split in SPLITS
    }


def report(records: list[Record]) -> dict[str, Any]:
    prompt_splits: defaultdict[str, set[str]] = defaultdict(set)
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        prompt_splits[canonical(record["agent_input"])].add(record["split"])
        group_splits[record["split_group"]].add(record["split"])
    by_split = {}
    mechanic_shares: dict[str, dict[str, float]] = {}
    labels = sorted({label for record in records for label in record["mechanic_labels"]})
    for split in SPLITS:
        selected = [record for record in records if record["split"] == split]
        ambiguous = sum(not record["target"]["identifiable"] for record in selected)
        by_split[split] = {
            "records": len(selected),
            "groups": len({record["split_group"] for record in selected}),
            "identifiable": len(selected) - ambiguous,
            "ambiguous": ambiguous,
            "ambiguous_rate": ambiguous / len(selected),
        }
    for label in labels:
        mechanic_shares[label] = {
            split: sum(
                label in record["mechanic_labels"]
                for record in records
                if record["split"] == split
            )
            / by_split[split]["records"]
            for split in SPLITS
        }
    return {
        "records": len(records),
        "unique_prompts": len(prompt_splits),
        "prompt_cross_split_overlaps": sum(len(splits) > 1 for splits in prompt_splits.values()),
        "context_cross_split_overlaps": sum(len(splits) > 1 for splits in group_splits.values()),
        "by_split": by_split,
        "ambiguity_rate_gap": max(value["ambiguous_rate"] for value in by_split.values())
        - min(value["ambiguous_rate"] for value in by_split.values()),
        "outcome_count_distribution": distributions(
            records, lambda record: str(len(record["target"]["possible_outcomes"]))
        ),
        "action_family_distribution": distributions(
            records, lambda record: record["agent_input"]["candidate_action"]["key"].split(":", 1)[0]
        ),
        "mechanic_shares": mechanic_shares,
        "mechanic_share_max_gap": max(
            max(shares.values()) - min(shares.values()) for shares in mechanic_shares.values()
        ),
    }


def render_markdown(values: dict[str, Any]) -> str:
    lines = [
        "# Dataset v3 stratification audit",
        "",
        "V3 assigns whole observation-context groups while constraining ambiguity, outcome-count,",
        "action-family, scenario-family, and supported mechanic-tag distributions.",
        "",
        "| Gate | Result |",
        "| --- | ---: |",
        f"| Unique prompts | {values['unique_prompts']:,} / {values['records']:,} |",
        f"| Prompt overlaps | {values['prompt_cross_split_overlaps']} |",
        f"| Context overlaps | {values['context_cross_split_overlaps']} |",
        f"| Ambiguity-rate max gap | {values['ambiguity_rate_gap']:.2%} |",
        f"| Mechanic-share max gap | {values['mechanic_share_max_gap']:.2%} |",
        "",
        "## Split composition",
        "",
        "| Split | Records | Context groups | Identifiable | Ambiguous | Ambiguity rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        row = values["by_split"][split]
        lines.append(
            f"| {split} | {row['records']:,} | {row['groups']:,} | {row['identifiable']:,} | "
            f"{row['ambiguous']:,} | {row['ambiguous_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The current test split remains diagnostic because earlier experiments informed the V3",
            "methodology. A newly generated untouched holdout is required for a final claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    values = report(read_dataset(Path(args.dataset) / "records" / "agent"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(values), encoding="utf-8")
    print(json.dumps(values, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
