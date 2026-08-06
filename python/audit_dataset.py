#!/usr/bin/env python3
"""Audit Simulagent transition data before fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from evaluate_predictions import SET_FIELDS, target_has_state_change


Record = dict[str, Any]
Signature = Callable[[Record], Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/full")
    parser.add_argument("--output", default="outputs/audit/full-audit.json")
    parser.add_argument("--markdown", default="docs/dataset-audit.md")
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_dataset(path: Path) -> list[Record]:
    records: list[Record] = []
    records_dir = path / "records" if (path / "records").is_dir() else path
    for split in ("train", "valid", "test"):
        file = records_dir / f"{split}.jsonl"
        with file.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def exact_prompt_signature(record: Record) -> Any:
    return record["agent_input"]


def exact_privileged_prompt_signature(record: Record) -> Any:
    return record["privileged_input"]


def observation_action_signature(record: Record) -> Any:
    value = record["agent_input"]
    return {
        "observation": value["observation"],
        "candidate_action": value["candidate_action"],
        "available_actions": value["available_actions"],
    }


def structural_signature(record: Record) -> Any:
    value = record["agent_input"]
    observation = value["observation"]
    return {
        "candidate_action": value["candidate_action"]["key"],
        "location": observation["location"],
        "turn": observation["turn"],
        "inventory": sorted(observation["inventory"]),
        "pressure": observation["pressure"],
        "signal": observation["signal"],
        "visible_objects": sorted(item["id"] for item in observation["visibleObjects"]),
        "exits": sorted(
            (item["direction"], item["roomName"], item["blocked"])
            for item in observation["exits"]
        ),
        "available_actions": sorted(item["key"] for item in value["available_actions"]),
    }


def audit_signature(records: list[Record], signature: Signature) -> dict[str, Any]:
    groups: defaultdict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[canonical(signature(record))].append(record)

    ambiguous_groups: list[tuple[str, list[Record], Counter[str]]] = []
    cross_split_groups = 0
    repeated_groups = 0
    duplicate_rows = 0
    best_possible = 0
    conditional_entropy = 0.0

    for key, rows in groups.items():
        target_counts = Counter(canonical(row["target"]) for row in rows)
        best_possible += max(target_counts.values())
        if len(rows) > 1:
            repeated_groups += 1
            duplicate_rows += len(rows) - 1
        if len({row["split"] for row in rows}) > 1:
            cross_split_groups += 1
        if len(target_counts) > 1:
            ambiguous_groups.append((key, rows, target_counts))
        group_entropy = -sum(
            (count / len(rows)) * math.log2(count / len(rows))
            for count in target_counts.values()
        )
        conditional_entropy += (len(rows) / len(records)) * group_entropy

    ambiguous_records = sum(len(rows) for _, rows, _ in ambiguous_groups)
    train_targets: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if record["split"] == "train":
            train_targets[canonical(signature(record))][canonical(record["target"])] += 1
    test_rows = [record for record in records if record["split"] == "test"]
    test_signatures = {canonical(signature(record)) for record in test_rows}
    seen_test_signatures = test_signatures & set(train_targets)
    seen_test_rows = [
        record for record in test_rows if canonical(signature(record)) in train_targets
    ]
    train_majority_correct = 0
    target_seen_in_train = 0
    for record in seen_test_rows:
        key = canonical(signature(record))
        target = canonical(record["target"])
        counts = train_targets[key]
        majority = min(counts, key=lambda value: (-counts[value], value))
        train_majority_correct += int(target == majority)
        target_seen_in_train += int(target in counts)
    examples = []
    for key, rows, targets in sorted(
        ambiguous_groups, key=lambda item: (-len(item[1]), item[0])
    )[:10]:
        examples.append(
            {
                "signature_sha256": hashlib.sha256(key.encode()).hexdigest(),
                "records": len(rows),
                "distinct_targets": len(targets),
                "scenarios": sorted({row["scenario_id"] for row in rows}),
                "splits": sorted({row["split"] for row in rows}),
                "action": rows[0]["agent_input"]["candidate_action"]["key"],
            }
        )

    return {
        "unique_signatures": len(groups),
        "repeated_signature_groups": repeated_groups,
        "duplicate_signature_rows": duplicate_rows,
        "ambiguous_signature_groups": len(ambiguous_groups),
        "ambiguous_records": ambiguous_records,
        "ambiguous_record_rate": ambiguous_records / len(records),
        "cross_split_signature_groups": cross_split_groups,
        "conditional_target_entropy_bits": conditional_entropy,
        "signature_limited_exact_match_upper_bound": best_possible / len(records),
        "test_transfer": {
            "test_records": len(test_rows),
            "test_signatures": len(test_signatures),
            "signatures_seen_in_train": len(seen_test_signatures),
            "signature_seen_in_train_rate": (
                len(seen_test_signatures) / len(test_signatures) if test_signatures else 0
            ),
            "records_seen_in_train": len(seen_test_rows),
            "record_seen_in_train_rate": (
                len(seen_test_rows) / len(test_rows) if test_rows else 0
            ),
            "target_seen_in_train_rate_on_seen_records": (
                target_seen_in_train / len(seen_test_rows) if seen_test_rows else 0
            ),
            "train_majority_exact_match_on_seen_records": (
                train_majority_correct / len(seen_test_rows) if seen_test_rows else 0
            ),
        },
        "ambiguous_examples": examples,
    }


def audit_records(records: list[Record]) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in records)
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    scenario_splits: defaultdict[str, set[str]] = defaultdict(set)
    state_counts = Counter(
        (row["split"], row["scenario_id"], row["state_id"]) for row in records
    )
    for row in records:
        group_splits[row["split_group"]].add(row["split"])
        scenario_splits[row["scenario_id"]].add(row["split"])

    target_rates = {
        "success": mean(bool(row["target"]["success"]) for row in records),
        "environment_changed": mean(
            bool(row["target"]["environment_changed"]) for row in records
        ),
        "flags_changed": mean(bool(row["target"]["flags_changed"]) for row in records),
        "state_change": mean(target_has_state_change(row["target"]) for row in records),
    }
    target_rates.update(
        {
            f"{field}_nonempty": mean(bool(row["target"][field]) for row in records)
            for field in SET_FIELDS
        }
    )

    per_split = {}
    for split in ("train", "valid", "test"):
        rows = [row for row in records if row["split"] == split]
        per_split[split] = {
            "records": len(rows),
            "scenarios": len({row["scenario_id"] for row in rows}),
            "groups": len({row["split_group"] for row in rows}),
            "states": len({(row["scenario_id"], row["state_id"]) for row in rows}),
            "state_change_rate": mean(target_has_state_change(row["target"]) for row in rows),
            "success_rate": mean(bool(row["target"]["success"]) for row in rows),
        }

    return {
        "records": len(records),
        "splits": dict(sorted(split_counts.items())),
        "scenarios": len({row["scenario_id"] for row in records}),
        "split_groups": len({row["split_group"] for row in records}),
        "states": len(state_counts),
        "record_id_duplicates": len(records) - len({row["id"] for row in records}),
        "split_integrity": {
            "groups_crossing_splits": sorted(
                group for group, splits in group_splits.items() if len(splits) > 1
            ),
            "scenarios_crossing_splits": sorted(
                scenario for scenario, splits in scenario_splits.items() if len(splits) > 1
            ),
        },
        "per_split": per_split,
        "transitions_per_state": summarize_numbers(list(state_counts.values())),
        "action_type_counts": dict(
            sorted(Counter(row["action"]["type"] for row in records).items())
        ),
        "scenario_family_counts": dict(
            sorted(Counter(row["scenario_family"] for row in records).items())
        ),
        "target_rates": target_rates,
        "signatures": {
            "exact_agent_prompt": audit_signature(records, exact_prompt_signature),
            "exact_privileged_prompt": audit_signature(
                records, exact_privileged_prompt_signature
            ),
            "observation_action": audit_signature(records, observation_action_signature),
            "lossy_structural_observation_action": audit_signature(
                records, structural_signature
            ),
        },
    }


def mean(values: Any) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def summarize_numbers(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def percentile(ordered: list[int], quantile: float) -> int:
    return ordered[round((len(ordered) - 1) * quantile)]


def render_markdown(report: dict[str, Any], dataset: str) -> str:
    lines = [
        "# Dataset audit",
        "",
        f"Dataset: `{dataset}`",
        "",
        "## Corpus",
        "",
        f"- Records: {report['records']:,}",
        f"- Scenarios: {report['scenarios']}",
        f"- Split groups: {report['split_groups']}",
        f"- Reachable states: {report['states']:,}",
        f"- State-changing targets: {report['target_rates']['state_change']:.1%}",
        f"- Successful actions: {report['target_rates']['success']:.1%}",
        "",
        "## Split integrity",
        "",
        f"- Groups crossing splits: {len(report['split_integrity']['groups_crossing_splits'])}",
        f"- Scenarios crossing splits: {len(report['split_integrity']['scenarios_crossing_splits'])}",
        f"- Duplicate record IDs: {report['record_id_duplicates']}",
        "",
        "## Observational signatures",
        "",
        "| Signature | Unique | Ambiguous groups | Ambiguous records | Cross-split overlaps | Exact-match ceiling |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in report["signatures"].items():
        lines.append(
            f"| {name} | {values['unique_signatures']:,} | "
            f"{values['ambiguous_signature_groups']:,} | {values['ambiguous_records']:,} | "
            f"{values['cross_split_signature_groups']:,} | "
            f"{values['signature_limited_exact_match_upper_bound']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The structural signature intentionally removes prose, beliefs, memories, and history. Its ceiling",
            "measures how much those omitted features matter; it is not an identifiability claim. The exact-agent",
            "signature is the relevant check for contradictory supervised examples.",
            "",
            "For the exact agent prompt, the audit also reports how many test prompts already occur in",
            "training. High coverage here means scenario-level splitting does not produce input-level novelty.",
            "",
            "## Test transfer diagnostic",
            "",
            "| Track | Test records seen verbatim in train | Train-majority exact match |",
            "| --- | ---: | ---: |",
            f"| Agent | {report['signatures']['exact_agent_prompt']['test_transfer']['record_seen_in_train_rate']:.2%} | "
            f"{report['signatures']['exact_agent_prompt']['test_transfer']['train_majority_exact_match_on_seen_records']:.2%} |",
            f"| Privileged snapshot | {report['signatures']['exact_privileged_prompt']['test_transfer']['record_seen_in_train_rate']:.2%} | "
            f"{report['signatures']['exact_privileged_prompt']['test_transfer']['train_majority_exact_match_on_seen_records']:.2%} |",
            "",
            "The current privileged snapshot includes flags, rooms, inventory, and scalar state, but not the",
            "scenario's transition rules. Its remaining contradictory labels show that it is not yet a complete",
            "Markov-state representation.",
            "",
            "## Action distribution",
            "",
            "| Action | Records |",
            "| --- | ---: |",
        ]
    )
    for action, count in report["action_type_counts"].items():
        lines.append(f"| {action} | {count:,} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = read_dataset(Path(args.dataset))
    report = audit_records(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report, args.dataset), encoding="utf-8")
    print(
        json.dumps(
            {
                "records": report["records"],
                "state_change_rate": report["target_rates"]["state_change"],
                "group_split_leaks": len(
                    report["split_integrity"]["groups_crossing_splits"]
                ),
                "agent_cross_split_prompt_groups": report["signatures"][
                    "exact_agent_prompt"
                ]["cross_split_signature_groups"],
                "agent_ambiguous_record_rate": report["signatures"]["exact_agent_prompt"][
                    "ambiguous_record_rate"
                ],
                "agent_test_prompt_seen_rate": report["signatures"]["exact_agent_prompt"][
                    "test_transfer"
                ]["record_seen_in_train_rate"],
                "privileged_ambiguous_record_rate": report["signatures"][
                    "exact_privileged_prompt"
                ]["ambiguous_record_rate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
