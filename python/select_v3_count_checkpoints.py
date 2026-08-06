#!/usr/bin/env python3
"""Select V3 count checkpoints on validation only and apply the preregistered gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


Metric = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/v3-calibration")
    parser.add_argument("--output", default="outputs/v3-calibration/selection.json")
    parser.add_argument("--markdown", default="docs/v3-calibration-results.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def select(metrics: list[Metric]) -> Metric:
    if not metrics:
        raise ValueError("No checkpoint metrics supplied.")
    return max(
        metrics,
        key=lambda value: (
            value["balanced_identifiability_accuracy"],
            value["ambiguity_detection"]["f1"],
            value["macro_accuracy_by_observed_gold_count"],
            -value["checkpoint_step"],
        ),
    )


def apply_gate(selected: list[Metric]) -> dict[str, Any]:
    balanced = [value["balanced_identifiability_accuracy"] for value in selected]
    nonconstant = [
        len(value["predicted_count_distribution"]) >= 2
        and value["ambiguity_detection"]["recall"] > 0
        and value["ambiguity_detection"]["tn"] > 0
        for value in selected
    ]
    checks = {
        "all_seeds_above_non_discriminating_boundary": all(value > 0.50 for value in balanced),
        "at_least_two_seeds_at_or_above_55pct": sum(value >= 0.55 for value in balanced) >= 2,
        "mean_balanced_identifiability_at_or_above_55pct": sum(balanced) / len(balanced) >= 0.55,
        "selected_seed_range_at_or_below_10pct": max(balanced) - min(balanced) <= 0.10,
        "all_selected_checkpoints_nonconstant": all(nonconstant),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mean_balanced_identifiability_accuracy": sum(balanced) / len(balanced),
        "balanced_identifiability_range": max(balanced) - min(balanced),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V3 outcome-count calibration results",
        "",
        "Checkpoint selection used the complete fixed validation split and constrained next-token",
        "scores over digits 1 through 5. No test metrics were used.",
        "",
        "| Seed | Selected step | Exact count | Balanced ID | Ambiguity F1 | Prediction counts |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["selected"]:
        lines.append(
            f"| {row['seed']} | {row['checkpoint_step']} | {row['accuracy']:.2%} | "
            f"{row['balanced_identifiability_accuracy']:.2%} | "
            f"{row['ambiguity_detection']['f1']:.2%} | "
            f"{json.dumps(row['predicted_count_distribution'], sort_keys=True)} |"
        )
    lines.extend(
        [
            "",
            f"**Calibration gate: {'PASS' if report['gate']['passed'] else 'FAIL'}**",
            "",
            "The gate was fixed before training: every seed must beat 50% balanced ID, at least",
            "two of three and the mean must reach 55%, the seed range must be at most ten points,",
            "and every selected checkpoint must predict both identifiable and ambiguous cases.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    selected = []
    for seed in args.seeds:
        metrics = [json.loads(path.read_text()) for path in sorted((root / f"seed-{seed}").glob("*.metrics.json"))]
        best = {**select(metrics), "seed": seed}
        selected.append(best)
    report = {"selection_split": "validation", "selected": selected, "gate": apply_gate(selected)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
