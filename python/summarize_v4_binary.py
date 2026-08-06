#!/usr/bin/env python3
"""Aggregate three frozen V4 seed evaluations and apply engineering/scientific gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from binary_metrics import nonconstant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/v4-binary")
    parser.add_argument("--baselines", default="outputs/baselines/v4-binary/summary.json")
    parser.add_argument("--output", default="outputs/v4-binary/summary.json")
    parser.add_argument("--markdown", default="docs/v4-binary-results.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def apply_gate(results: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    balanced = [result["validation"]["balanced_accuracy"] for result in results]
    engineering_checks = {
        "all_seeds_above_50pct": all(value > 0.50 for value in balanced),
        "at_least_two_seeds_at_or_above_60pct": sum(value >= 0.60 for value in balanced) >= 2,
        "mean_at_or_above_60pct": sum(balanced) / len(balanced) >= 0.60,
        "seed_range_at_or_below_10pct": max(balanced) - min(balanced) <= 0.10,
        "all_validation_predictions_nonconstant": all(
            nonconstant(result["validation"]) for result in results
        ),
    }
    baseline_balanced = baseline["full"]["validation"]["balanced_accuracy"]
    scientific_checks = {
        "mean_matches_primary_full_input_token_baseline": sum(balanced) / len(balanced)
        >= baseline_balanced,
        "all_seeds_match_primary_full_input_token_baseline": all(
            value >= baseline_balanced for value in balanced
        ),
    }
    return {
        "engineering_passed": all(engineering_checks.values()),
        "engineering_checks": engineering_checks,
        "scientific_passed": all(scientific_checks.values()),
        "scientific_checks": scientific_checks,
        "mean_validation_balanced_accuracy": sum(balanced) / len(balanced),
        "validation_balanced_accuracy_range": max(balanced) - min(balanced),
        "token_baseline_reference": {
            "variant": "full",
            "validation_balanced_accuracy": baseline_balanced,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V4 binary identifiability results",
        "",
        "Each seed selected its LoRA checkpoint and A/B logit threshold only on the",
        "context-disjoint calibration fold. The selected pair was then evaluated once on V3",
        "validation. V3 test remained closed.",
        "",
        "| Seed | Step | Threshold | Calibration balanced | Validation balanced | Validation F1 | Validation AUC | Predictions (I/A) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in summary["results"]:
        selected = result["selected"]
        validation = result["validation"]
        distribution = validation["prediction_distribution"]
        lines.append(
            f"| {result['seed']} | {selected['checkpoint_step']} | {selected['threshold']:.3f} | "
            f"{selected['calibration']['balanced_accuracy']:.2%} | "
            f"{validation['balanced_accuracy']:.2%} | {validation['ambiguity']['f1']:.2%} | "
            f"{validation['roc_auc']:.3f} | {distribution['identifiable']}/{distribution['ambiguous']} |"
        )
    gate = summary["gate"]
    lines.extend(
        [
            "",
            f"**Engineering stability gate: {'PASS' if gate['engineering_passed'] else 'FAIL'}**",
            "",
            f"**Scientific token-baseline gate: {'PASS' if gate['scientific_passed'] else 'FAIL'}**",
            "",
            f"Mean validation balanced accuracy: {gate['mean_validation_balanced_accuracy']:.2%}.",
            f"Primary full-input token baseline: "
            f"{gate['token_baseline_reference']['validation_balanced_accuracy']:.2%}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = []
    for seed in args.seeds:
        result = json.loads((Path(args.root) / f"seed-{seed}" / "result.json").read_text())
        result["seed"] = seed
        results.append(result)
    baseline = json.loads(Path(args.baselines).read_text())
    summary = {"results": results, "gate": apply_gate(results, baseline)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
