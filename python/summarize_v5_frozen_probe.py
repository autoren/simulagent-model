#!/usr/bin/env python3
"""Aggregate the fixed V5 0.8B frozen-probe variants and seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = ("full", "no_history")
SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/v5-frozen-probe/qwen35-0.8b")
    parser.add_argument("--output", default="outputs/v5-frozen-probe/qwen35-0.8b/summary.json")
    parser.add_argument("--markdown", default="docs/v5-frozen-probe-results.md")
    return parser.parse_args()


def load_results(root: Path) -> dict[str, list[dict[str, Any]]]:
    results = {}
    for variant in VARIANTS:
        values = []
        for seed in SEEDS:
            path = root / variant / "probe" / f"seed-{seed}" / "result.json"
            value = json.loads(path.read_text())
            if value["input_variant"] != variant or value["seed"] != seed:
                raise ValueError(f"Result identity does not match its path: {path}")
            if value["test_records_read"] != 0:
                raise ValueError(f"V5 result opened test records: {path}")
            values.append(value)
        results[variant] = values
    return results


def summarize_variant(values: list[dict[str, Any]]) -> dict[str, Any]:
    balanced = [value["validation"]["balanced_accuracy"] for value in values]
    auc = [value["validation"]["roc_auc"] for value in values]
    return {
        "seeds": [value["seed"] for value in values],
        "selected_features": [value["selected"]["feature"] for value in values],
        "selected_c_values": [value["selected"]["c_value"] for value in values],
        "validation_balanced_accuracy": balanced,
        "validation_roc_auc": auc,
        "mean_validation_balanced_accuracy": float(np.mean(balanced)),
        "validation_balanced_accuracy_range": float(max(balanced) - min(balanced)),
        "mean_validation_roc_auc": float(np.mean(auc)),
        "validation_error_concentration": [
            value["validation_error_concentration"] for value in values
        ],
        "bootstrap_intervals": [
            value["validation_grouped_bootstrap"] for value in values
        ],
    }


def markdown(summary: dict[str, Any], results: dict[str, list[dict[str, Any]]]) -> str:
    rows = []
    for variant in VARIANTS:
        for value in results[variant]:
            interval = value["validation_grouped_bootstrap"][
                "balanced_accuracy_95_percentile_interval"
            ]
            rows.append(
                "| {variant} | {seed} | `{feature}` | {c:g} | {cal:.2%} | {val:.2%} | "
                "{auc:.3f} | {low:.2%}–{high:.2%} |".format(
                    variant=variant,
                    seed=value["seed"],
                    feature=value["selected"]["feature"],
                    c=value["selected"]["c_value"],
                    cal=value["selected"]["calibration"]["balanced_accuracy"],
                    val=value["validation"]["balanced_accuracy"],
                    auc=value["validation"]["roc_auc"],
                    low=interval[0],
                    high=interval[1],
                )
            )
    full = summary["variants"]["full"]
    no_history = summary["variants"]["no_history"]
    full_errors = full["validation_error_concentration"][0]
    no_history_errors = no_history["validation_error_concentration"][0]
    return "\n".join(
        [
            "# V5 frozen 0.8B representation-probe results",
            "",
            "## Decision",
            "",
            "The frozen Qwen3.5-0.8B representation gate passes decisively. A class-balanced "
            "float32 logistic head over the calibration-selected hidden representation reaches "
            f"{full['mean_validation_balanced_accuracy']:.2%} mean validation balanced accuracy "
            f"on full input and {no_history['mean_validation_balanced_accuracy']:.2%} after removing "
            "history and memories. The generative A/B vocabulary interface, not the absence of "
            "linearly accessible signal, was the immediate V4 bottleneck.",
            "",
            "This is not yet evidence of semantic epistemic reasoning. The strong no-history result "
            "is compatible with static-field, action-template, or scenario-family shortcuts. The next "
            "scientific requirement is a newly generated untouched challenge holdout with entity "
            "renamings, paraphrases, evidence-rung minimal pairs, and held-out mechanics. LoRA behind "
            "the discriminative head is eligible, but should not be interpreted without that holdout.",
            "",
            "## Results",
            "",
            "| Input | Seed | Selected feature | C | Calibration balanced | Validation balanced | AUC | Group-bootstrap 95% interval |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "The three seeds exercise the stochastic optimization order of the float32 SAGA fits; "
            "they do not create three independent pretrained Qwen models.",
            "",
            f"Full input makes {full_errors['errors']} errors across "
            f"{full_errors['context_groups_with_errors']} of 19 validation contexts. The no-history "
            f"variant makes {no_history_errors['errors']} errors across "
            f"{no_history_errors['context_groups_with_errors']} contexts, including "
            f"{no_history_errors['completely_misclassified_context_groups']} contexts for which every "
            "record is wrong. This concentration is why record-level performance alone is not enough "
            "to establish shortcut-resistant generalization.",
            "",
            "## Firewall and limitations",
            "",
            "- Features were extracted from 1,037 training, 181 calibration, and 154 validation records.",
            "- Layer, pooling, regularization, and threshold were selected only on calibration.",
            "- Validation contains 19 context groups and is used only for frozen evaluation.",
            "- No prompt was truncated; source hidden states were bfloat16 and probe inputs/weights were float32.",
            "- Validation errors are reported by context group because record-level accuracy can hide group concentration.",
            "- V3 test records read: 0.",
            "- The extraction path reads only `agent_input` plus labels needed for supervised fitting; metadata and hidden-state inputs exclude target outcomes, mechanic labels, empirical support, and scenario IDs.",
            "- Because 0.8B already passes the representational gate, 4B and 9B frozen extraction is not required to answer the capacity question and is deferred.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    results = load_results(Path(args.root))
    summary = {
        "experiment": "v5_frozen_linear_probe_0.8b",
        "model": results["full"][0]["model"],
        "seeds": list(SEEDS),
        "test_records_read": 0,
        "variants": {variant: summarize_variant(values) for variant, values in results.items()},
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown(summary, results))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
