#!/usr/bin/env python3
"""Summarize the post-hoc V4 float32 direct-margin diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


METHODS = ("bf16_vocabulary_logits", "fp32_direct_label_projection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/v4-fp32")
    parser.add_argument("--output", default="outputs/v4-fp32/summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    results = [json.loads(path.read_text()) for path in sorted(root.glob("seed-*/result.json"))]
    if not results:
        raise FileNotFoundError(f"No seed results below {root}")
    seeds = []
    for result in results:
        seed = int(Path(result["adapter_path"]).name.rsplit("-", 1)[-1])
        seeds.append(
            {
                "seed": seed,
                "checkpoint_step": result["checkpoint_step"],
                "methods": result["methods"],
            }
        )
    aggregate: dict[str, Any] = {}
    for method in METHODS:
        aggregate[method] = {
            "mean_validation_balanced_accuracy": mean(
                seed["methods"][method]["validation"]["balanced_accuracy"] for seed in seeds
            ),
            "mean_validation_roc_auc": mean(
                seed["methods"][method]["validation"]["roc_auc"] for seed in seeds
            ),
            "mean_validation_pairwise_log_loss": mean(
                seed["methods"][method]["validation"]["pairwise_log_loss"] for seed in seeds
            ),
            "validation_unique_scores": [
                seed["methods"][method]["validation"]["unique_scores"] for seed in seeds
            ],
        }
    summary = {
        "diagnostic": "post_hoc_v4_fp32_direct_margin",
        "seed_count": len(seeds),
        "test_records_read": 0,
        "seeds": seeds,
        "aggregate": aggregate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
