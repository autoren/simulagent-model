#!/usr/bin/env python3
"""Analyze validation ranking hidden beneath constrained digit argmax decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


Record = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/v3/records/agent/valid.jsonl")
    parser.add_argument("--root", default="outputs/v3-calibration")
    parser.add_argument("--output", default="outputs/v3-calibration/logit-diagnostics.json")
    parser.add_argument("--markdown", default="docs/v3-logit-diagnostics.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def analyze(gold: dict[str, bool], rows: list[Record]) -> dict[str, Any]:
    scored = []
    for row in rows:
        logits = row["candidate_logits"]
        ambiguity_score = max(logits[str(count)] for count in range(2, 6)) - logits["1"]
        scored.append((ambiguity_score, gold[row["id"]]))
    positive = [score for score, ambiguous in scored if ambiguous]
    negative = [score for score, ambiguous in scored if not ambiguous]
    auc = sum(
        (left > right) + 0.5 * (left == right) for left in positive for right in negative
    ) / (len(positive) * len(negative))
    candidates = sorted({score for score, _ in scored})
    best = {"balanced_identifiability_accuracy": 0.0}
    for threshold in candidates:
        true_positive = sum(score >= threshold and ambiguous for score, ambiguous in scored)
        false_negative = sum(score < threshold and ambiguous for score, ambiguous in scored)
        true_negative = sum(score < threshold and not ambiguous for score, ambiguous in scored)
        false_positive = sum(score >= threshold and not ambiguous for score, ambiguous in scored)
        ambiguity_recall = true_positive / (true_positive + false_negative)
        identifiable_recall = true_negative / (true_negative + false_positive)
        balanced = (ambiguity_recall + identifiable_recall) / 2
        if balanced > best["balanced_identifiability_accuracy"]:
            best = {
                "balanced_identifiability_accuracy": balanced,
                "threshold": threshold,
                "ambiguity_recall": ambiguity_recall,
                "identifiable_recall": identifiable_recall,
            }
    return {
        "roc_auc": auc,
        "posthoc_best_same_validation_threshold": best,
        "score_min": min(score for score, _ in scored),
        "score_max": max(score for score, _ in scored),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V3 count-logit diagnostics",
        "",
        "These are secondary diagnostics on the same validation data used to inspect checkpoints.",
        "Post-hoc thresholds are optimistic and are not eligible for checkpoint selection or the",
        "calibration gate. ROC AUC measures ranking independently of the default digit argmax.",
        "",
        "| Seed | Step | ROC AUC | Post-hoc best balanced ID |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in report["checkpoints"]:
        lines.append(
            f"| {row['seed']} | {row['checkpoint_step']} | {row['roc_auc']:.3f} | "
            f"{row['posthoc_best_same_validation_threshold']['balanced_identifiability_accuracy']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    gold = {
        record["id"]: not record["target"]["identifiable"]
        for record in read_jsonl(Path(args.gold))
    }
    checkpoints = []
    root = Path(args.root)
    for seed in args.seeds:
        for path in sorted((root / f"seed-{seed}").glob("step-*.jsonl")):
            rows = read_jsonl(path)
            step = rows[0]["checkpoint_step"]
            checkpoints.append({"seed": seed, "checkpoint_step": step, **analyze(gold, rows)})
    report = {
        "split": "validation",
        "selection_eligible": False,
        "warning": "Post-hoc thresholds use the evaluation labels and are diagnostic only.",
        "checkpoints": checkpoints,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
