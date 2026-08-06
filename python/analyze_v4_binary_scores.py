#!/usr/bin/env python3
"""Report the score resolution of selected V4 binary checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/v4-binary")
    parser.add_argument("--output", default="outputs/v4-binary/score-diagnostics.json")
    parser.add_argument("--markdown", default="docs/v4-score-diagnostics.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in rows]
    unique = sorted(set(scores))
    spacings = [right - left for left, right in zip(unique, unique[1:])]
    return {
        "examples": len(rows),
        "unique_score_count": len(unique),
        "unique_scores": unique,
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "minimum_nonzero_spacing": min(spacings) if spacings else None,
        "score_distribution": dict(sorted(Counter(scores).items())),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4 binary score-resolution diagnostics",
        "",
        "The selected A/B candidate logits were emitted at coarse numeric precision. Converting",
        "them to Python floats after the forward pass does not recover ranking information that",
        "was already quantized in the language-model output head.",
        "",
        "| Seed | Step | Calibration unique margins | Validation unique margins | Validation range |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report["seeds"]:
        validation = result["validation"]
        lines.append(
            f"| {result['seed']} | {result['checkpoint_step']} | "
            f"{result['calibration']['unique_score_count']} | "
            f"{validation['unique_score_count']} | "
            f"{validation['minimum_score']:.4f} to {validation['maximum_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Across selected seeds, each example was placed into only two or three margin bins.",
            "This explains why fitted thresholds could change class proportions but could not",
            "extract a fine-grained ordering. The next experiment should use a dedicated float32",
            "classification head over hidden representations rather than subtracting two",
            "low-precision vocabulary logits.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    seeds = []
    for seed in args.seeds:
        seed_root = root / f"seed-{seed}"
        result = json.loads((seed_root / "result.json").read_text())
        step = result["selected"]["checkpoint_step"]
        calibration = read_jsonl(seed_root / f"calibration-step-{step:07d}.scores.jsonl")
        validation = read_jsonl(seed_root / "validation.scores.jsonl")
        seeds.append(
            {
                "seed": seed,
                "checkpoint_step": step,
                "calibration": describe(calibration),
                "validation": describe(validation),
            }
        )
    report = {"score": "logit(B) - logit(A)", "seeds": seeds}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
