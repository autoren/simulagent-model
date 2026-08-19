#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v214_deterministic_candidate_version_space_controls import run_controls


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--fit-public", type=Path, required=True)
    parser.add_argument("--fit-labels", type=Path, required=True)
    parser.add_argument("--evaluation-public", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    predictions = run_controls(
        read_jsonl(args.fit_public),
        read_jsonl(args.fit_labels),
        read_jsonl(args.evaluation_public),
        json.loads(args.semantics.read_text()),
        json.loads(args.config.read_text()),
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(json.dumps(predictions, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
