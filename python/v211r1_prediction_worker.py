#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v211r1_compositional_baseline_name_repair import predict_evaluation


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned", type=Path, required=True)
    parser.add_argument("--evaluation-surface", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    learned = json.loads(args.learned.read_text())
    surfaces = read_jsonl(args.evaluation_surface)
    if any(set(row) != {"record_id", "context_id", "utterance"} for row in surfaces):
        raise RuntimeError("V211r1 worker received forbidden fields")
    predictions = predict_evaluation(surfaces, learned)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions))


if __name__ == "__main__":
    main()
