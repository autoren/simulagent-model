#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v211_deterministic_residual_baselines import fit_baselines, predict_evaluation


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-surface", type=Path, required=True)
    parser.add_argument("--calibration-truth", type=Path, required=True)
    parser.add_argument("--evaluation-surface", type=Path, required=True)
    parser.add_argument("--learned", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    calibration_surface = read_jsonl(args.calibration_surface)
    calibration_truth = read_jsonl(args.calibration_truth)
    evaluation_surface = read_jsonl(args.evaluation_surface)
    allowed = set(config["baselines"]["predictionInputFields"])
    if any(set(row) != allowed for row in evaluation_surface):
        raise RuntimeError("V211 prediction worker received forbidden evaluation fields")
    learned = fit_baselines(calibration_surface, calibration_truth, config)
    predictions = predict_evaluation(evaluation_surface, learned)
    args.learned.parent.mkdir(parents=True, exist_ok=True)
    args.learned.write_text(json.dumps(learned, indent=2, sort_keys=True) + "\n")
    args.predictions.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions))


if __name__ == "__main__":
    main()
