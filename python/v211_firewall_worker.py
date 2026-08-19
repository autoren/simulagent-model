#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v211_deterministic_residual_baselines import select_v210_residual, split_residual


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--calibration-surface", type=Path, required=True)
    parser.add_argument("--calibration-truth", type=Path, required=True)
    parser.add_argument("--evaluation-surface", type=Path, required=True)
    parser.add_argument("--evaluation-truth", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    surfaces = read_jsonl(args.surface)
    truths = read_jsonl(args.truth)
    projections = read_jsonl(args.projection)
    residual = select_v210_residual(surfaces, projections, config)
    split = split_residual(residual, config)
    truth_by_id = {row["record_id"]: row for row in truths}
    calibration_ids = set(split["calibration_record_ids"])
    evaluation_ids = set(split["evaluation_record_ids"])
    prediction_fields = tuple(config["baselines"]["predictionInputFields"])
    calibration_surface = [{key: row[key] for key in prediction_fields} for row in residual if row["record_id"] in calibration_ids]
    evaluation_surface = [{key: row[key] for key in prediction_fields} for row in residual if row["record_id"] in evaluation_ids]
    calibration_truth = [truth_by_id[row["record_id"]] for row in residual if row["record_id"] in calibration_ids]
    evaluation_truth = [truth_by_id[row["record_id"]] for row in residual if row["record_id"] in evaluation_ids]
    write_json(args.split, split)
    write_jsonl(args.calibration_surface, calibration_surface)
    write_jsonl(args.calibration_truth, calibration_truth)
    write_jsonl(args.evaluation_surface, evaluation_surface)
    write_jsonl(args.evaluation_truth, evaluation_truth)


if __name__ == "__main__":
    main()
