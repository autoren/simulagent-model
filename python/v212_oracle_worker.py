#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v212_open_class_identifiability_oracle import build_predictions


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--public-cases", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    semantics = json.loads(args.semantics.read_text())
    public_cases = read_jsonl(args.public_cases)
    write_jsonl(args.predictions, build_predictions(public_cases, semantics))


if __name__ == "__main__":
    main()
