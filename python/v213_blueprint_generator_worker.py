#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v213_fresh_programmatic_concept_population import generate_population


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--public-blueprints", type=Path, required=True)
    parser.add_argument("--sealed-truth", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    semantics = json.loads(args.semantics.read_text())
    blueprints, truth, split, manifest = generate_population(config, semantics)
    write_json(args.manifest, manifest)
    write_jsonl(args.public_blueprints, blueprints)
    write_jsonl(args.sealed_truth, truth)
    write_json(args.split, split)


if __name__ == "__main__":
    main()
