#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-blueprints", type=Path, required=True)
    parser.add_argument("--public-records", type=Path, required=True)
    args = parser.parse_args()
    rows = sorted(
        (row["public_record"] for row in read_jsonl(args.public_blueprints)),
        key=lambda row: row["case_id"],
    )
    args.public_records.parent.mkdir(parents=True, exist_ok=True)
    args.public_records.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


if __name__ == "__main__":
    main()
