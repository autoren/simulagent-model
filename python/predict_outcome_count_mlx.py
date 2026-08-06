#!/usr/bin/env python3
"""Generate compact outcome-count predictions for the v2 calibration task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mlx_lm import generate, load


SYSTEM_PROMPT = " ".join(
    (
        "Count the distinct transitions supported by observationally equivalent deterministic worlds.",
        "Use only the supplied observation history and candidate action.",
        "Do not predict transition contents and do not add explanation.",
        "Return exactly one ASCII digit from 1 through 5 and nothing else.",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="data/v2/records/agent/test.jsonl")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-4bit")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    args = parse_args()
    kwargs = {"adapter_path": args.adapter_path} if args.adapter_path else {}
    model, tokenizer = load(args.model, **kwargs)
    records = read_jsonl(Path(args.records))
    if args.limit is not None:
        records = records[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            user_input = {**record["agent_input"], "task": "count_possible_transitions"}
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_input, sort_keys=True, separators=(",", ":")),
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prediction = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
            handle.write(json.dumps({"id": record["id"], "prediction": prediction}) + "\n")
            handle.flush()
            if args.progress_every > 0 and (
                index % args.progress_every == 0 or index == len(records)
            ):
                print(f"Generated {index}/{len(records)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
