#!/usr/bin/env python3
"""Run an MLX model or adapter over a Simulagent oracle record set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlx_lm import generate, load


SYSTEM_PROMPT = " ".join(
    (
        "Predict the exact next transition in a deterministic simulator.",
        "Use only the supplied state or observation history and candidate action.",
        "Do not choose a different action. Do not add explanation.",
        "Return one JSON object with exactly these fields: blocked_actions_added, "
        "blocked_actions_removed, environment_changed, flags_changed, "
        "hidden_actions_concealed, hidden_actions_revealed, inventory_added, "
        "inventory_removed, next_location, reachable_room_delta, success, "
        "visible_actions_added, visible_actions_removed.",
        "Use empty arrays and an empty flags_changed object when nothing changes.",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="data/pilot/records/test.jsonl")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-4bit")
    parser.add_argument("--adapter-path")
    parser.add_argument("--track", choices=("agent", "privileged"), default="agent")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    load_kwargs = {"adapter_path": args.adapter_path} if args.adapter_path else {}
    model, tokenizer = load(args.model, **load_kwargs)
    records = read_jsonl(Path(args.records))
    if args.limit is not None:
        records = records[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            input_key = "agent_input" if args.track == "agent" else "privileged_input"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(record[input_key], sort_keys=True, separators=(",", ":")),
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
            handle.write(json.dumps({"id": record["id"], "prediction": response}) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
