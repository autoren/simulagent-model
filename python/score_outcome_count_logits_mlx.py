#!/usr/bin/env python3
"""Score constrained outcome counts from next-token logits for every saved checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

from evaluate_outcome_count import evaluate_counts


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
    parser.add_argument("--records", default="data/v3/records/agent/valid.jsonl")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-4bit")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"(\d+)_adapters\.safetensors", path.name)
    if not match:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def logits_from(output: Any) -> Any:
    return output.logits if hasattr(output, "logits") else output


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    checkpoints = sorted(adapter_path.glob("*_adapters.safetensors"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No saved checkpoints in {adapter_path}")
    records = read_jsonl(Path(args.records))
    model, tokenizer = load(args.model, adapter_path=str(adapter_path))
    model.eval()
    digit_tokens = {
        count: tokenizer.encode(str(count), add_special_tokens=False) for count in range(1, 6)
    }
    if any(len(tokens) != 1 for tokens in digit_tokens.values()):
        raise RuntimeError(f"Outcome counts are not single tokens: {digit_tokens}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        model.load_weights(str(checkpoint), strict=False)
        mx.eval(model.parameters())
        predictions = []
        truncated = 0
        for index, record in enumerate(records, start=1):
            user_input = {**record["agent_input"], "task": "count_possible_transitions"}
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(user_input, sort_keys=True, separators=(",", ":")),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_tokens = tokenizer.encode(prompt)
            if len(prompt_tokens) > args.max_seq_length:
                prompt_tokens = prompt_tokens[-args.max_seq_length :]
                truncated += 1
            output = model(mx.array([prompt_tokens]))
            last_logits = logits_from(output)[0, -1]
            token_ids = [digit_tokens[count][0] for count in range(1, 6)]
            selected = last_logits[mx.array(token_ids)]
            mx.eval(selected)
            scores = [float(value) for value in selected.tolist()]
            maximum = max(scores)
            denominator = sum(math.exp(value - maximum) for value in scores)
            probabilities = [math.exp(value - maximum) / denominator for value in scores]
            predicted = max(range(1, 6), key=lambda count: (scores[count - 1], -count))
            predictions.append(
                {
                    "id": record["id"],
                    "prediction": predicted,
                    "candidate_logits": {
                        str(count): scores[count - 1] for count in range(1, 6)
                    },
                    "candidate_probabilities": {
                        str(count): probabilities[count - 1] for count in range(1, 6)
                    },
                    "checkpoint_step": step,
                }
            )
            if args.progress_every > 0 and (
                index % args.progress_every == 0 or index == len(records)
            ):
                print(
                    f"step {step}: scored {index}/{len(records)}",
                    file=sys.stderr,
                    flush=True,
                )
            mx.clear_cache()
        prediction_path = output_dir / f"step-{step:07d}.jsonl"
        prediction_path.write_text(
            "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
        )
        metrics = evaluate_counts(records, predictions)
        metrics.update(
            {
                "checkpoint_step": step,
                "adapter_checkpoint": str(checkpoint),
                "selection_split": "validation",
                "constrained_candidate_digits": True,
                "truncated_prompts": truncated,
            }
        )
        (output_dir / f"step-{step:07d}.metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
