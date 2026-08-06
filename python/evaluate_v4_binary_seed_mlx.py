#!/usr/bin/env python3
"""Select one binary LoRA checkpoint/threshold on calibration, then open validation once."""

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

from binary_metrics import evaluate_binary, fit_threshold


SYSTEM_PROMPT = " ".join(
    (
        "Classify whether the candidate action has exactly one transition supported by the observation history.",
        "Use only the supplied observation history and candidate action.",
        "Return exactly one uppercase ASCII letter and nothing else.",
        "Return A when exactly one transition is supported (identifiable).",
        "Return B when multiple transitions are supported (ambiguous).",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-records", default="data/v4/records/calibration.jsonl")
    parser.add_argument("--validation-records", default="data/v4/records/validation.jsonl")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-4bit")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"(\d+)_adapters\.safetensors", path.name)
    if not match:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def logits_from(output: Any) -> Any:
    return output.logits if hasattr(output, "logits") else output


def score_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    token_ids: dict[str, int],
    max_seq_length: int,
    progress_every: int,
    label: str,
) -> tuple[list[dict[str, Any]], int]:
    rows = []
    truncated = 0
    for index, record in enumerate(records, start=1):
        user_input = {**record["agent_input"], "task": "classify_identifiability"}
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
        if len(prompt_tokens) > max_seq_length:
            prompt_tokens = prompt_tokens[-max_seq_length:]
            truncated += 1
        output = model(mx.array([prompt_tokens]))
        logits = logits_from(output)[0, -1]
        selected = logits[mx.array([token_ids["A"], token_ids["B"]])]
        mx.eval(selected)
        identifiable_logit, ambiguous_logit = [float(value) for value in selected.tolist()]
        margin = ambiguous_logit - identifiable_logit
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, margin))))
        rows.append(
            {
                "id": record["id"],
                "gold_ambiguous": not record["target"]["identifiable"],
                "score": margin,
                "candidate_logits": {"A": identifiable_logit, "B": ambiguous_logit},
                "ambiguity_probability_at_zero_threshold": probability,
            }
        )
        if progress_every > 0 and (index % progress_every == 0 or index == len(records)):
            print(f"{label}: scored {index}/{len(records)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    return rows, truncated


def metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return evaluate_binary(
        [row["gold_ambiguous"] for row in rows],
        [row["score"] for row in rows],
        threshold,
    )


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    checkpoints = sorted(adapter_path.glob("*_adapters.safetensors"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No saved checkpoints in {adapter_path}")
    calibration_records = read_jsonl(Path(args.calibration_records))
    validation_records = read_jsonl(Path(args.validation_records))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(args.model, adapter_path=str(adapter_path))
    model.eval()
    encoded = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")}
    if any(len(tokens) != 1 for tokens in encoded.values()):
        raise RuntimeError(f"Binary labels are not single tokens: {encoded}")
    token_ids = {label: tokens[0] for label, tokens in encoded.items()}

    candidates = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        model.load_weights(str(checkpoint), strict=False)
        mx.eval(model.parameters())
        rows, truncated = score_records(
            model,
            tokenizer,
            calibration_records,
            token_ids,
            args.max_seq_length,
            args.progress_every,
            f"calibration step {step}",
        )
        fitted = fit_threshold(
            [row["gold_ambiguous"] for row in rows], [row["score"] for row in rows]
        )
        candidate = {
            "checkpoint_step": step,
            "checkpoint": str(checkpoint),
            "threshold": fitted["threshold"],
            "calibration": fitted,
            "truncated_calibration_prompts": truncated,
        }
        candidates.append(candidate)
        write_jsonl(output_dir / f"calibration-step-{step:07d}.scores.jsonl", rows)
        (output_dir / f"calibration-step-{step:07d}.metrics.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n"
        )

    selected = max(
        candidates,
        key=lambda value: (
            value["calibration"]["balanced_accuracy"],
            value["calibration"]["ambiguity"]["f1"],
            value["calibration"]["roc_auc"],
            -value["checkpoint_step"],
        ),
    )
    selected_checkpoint = Path(selected["checkpoint"])
    model.load_weights(str(selected_checkpoint), strict=False)
    mx.eval(model.parameters())
    validation_rows, truncated = score_records(
        model,
        tokenizer,
        validation_records,
        token_ids,
        args.max_seq_length,
        args.progress_every,
        f"validation selected step {selected['checkpoint_step']}",
    )
    validation = metrics(validation_rows, selected["threshold"])
    result = {
        "model": args.model,
        "adapter_path": str(adapter_path),
        "checkpoint_selection_split": "calibration",
        "threshold_selection_split": "calibration",
        "evaluation_split": "validation",
        "test_records_read": 0,
        "candidate_labels": {"A": "identifiable", "B": "ambiguous"},
        "selected": selected,
        "validation": validation,
        "truncated_validation_prompts": truncated,
    }
    write_jsonl(output_dir / "validation.scores.jsonl", validation_rows)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
