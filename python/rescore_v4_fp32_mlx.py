#!/usr/bin/env python3
"""Rescore one selected V4 checkpoint before and after the bfloat16 LM head.

This is a post-hoc development diagnostic. It keeps the checkpoint selected by
the original calibration procedure, never reads V3 test, and refits only the
decision threshold on the V4 calibration fold.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

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
    parser.add_argument("--original-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-records", default="data/v4/records/calibration.jsonl")
    parser.add_argument("--validation-records", default="data/v4/records/validation.jsonl")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def pairwise_log_loss(gold_ambiguous: Sequence[bool], scores: Sequence[float]) -> float:
    if len(gold_ambiguous) != len(scores) or not scores:
        raise ValueError("Gold labels and non-empty scores must have equal lengths.")
    losses = [
        softplus(-score) if gold else softplus(score)
        for gold, score in zip(gold_ambiguous, scores)
    ]
    return sum(losses) / len(losses)


def dequantized_label_rows(model: Any, token_ids: dict[str, int]) -> mx.array:
    embedding = model.language_model.model.embed_tokens
    indices = mx.array([token_ids["A"], token_ids["B"]])
    return mx.dequantize(
        embedding.weight[indices],
        embedding.scales[indices],
        embedding.biases[indices],
        group_size=embedding.group_size,
        bits=embedding.bits,
        mode=embedding.mode,
        dtype=mx.float32,
    )


def score_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    token_ids: dict[str, int],
    label_rows_fp32: mx.array,
    max_seq_length: int,
    progress_every: int,
    label: str,
) -> tuple[list[dict[str, Any]], int, dict[str, str]]:
    rows = []
    truncated = 0
    observed_dtypes: dict[str, str] = {}
    label_indices = mx.array([token_ids["A"], token_ids["B"]])
    embedding = model.language_model.model.embed_tokens
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

        hidden = model.language_model.model(mx.array([prompt_tokens]))[0, -1]
        bf16_logits = embedding.as_linear(hidden)[label_indices]
        fp32_logits = hidden.astype(mx.float32) @ label_rows_fp32.T
        mx.eval(hidden, bf16_logits, fp32_logits)
        observed_dtypes = {
            "hidden": str(hidden.dtype),
            "lm_head_logits": str(bf16_logits.dtype),
            "dequantized_label_rows": str(label_rows_fp32.dtype),
            "direct_logits": str(fp32_logits.dtype),
        }
        bf16_values = [float(value) for value in bf16_logits.tolist()]
        fp32_values = [float(value) for value in fp32_logits.tolist()]
        rows.append(
            {
                "id": record["id"],
                "split_group": record["split_group"],
                "gold_ambiguous": not record["target"]["identifiable"],
                "bf16_score": bf16_values[1] - bf16_values[0],
                "fp32_direct_score": fp32_values[1] - fp32_values[0],
                "bf16_candidate_logits": {"A": bf16_values[0], "B": bf16_values[1]},
                "fp32_direct_logits": {"A": fp32_values[0], "B": fp32_values[1]},
            }
        )
        if progress_every > 0 and (index % progress_every == 0 or index == len(records)):
            print(f"{label}: scored {index}/{len(records)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    return rows, truncated, observed_dtypes


def summarize_scores(
    calibration_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], score_key: str
) -> dict[str, Any]:
    calibration_gold = [row["gold_ambiguous"] for row in calibration_rows]
    calibration_scores = [row[score_key] for row in calibration_rows]
    validation_gold = [row["gold_ambiguous"] for row in validation_rows]
    validation_scores = [row[score_key] for row in validation_rows]
    fitted = fit_threshold(calibration_gold, calibration_scores)
    return {
        "score_key": score_key,
        "calibration": {
            **fitted,
            "unique_scores": len(set(calibration_scores)),
            "score_range": [min(calibration_scores), max(calibration_scores)],
            "pairwise_log_loss": pairwise_log_loss(calibration_gold, calibration_scores),
        },
        "validation": {
            **evaluate_binary(validation_gold, validation_scores, fitted["threshold"]),
            "unique_scores": len(set(validation_scores)),
            "score_range": [min(validation_scores), max(validation_scores)],
            "pairwise_log_loss": pairwise_log_loss(validation_gold, validation_scores),
        },
    }


def main() -> None:
    args = parse_args()
    original_path = Path(args.original_result)
    original = read_json(original_path)
    calibration_records = read_jsonl(Path(args.calibration_records))
    validation_records = read_jsonl(Path(args.validation_records))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(original["model"], adapter_path=original["adapter_path"])
    checkpoint = Path(original["selected"]["checkpoint"])
    model.load_weights(str(checkpoint), strict=False)
    mx.eval(model.parameters())
    model.eval()
    encoded = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")}
    if any(len(tokens) != 1 for tokens in encoded.values()):
        raise RuntimeError(f"Binary labels are not single tokens: {encoded}")
    token_ids = {label: tokens[0] for label, tokens in encoded.items()}
    label_rows_fp32 = dequantized_label_rows(model, token_ids)
    mx.eval(label_rows_fp32)

    calibration_rows, truncated_calibration, dtypes = score_records(
        model,
        tokenizer,
        calibration_records,
        token_ids,
        label_rows_fp32,
        args.max_seq_length,
        args.progress_every,
        "calibration",
    )
    validation_rows, truncated_validation, validation_dtypes = score_records(
        model,
        tokenizer,
        validation_records,
        token_ids,
        label_rows_fp32,
        args.max_seq_length,
        args.progress_every,
        "validation",
    )
    if validation_dtypes != dtypes:
        raise RuntimeError(f"Observed dtypes changed across splits: {dtypes} vs {validation_dtypes}")

    result = {
        "diagnostic": "post_hoc_v4_fp32_direct_margin",
        "original_result": str(original_path),
        "model": original["model"],
        "adapter_path": original["adapter_path"],
        "checkpoint": str(checkpoint),
        "checkpoint_step": original["selected"]["checkpoint_step"],
        "checkpoint_selection_split": "original_v4_calibration",
        "threshold_selection_split": "calibration",
        "evaluation_split": "validation",
        "test_records_read": 0,
        "dtypes": dtypes,
        "truncated_calibration_prompts": truncated_calibration,
        "truncated_validation_prompts": truncated_validation,
        "methods": {
            "bf16_vocabulary_logits": summarize_scores(
                calibration_rows, validation_rows, "bf16_score"
            ),
            "fp32_direct_label_projection": summarize_scores(
                calibration_rows, validation_rows, "fp32_direct_score"
            ),
        },
    }
    write_jsonl(output_dir / "calibration.scores.jsonl", calibration_rows)
    write_jsonl(output_dir / "validation.scores.jsonl", validation_rows)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
