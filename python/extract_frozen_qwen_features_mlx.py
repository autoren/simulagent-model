#!/usr/bin/env python3
"""Extract context-disjoint frozen Qwen hidden representations for V5 probes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask


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
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-4bit")
    parser.add_argument("--dataset", default="data/v4/records")
    parser.add_argument("--output-dir", default="outputs/v5-frozen-probe/qwen35-0.8b/full/features")
    parser.add_argument("--input-variant", choices=("full", "no_history"), default="full")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def input_variant(record: dict[str, Any], variant: str) -> dict[str, Any]:
    value = copy.deepcopy(record["agent_input"])
    if variant == "no_history":
        value.pop("recent_history", None)
        value["observation"].pop("memories", None)
    value["task"] = "classify_identifiability"
    return value


def quartile_layers(layer_count: int) -> list[int]:
    layers = {max(1, round(layer_count * fraction)) for fraction in (0.25, 0.5, 0.75, 1.0)}
    return sorted(layers)


def forward_captures(model: Any, inputs: mx.array, capture_layers: set[int]) -> dict[int, mx.array]:
    text_model = model.language_model.model
    hidden = text_model.embed_tokens(inputs)
    cache = [None] * len(text_model.layers)
    full_attention_mask = create_attention_mask(hidden, cache[text_model.fa_idx])
    state_space_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx])
    captures = {}
    for layer_number, (layer, layer_cache) in enumerate(zip(text_model.layers, cache), start=1):
        mask = state_space_mask if layer.is_linear else full_attention_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)
        if layer_number in capture_layers:
            captures[layer_number] = text_model.norm(hidden) if layer_number == len(cache) else hidden
    return captures


def extract_split(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    layers: list[int],
    variant: str,
    max_seq_length: int,
    progress_every: int,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    values: dict[str, list[np.ndarray]] = {
        f"layer_{layer:02d}_{pooling}": [] for layer in layers for pooling in ("last", "mean")
    }
    ids = []
    groups = []
    labels = []
    lengths = []
    truncated = 0
    observed_dtypes: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        user_input = input_variant(record, variant)
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
        captures = forward_captures(model, mx.array([prompt_tokens]), set(layers))
        mx.eval(*captures.values())
        for layer in layers:
            hidden = captures[layer][0]
            observed_dtypes[f"layer_{layer:02d}"] = str(hidden.dtype)
            values[f"layer_{layer:02d}_last"].append(
                np.asarray(hidden[-1].astype(mx.float32), dtype=np.float32)
            )
            values[f"layer_{layer:02d}_mean"].append(
                np.asarray(mx.mean(hidden.astype(mx.float32), axis=0), dtype=np.float32)
            )
        ids.append(record["id"])
        groups.append(record["split_group"])
        labels.append(not record["target"]["identifiable"])
        lengths.append(len(prompt_tokens))
        if progress_every > 0 and (index % progress_every == 0 or index == len(records)):
            print(f"{split}: extracted {index}/{len(records)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    arrays: dict[str, np.ndarray] = {
        "ids": np.asarray(ids),
        "groups": np.asarray(groups),
        "gold_ambiguous": np.asarray(labels, dtype=np.uint8),
        "prompt_lengths": np.asarray(lengths, dtype=np.int32),
    }
    arrays.update({key: np.stack(rows) for key, rows in values.items()})
    return arrays, {
        "records": len(records),
        "context_groups": len(set(groups)),
        "truncated_prompts": truncated,
        "minimum_prompt_tokens": min(lengths),
        "maximum_prompt_tokens": max(lengths),
        "dtypes": observed_dtypes,
    }


def main() -> None:
    args = parse_args()
    model, tokenizer = load(args.model)
    model.eval()
    layer_count = len(model.language_model.model.layers)
    layers = quartile_layers(layer_count)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "frozen": True,
        "adapter_path": None,
        "dataset": args.dataset,
        "input_variant": args.input_variant,
        "layer_count": layer_count,
        "captured_layers": layers,
        "pooling": ["last", "mean"],
        "test_records_read": 0,
        "splits": {},
    }
    for split in ("train", "calibration", "validation"):
        records = read_jsonl(Path(args.dataset) / f"{split}.jsonl")
        arrays, split_metadata = extract_split(
            model,
            tokenizer,
            records,
            layers,
            args.input_variant,
            args.max_seq_length,
            args.progress_every,
            split,
        )
        np.savez_compressed(output_dir / f"{split}.npz", **arrays)
        metadata["splits"][split] = split_metadata
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
