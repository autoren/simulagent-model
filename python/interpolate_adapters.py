#!/usr/bin/env python3
"""Linearly interpolate compatible MLX LoRA adapters for calibration experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    left_dir = Path(args.left)
    right_dir = Path(args.right)
    output_dir = Path(args.output)
    left_config = json.loads((left_dir / "adapter_config.json").read_text())
    right_config = json.loads((right_dir / "adapter_config.json").read_text())
    compatibility_fields = ("model", "fine_tune_type", "num_layers", "lora_parameters")
    if any(left_config.get(field) != right_config.get(field) for field in compatibility_fields):
        raise ValueError("Adapter configurations differ and cannot be interpolated")
    left = mx.load(str(left_dir / "adapters.safetensors"))
    right = mx.load(str(right_dir / "adapters.safetensors"))
    if set(left) != set(right):
        raise ValueError("Adapter tensor keys differ and cannot be interpolated")
    merged = {
        key: (1.0 - args.alpha) * left[key] + args.alpha * right[key]
        for key in left
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(left_dir / "adapter_config.json", output_dir / "adapter_config.json")
    mx.save_safetensors(str(output_dir / "adapters.safetensors"), merged)
    print(
        json.dumps(
            {
                "left": str(left_dir),
                "right": str(right_dir),
                "alpha": args.alpha,
                "output": str(output_dir),
                "tensors": len(merged),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
