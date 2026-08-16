#!/usr/bin/env python3
"""Apply only the hash-locked V31 MLX value-and-grad call-binding repair."""

from __future__ import annotations

import argparse
import json
import sys

import mlx.nn as nn

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("frozen", "lora"), required=True)
    parser.add_argument("--amendment", default="configs/v31-api-binding-repair-lock.json")
    args = parser.parse_args()
    amendment_path = (PROJECT_ROOT / args.amendment).resolve()
    amendment = json.loads(amendment_path.read_text())
    protocol_path = PROJECT_ROOT / amendment["protocol_lock"]
    if file_sha256(protocol_path) != amendment["protocol_lock_sha256"]:
        raise RuntimeError("V31 source protocol changed after API-repair amendment")
    for path, expected in amendment["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V31 API-repair implementation changed: {path}")
    original_value_and_grad = nn.value_and_grad

    def repaired_value_and_grad(model, function):
        return original_value_and_grad(
            model, lambda *values, **kwargs: function(model, *values, **kwargs)
        )

    nn.value_and_grad = repaired_value_and_grad
    if args.system == "frozen":
        import train_v31_frozen_readout as training
        sys.argv = [training.__file__, "--lock", str(protocol_path)]
    else:
        import train_v31_lora_readout_mlx as training
        sys.argv = [training.__file__, "--lock", str(protocol_path)]
    training.main()


if __name__ == "__main__":
    main()
