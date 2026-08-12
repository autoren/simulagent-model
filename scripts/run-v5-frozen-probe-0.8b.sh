#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for variant in full no_history; do
  root="outputs/v5-frozen-probe/qwen35-0.8b/${variant}"
  .venv/bin/python python/extract_frozen_qwen_features_mlx.py \
    --model "mlx-community/Qwen3.5-0.8B-4bit" \
    --input-variant "${variant}" \
    --output-dir "${root}/features"
  for seed in 0 1 2; do
    .venv/bin/python python/train_frozen_linear_probe.py \
      --features "${root}/features" \
      --seed "${seed}" \
      --output-dir "${root}/probe/seed-${seed}"
  done
done

.venv/bin/python python/summarize_v5_frozen_probe.py
