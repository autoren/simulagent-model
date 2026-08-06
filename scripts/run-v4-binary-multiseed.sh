#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run dataset:validate:v4
npm run baselines:v4:binary

for seed in 0 1 2; do
  adapter_path="adapters/qwen35-0.8b-v4-binary-seed-${seed}"
  output_path="outputs/v4-binary/seed-${seed}"
  .venv/bin/mlx_lm.lora \
    --config configs/qwen35-0.8b-v4-binary.yaml \
    --seed "${seed}" \
    --adapter-path "${adapter_path}"
  .venv/bin/python python/evaluate_v4_binary_seed_mlx.py \
    --adapter-path "${adapter_path}" \
    --output-dir "${output_path}"
done

.venv/bin/python python/summarize_v4_binary.py
