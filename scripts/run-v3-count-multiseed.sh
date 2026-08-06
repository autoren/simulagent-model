#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run dataset:validate:v3

for seed in 0 1 2; do
  adapter_path="adapters/qwen35-0.8b-v3-outcome-count-seed-${seed}"
  output_path="outputs/v3-calibration/seed-${seed}"
  .venv/bin/mlx_lm.lora \
    --config configs/qwen35-0.8b-v3-outcome-count.yaml \
    --seed "${seed}" \
    --adapter-path "${adapter_path}"
  .venv/bin/python python/score_outcome_count_logits_mlx.py \
    --records data/v3/records/agent/valid.jsonl \
    --adapter-path "${adapter_path}" \
    --output-dir "${output_path}"
done

.venv/bin/python python/analyze_v3_count_logits.py
.venv/bin/python python/select_v3_count_checkpoints.py
