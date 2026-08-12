#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for seed in 0 1 2; do
  .venv/bin/python python/rescore_v4_fp32_mlx.py \
    --original-result "outputs/v4-binary/seed-${seed}/result.json" \
    --output-dir "outputs/v4-fp32/seed-${seed}"
done

.venv/bin/python python/summarize_v4_fp32.py
