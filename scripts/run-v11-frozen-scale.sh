#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run build
npm test
npm run test:python

if [[ ! -f configs/v11-frozen-scale-lock.json ]]; then
  .venv/bin/python python/freeze_v11_frozen_scale.py
fi
for model_key in qwen35_4b qwen35_9b; do
  if [[ ! -f "outputs/v11-frozen-scale/features/${model_key}/metadata.json" ]]; then
    .venv/bin/python python/extract_v11_scale_features_mlx.py --model-key "${model_key}" --progress-every 250
  fi
done
if [[ ! -f outputs/v11-frozen-scale/evaluation/result.json ]]; then
  .venv/bin/python python/evaluate_v11_frozen_scale.py
fi
.venv/bin/python python/summarize_v11.py
