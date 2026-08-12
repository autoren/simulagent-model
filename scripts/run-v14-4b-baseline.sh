#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run build
npm test
npm run test:python

if [[ ! -f configs/v14-4b-baseline-lock.json ]]; then
  .venv/bin/python python/freeze_v14_4b_baseline.py
fi
if [[ ! -f outputs/v14-4b-baseline/features/metadata.json ]]; then
  .venv/bin/python python/extract_v14_4b_token_mean_mlx.py --progress-every 100
fi
if [[ ! -f outputs/v14-4b-baseline/evaluation/result.json ]]; then
  .venv/bin/python python/evaluate_v14_4b_baseline.py
fi
.venv/bin/python python/summarize_v14.py
