#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run build
npm test
npm run test:python

if [[ ! -f configs/v17-final-construction-lock.json ]]; then
  if [[ -e data/v17-final ]]; then
    echo "V17 data exists without a construction lock; refusing to continue." >&2
    exit 1
  fi
  .venv/bin/python python/freeze_v17_final_mechanic.py
fi
if [[ ! -f data/v17-final/manifest.json ]]; then
  npm run dataset:v17:final
fi
if [[ ! -f configs/v17-final-evaluation-lock.json ]]; then
  .venv/bin/python python/seal_v17_final_dataset.py
fi
if [[ ! -f outputs/v17-final/features/metadata.json ]]; then
  .venv/bin/python python/extract_v17_final_features_mlx.py --progress-every 100
fi
if [[ ! -f outputs/v17-final/evaluation/result.json ]]; then
  .venv/bin/python python/evaluate_v17_final_mechanic.py
fi
.venv/bin/python python/summarize_v17.py
if [[ ! -f outputs/v17-final/post-result-audit.json ]]; then
  .venv/bin/python python/audit_v17_final_result.py
fi
