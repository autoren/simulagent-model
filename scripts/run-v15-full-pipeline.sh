#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run build
npm test
npm run test:python

if [[ ! -f configs/v15-full-pipeline-lock.json ]]; then
  .venv/bin/python python/freeze_v15_full_pipeline.py
fi
if [[ ! -f outputs/v15-full-pipeline/features/metadata.json ]]; then
  .venv/bin/python python/extract_v15_full_features_mlx.py --progress-every 100
fi
if [[ ! -f outputs/v15-full-pipeline/evaluation/result.json ]]; then
  .venv/bin/python python/evaluate_v15_full_pipeline.py
fi
if [[ ! -f outputs/v15-full-pipeline/group-scope-audit.json ]]; then
  .venv/bin/python python/audit_v15_group_scope.py
fi
.venv/bin/python python/summarize_v15.py
