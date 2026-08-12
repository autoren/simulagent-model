#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run build
npm test
npm run test:python

if [[ ! -f configs/v10-grounding-lock.json ]]; then
  .venv/bin/python python/freeze_v10_grounding.py
fi
if [[ ! -f data/v10/manifest.json ]]; then
  npx tsx src/compile-v10.ts --config configs/dataset.v10.json --lock configs/v10-grounding-lock.json
fi
if [[ ! -f outputs/v10-pre-model/shortcut-audit.json ]]; then
  .venv/bin/python python/audit_v10_shortcuts.py
fi
if [[ ! -f configs/v10-frozen-lock.json ]]; then
  .venv/bin/python python/freeze_v10_frozen.py
fi
if [[ ! -f outputs/v10-frozen/features/metadata.json ]]; then
  .venv/bin/python python/extract_v10_features_mlx.py
fi
if [[ ! -f outputs/v10-frozen/evaluation/result.json ]]; then
  .venv/bin/python python/evaluate_v10_frozen.py
fi
.venv/bin/python python/summarize_v10.py
