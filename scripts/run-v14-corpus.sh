#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run build
npm test
npm run test:python

if [[ ! -f configs/v14-grounding-lock.json ]]; then
  .venv/bin/python python/freeze_v14_grounding.py
fi
if [[ ! -f data/v14/manifest.json ]]; then
  npm run dataset:v14
fi
if [[ ! -f outputs/v14-pre-model/shortcut-audit.json ]]; then
  npm run audit:v14
fi
