#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run build
npm test
npm run test:python

if [[ ! -f configs/v13-token-local-lock.json ]]; then
  .venv/bin/python python/freeze_v13_token_local.py
fi
if [[ ! -f outputs/v13-token-local/features/metadata.json ]]; then
  .venv/bin/python python/extract_v13_token_local_mlx.py --progress-every 100
fi
if [[ ! -f outputs/v13-token-local/evaluation/result.json ]]; then
  .venv/bin/python python/evaluate_v13_token_local.py
fi
if [[ ! -f outputs/v13-token-local/operator-support-audit.json ]]; then
  .venv/bin/python python/audit_v13_operator_support.py
fi
.venv/bin/python python/summarize_v13.py
