#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -e outputs/v5-challenge/frozen-probe/result.json ]]; then
  echo "Locked V5 challenge result already exists; refusing a second evaluation." >&2
  exit 1
fi

.venv/bin/python python/freeze_v5_probe.py >/dev/null
npx tsx src/compile-v5-challenge.ts --config configs/dataset.v5.challenge.json >/dev/null
.venv/bin/python python/evaluate_v5_challenge_mlx.py
.venv/bin/python python/summarize_v5_challenge.py
