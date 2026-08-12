#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -e outputs/v6-mechanic-holdout/frozen-probe/result.json ]]; then
  echo "Locked V6 mechanic-holdout result already exists; refusing a second evaluation." >&2
  exit 1
fi

if [[ ! -e configs/v6-protocol-lock.json ]]; then
  npm run dataset:v6 >/dev/null
fi
npm run protocol:v6:freeze >/dev/null
npm run probe:v6:extract
npm run probe:v6:train
npm run probe:v6:freeze >/dev/null
npm run holdout:v6:evaluate
npm run holdout:v6:summarize
