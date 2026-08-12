#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -e outputs/v7r2-untouched/frozen-probe/result.json ]]; then
  echo "Locked V7 r2 untouched-mechanic result already exists; refusing a second evaluation." >&2
else
  echo "The V7 r2 reserve has already been evaluated once; a missing result does not authorize rescoring." >&2
fi
echo "See docs/v7-results.md for the immutable result and LoRA decision." >&2
exit 1
