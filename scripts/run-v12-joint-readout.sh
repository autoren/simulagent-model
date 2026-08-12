#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npm run build
npm test
npm run test:python

if [[ ! -f configs/v12-joint-readout-lock.json ]]; then
  .venv/bin/python python/freeze_v12_joint_readout.py
fi
if [[ ! -f outputs/v12-joint-readout/evaluation/result.json ]]; then
  .venv/bin/python python/evaluate_v12_joint_readout.py
fi
.venv/bin/python python/summarize_v12.py
