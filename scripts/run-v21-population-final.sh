#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f configs/v21-multimechanic-execution-lock.json ]]; then
  .venv/bin/python python/freeze_v21_execution.py
fi
if [[ ! -f outputs/v21-final/seed-draw.json ]]; then
  .venv/bin/python python/materialize_v21_final_suite.py
fi
if [[ ! -f outputs/v21-final/pre-extraction-audit.json ]]; then
  .venv/bin/python python/audit_v21_final_suite.py
fi
if [[ ! -f configs/v21-final-dataset-seal.json ]]; then
  .venv/bin/python python/seal_v21_final_dataset.py
fi
if [[ ! -d outputs/v21-final/features ]]; then
  .venv/bin/python python/extract_v21_final_features_mlx.py
fi
if [[ ! -d outputs/v21-final/evaluation ]]; then
  .venv/bin/python python/evaluate_v21_final.py
fi
if [[ ! -f outputs/v21-final/post-result-audit.json ]]; then
  .venv/bin/python python/audit_v21_result.py
fi
.venv/bin/python python/summarize_v21.py
