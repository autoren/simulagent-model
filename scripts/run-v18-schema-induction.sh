#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."
.venv/bin/python python/generate_v18_schema_benchmark.py --config configs/dataset.v18.json
.venv/bin/python python/audit_v18_benchmark.py --config configs/dataset.v18.json --dataset data/v18 --output outputs/v18-schema-induction/audit.json
.venv/bin/python python/run_v18_schema_baselines.py --config configs/dataset.v18.json --dataset data/v18 --audit outputs/v18-schema-induction/audit.json --output outputs/v18-schema-induction/baselines.json --markdown docs/v18-results.md
