#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/python python/generate_v22_relational_development.py
.venv/bin/python python/audit_v22_relational.py
.venv/bin/python python/run_v22_oracle_baselines.py
