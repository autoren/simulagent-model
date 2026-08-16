#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=python

.venv/bin/python python/generate_v53_smc2.py
.venv/bin/python python/audit_v53_populations.py
.venv/bin/python python/seal_v53_populations.py
.venv/bin/python python/evaluate_v53_smc2.py
.venv/bin/python python/audit_and_summarize_v53.py
.venv/bin/python python/freeze_v53_outcome.py
