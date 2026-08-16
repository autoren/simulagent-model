#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=python

.venv/bin/python python/generate_v54_eig.py
.venv/bin/python python/audit_v54_populations.py
.venv/bin/python python/seal_v54_populations.py
.venv/bin/python python/evaluate_v54_eig.py
.venv/bin/python python/audit_and_summarize_v54.py
.venv/bin/python python/freeze_v54_outcome.py
