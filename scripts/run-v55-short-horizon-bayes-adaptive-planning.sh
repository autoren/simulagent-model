#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=python .venv/bin/python python/generate_v55_planning.py
PYTHONPATH=python .venv/bin/python python/audit_v55_populations.py
PYTHONPATH=python .venv/bin/python python/seal_v55_population.py
PYTHONPATH=python .venv/bin/python python/evaluate_v55_planning.py
PYTHONPATH=python .venv/bin/python python/audit_and_summarize_v55.py
PYTHONPATH=python .venv/bin/python python/freeze_v55_outcome.py
