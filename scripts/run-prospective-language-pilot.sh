#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if ! .venv/bin/python -c "import streamlit" >/dev/null 2>&1; then
  echo "Streamlit is not installed in .venv. Run:"
  echo ".venv/bin/python -m pip install -r requirements-pilot-ui.txt"
  exit 1
fi

exec .venv/bin/streamlit run python/prospective_language_pilot_app.py "$@"
