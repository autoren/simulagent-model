#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
source .venv/bin/activate

mlx_lm.lora --config configs/qwen35-9b-v2-agent-smoke.yaml
