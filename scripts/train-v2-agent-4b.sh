#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/mlx_lm.lora ]]; then
  echo "MLX-LM is not installed. Run ./scripts/bootstrap.sh first." >&2
  exit 1
fi

npm run dataset:validate:v2
.venv/bin/mlx_lm.lora --config configs/qwen35-4b-v2-agent.yaml
