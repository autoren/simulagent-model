#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
npm run dataset:v30
npm run audit:v30
npm run protocol:v30:freeze
npm run evaluate:v30
npm run audit:v30:result

if .venv/bin/python -c 'import json; raise SystemExit(0 if json.load(open("outputs/v30-signed-fact-language/evaluation/result.json"))["v28_integration_authorized"] else 1)'
then
  npm run integrate:v30:v28
  npm run audit:v30:integration
fi
