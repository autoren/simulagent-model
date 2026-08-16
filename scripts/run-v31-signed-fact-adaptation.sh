#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
npm run dataset:v31
npm run audit:v31
npm run protocol:v31:freeze
npm run features:v31
npm run train:v31:frozen
npm run train:v31:lora
npm run protocol:v31:trained:freeze
npm run evaluate:v31
npm run audit:v31:result

if .venv/bin/python -c 'import json; raise SystemExit(0 if json.load(open("outputs/v31-signed-fact-adaptation/sealed-evaluation/result.json"))["v28_integration_authorized"] else 1)'
then
  npm run integrate:v31:v28
fi
