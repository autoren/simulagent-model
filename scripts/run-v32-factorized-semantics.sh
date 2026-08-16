#!/usr/bin/env bash
set -euo pipefail

npm run dataset:v32
npm run audit:v32
npm run protocol:v32:freeze
npm run features:v32
npm run train:v32
npm run protocol:v32:trained:freeze
npm run evaluate:v32
npm run audit:v32:result

if .venv/bin/python -c 'import json; raise SystemExit(0 if json.load(open("outputs/v32-factorized-semantics/post-result-audit.json"))["v28_integration_authorized"] else 1)'; then
  npm run integrate:v32:v28
else
  echo "V32 absolute language gates did not authorize V28; stopping as preregistered."
fi
