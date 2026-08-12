#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
npm run protocol:v25:freeze
npm run features:v25
npm run evaluate:v25
npm run audit:v25:result
