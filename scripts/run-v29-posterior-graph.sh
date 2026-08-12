#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
npm run protocol:v29:freeze
npm run evaluate:v29
npm run audit:v29:result
