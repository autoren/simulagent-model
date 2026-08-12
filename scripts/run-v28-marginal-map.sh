#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
npm run protocol:v28:freeze
npm run evaluate:v28
npm run audit:v28:result
