#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
npm run protocol:v26:freeze
npm run evaluate:v26
npm run audit:v26:result
