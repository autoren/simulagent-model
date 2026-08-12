#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
npm run protocol:v24:freeze
npm run features:v24
npm run evaluate:v24
npm run audit:v24:result
