#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
npm run protocol:v27:freeze
npm run score:v27:edges
npm run evaluate:v27
npm run audit:v27:result
