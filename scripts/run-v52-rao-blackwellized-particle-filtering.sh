#!/usr/bin/env bash
set -euo pipefail

npm run audit:v52:implementation
npm run protocol:v52:implementation:freeze
npm run dataset:v52
npm run audit:v52:populations
npm run protocol:v52:populations:seal
npm run evaluate:v52
npm run audit:v52:result
npm run protocol:v52:outcome:freeze
