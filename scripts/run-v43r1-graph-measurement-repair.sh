#!/usr/bin/env bash
set -euo pipefail
npm run audit:v43r1:implementation
npm run protocol:v43r1:implementation:freeze
npm run rescore:v43r1
npm run audit:v43r1:result
npm run protocol:v43r1:outcome:freeze
