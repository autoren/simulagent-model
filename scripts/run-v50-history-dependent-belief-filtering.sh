#!/usr/bin/env bash
set -euo pipefail

npm run audit:v50:implementation
npm run protocol:v50:implementation:freeze
npm run dataset:v50
npm run audit:v50:corpus
npm run protocol:v50:corpus:seal
npm run develop:v50
npm run audit:v50:result
npm run protocol:v50:outcome:freeze
