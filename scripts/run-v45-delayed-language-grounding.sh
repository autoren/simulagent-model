#!/usr/bin/env bash
set -euo pipefail

npm run audit:v45:implementation
npm run protocol:v45:implementation:freeze
npm run dataset:v45
npm run audit:v45:corpus
npm run protocol:v45:corpus:seal
npm run develop:v45
npm run audit:v45:result
npm run protocol:v45:outcome:freeze
