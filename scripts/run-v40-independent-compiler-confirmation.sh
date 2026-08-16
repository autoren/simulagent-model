#!/usr/bin/env bash
set -euo pipefail

npm run audit:v40:implementation
npm run protocol:v40:implementation:freeze
npm run dataset:v40
npm run audit:v40:corpus
npm run protocol:v40:corpus:seal
npm run evaluate:v40
npm run audit:v40:result
npm run protocol:v40:outcome:freeze
