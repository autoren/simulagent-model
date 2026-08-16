#!/usr/bin/env bash
set -euo pipefail
npm run audit:v41:implementation
npm run protocol:v41:implementation:freeze
npm run dataset:v41
npm run audit:v41:corpus
npm run protocol:v41:corpus:seal
npm run evaluate:v41
npm run audit:v41:result
npm run protocol:v41:outcome:freeze
