#!/usr/bin/env bash
set -euo pipefail

npm run audit:v49:implementation
npm run protocol:v49:implementation:freeze
npm run dataset:v49
npm run audit:v49:corpus
npm run protocol:v49:corpus:seal
npm run develop:v49
npm run audit:v49:result
npm run protocol:v49:outcome:freeze
