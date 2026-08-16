#!/usr/bin/env bash
set -euo pipefail

npm run audit:v43:implementation
npm run protocol:v43:implementation:freeze
npm run dataset:v43
npm run audit:v43:corpus
npm run protocol:v43:corpus:seal
npm run develop:v43
npm run audit:v43:result
npm run protocol:v43:outcome:freeze
