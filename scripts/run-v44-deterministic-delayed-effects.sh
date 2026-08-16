#!/usr/bin/env bash
set -euo pipefail
npm run audit:v44:implementation
npm run protocol:v44:implementation:freeze
npm run dataset:v44
npm run audit:v44:corpus
npm run protocol:v44:corpus:seal
npm run develop:v44
npm run audit:v44:result
npm run protocol:v44:outcome:freeze
