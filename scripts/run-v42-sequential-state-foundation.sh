#!/usr/bin/env bash
set -euo pipefail

npm run audit:v42:implementation
npm run protocol:v42:implementation:freeze
npm run dataset:v42
npm run audit:v42:corpus
npm run protocol:v42:corpus:seal
npm run develop:v42
npm run audit:v42:result
npm run protocol:v42:outcome:freeze
