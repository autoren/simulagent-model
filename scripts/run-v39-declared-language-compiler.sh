#!/usr/bin/env bash
set -euo pipefail

npm run audit:v39:implementation
npm run protocol:v39:implementation:freeze
npm run dataset:v39
npm run audit:v39:corpus
npm run protocol:v39:corpus:seal
npm run evaluate:v39
npm run audit:v39:result
npm run protocol:v39:outcome:freeze
