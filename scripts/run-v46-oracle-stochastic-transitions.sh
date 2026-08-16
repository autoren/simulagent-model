#!/usr/bin/env bash
set -euo pipefail
npm run audit:v46:implementation
npm run protocol:v46:implementation:freeze
npm run dataset:v46
npm run audit:v46:corpus
npm run protocol:v46:corpus:seal
npm run develop:v46
npm run audit:v46:result
npm run protocol:v46:outcome:freeze
