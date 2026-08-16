#!/usr/bin/env bash
set -euo pipefail
npm run audit:v47:implementation
npm run protocol:v47:implementation:freeze
npm run dataset:v47
npm run audit:v47:corpus
npm run protocol:v47:corpus:seal
npm run develop:v47
npm run audit:v47:result
npm run protocol:v47:outcome:freeze
