#!/usr/bin/env bash
set -euo pipefail
npm run audit:v48:implementation
npm run protocol:v48:implementation:freeze
npm run dataset:v48
npm run audit:v48:corpus
npm run protocol:v48:corpus:seal
npm run develop:v48
npm run audit:v48:result
npm run protocol:v48:outcome:freeze
