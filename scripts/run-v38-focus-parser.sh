#!/usr/bin/env bash
set -euo pipefail
npm run audit:v38:implementation
npm run protocol:v38:implementation:freeze
npm run dataset:v38
npm run audit:v38:corpus
npm run protocol:v38:corpus:seal
npm run features:v38
npm run protocol:v38:features:freeze
npm run evaluate:v38
npm run audit:v38:result
npm run protocol:v38:outcome:freeze
