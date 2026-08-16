#!/usr/bin/env bash
set -euo pipefail

npm run audit:v37:implementation
npm run protocol:v37:implementation:freeze
npm run dataset:v37
npm run audit:v37:corpus
npm run protocol:v37:corpus:seal
npm run features:v37
npm run protocol:v37:features:freeze
npm run evaluate:v37
npm run audit:v37:result
npm run protocol:v37:outcome:freeze
