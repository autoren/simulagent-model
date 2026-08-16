#!/usr/bin/env bash
set -euo pipefail

npm run audit:v51:implementation
npm run protocol:v51:implementation:freeze
npm run dataset:v51
npm run audit:v51:corpus
npm run protocol:v51:corpus:seal
npm run calibrate:v51
npm run audit:v51:result
npm run protocol:v51:outcome:freeze
