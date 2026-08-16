#!/usr/bin/env bash
set -euo pipefail

npm run audit:v51r1:repair
npm run audit:v51r1:corpus
npm run protocol:v51r1:corpus:seal
npm run calibrate:v51r1
npm run audit:v51r1:result
npm run protocol:v51r1:outcome:freeze
