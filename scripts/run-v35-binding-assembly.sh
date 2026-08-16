#!/usr/bin/env bash
set -euo pipefail

npm run audit:v35
npm run protocol:v35:freeze
npm run features:v35
npm run develop:v35
npm run audit:v35:result
npm run protocol:v35:outcome:freeze
