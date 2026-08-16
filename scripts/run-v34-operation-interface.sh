#!/usr/bin/env bash
set -euo pipefail

npm run audit:v34
npm run protocol:v34:freeze
npm run features:v34
npm run develop:v34
npm run audit:v34:result
npm run protocol:v34:outcome:freeze
