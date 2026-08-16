#!/usr/bin/env bash
set -euo pipefail

npm run audit:v33
npm run protocol:v33:freeze
npm run develop:v33
npm run audit:v33:result
npm run protocol:v33:outcome:freeze
