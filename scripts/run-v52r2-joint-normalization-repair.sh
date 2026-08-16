#!/usr/bin/env bash
set -euo pipefail

npm run audit:v52r2:implementation
npm run protocol:v52r2:implementation:freeze
npm run evaluate:v52r2
npm run audit:v52r2:result
npm run protocol:v52r2:outcome:freeze
