#!/usr/bin/env bash
set -euo pipefail

npm run audit:v36:implementation
npm run protocol:v36:implementation:freeze
npm run fit:v36:interface
npm run audit:v36:interface
npm run protocol:v36:interface:freeze
npm run dataset:v36
npm run audit:v36:corpus
npm run protocol:v36:seal
npm run features:v36
npm run protocol:v36:features:freeze
npm run evaluate:v36
npm run audit:v36:result
npm run protocol:v36:outcome:freeze
