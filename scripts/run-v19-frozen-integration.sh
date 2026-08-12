#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."
.venv/bin/python python/build_v19_grounding_views.py --config configs/v19-frozen-integration.json
.venv/bin/python python/audit_v19_compatibility.py --config configs/v19-frozen-integration.json --dataset data/v19 --output outputs/v19-frozen-integration/pre-extraction-audit.json
.venv/bin/python python/freeze_v19_integration.py --config configs/v19-frozen-integration.json --plan docs/v19-frozen-integration-plan.md --audit outputs/v19-frozen-integration/pre-extraction-audit.json --output configs/v19-frozen-integration-lock.json
.venv/bin/python python/extract_v19_grounding_features_mlx.py --lock configs/v19-frozen-integration-lock.json --output-dir outputs/v19-frozen-integration/features
.venv/bin/python python/evaluate_v19_frozen_integration.py --lock configs/v19-frozen-integration-lock.json --features outputs/v19-frozen-integration/features --output-dir outputs/v19-frozen-integration/evaluation
.venv/bin/python python/replay_v19_error_conditioning.py --lock configs/v19-frozen-integration-lock.json --result outputs/v19-frozen-integration/evaluation/result.json --output outputs/v19-frozen-integration/error-conditioning-replay.json
.venv/bin/python python/audit_v19_result.py --lock configs/v19-frozen-integration-lock.json --result outputs/v19-frozen-integration/evaluation/result.json --correction outputs/v19-frozen-integration/error-conditioning-replay.json --output outputs/v19-frozen-integration/post-result-audit.json
.venv/bin/python python/summarize_v19.py --result outputs/v19-frozen-integration/evaluation/result.json --audit outputs/v19-frozen-integration/post-result-audit.json --correction outputs/v19-frozen-integration/error-conditioning-replay.json --output docs/v19-results.md
