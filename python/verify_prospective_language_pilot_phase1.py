#!/usr/bin/env python3
"""Read-only acceptance-gate audit for a completed prospective pilot Phase 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prospective_language_pilot import PilotProtocolError, load_study_config, verify_phase_1_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "prospective-language-pilot-v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a completed Phase 1 bundle satisfies every integrity gate."
    )
    parser.add_argument("participant_dir", type=Path, help="Participant session directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Frozen study config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_study_config(args.config)
        report = verify_phase_1_bundle(config, args.participant_dir)
    except (PilotProtocolError, OSError) as exc:
        print(json.dumps({"verification": "fail", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
