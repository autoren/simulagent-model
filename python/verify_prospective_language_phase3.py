#!/usr/bin/env python3
"""Read-only completion audit for the prospective pilot clarification batch."""

from __future__ import annotations

import json
from pathlib import Path

from prospective_language_phase3 import verify_phase3_bundle
from prospective_language_pilot import PilotProtocolError


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "prospective-language-pilot-v1-phase3.json"
PARTICIPANT_DIR = ROOT / "data" / "prospective-language-pilot" / "prospective-language-pilot-v1" / "P001"


def main() -> int:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        report = verify_phase3_bundle(config, PARTICIPANT_DIR)
    except (PilotProtocolError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"verification": "fail", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
