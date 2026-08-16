#!/usr/bin/env python3
"""Create the V31 trained lock while attesting the API-binding amendment."""

from __future__ import annotations

import hashlib
import json
import sys

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    amendment_path = PROJECT_ROOT / "configs/v31-api-binding-repair-lock.json"
    amendment = json.loads(amendment_path.read_text())
    for path, expected in amendment["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V31 API-repair implementation changed: {path}")
    temporary = PROJECT_ROOT / "configs/v31-trained-systems-unamended.json"
    final = PROJECT_ROOT / "configs/v31-trained-systems-lock.json"
    if temporary.exists() or final.exists():
        raise RuntimeError("V31 repaired trained-system lock was already attempted")
    import freeze_v31_trained_systems as original
    sys.argv = [
        original.__file__, "--protocol-lock", str(PROJECT_ROOT / amendment["protocol_lock"]),
        "--output", str(temporary),
    ]
    original.main()
    trained = json.loads(temporary.read_text())
    trained["api_binding_repair"] = {
        "amendment": str(amendment_path.relative_to(PROJECT_ROOT)),
        "amendment_sha256": file_sha256(amendment_path),
        "unamended_intermediate": str(temporary.relative_to(PROJECT_ROOT)),
        "unamended_intermediate_sha256": file_sha256(temporary),
    }
    trained["lock_payload_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in trained.items() if key != "lock_payload_sha256"},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    final.write_text(json.dumps(trained, indent=2, sort_keys=True) + "\n")
    print(json.dumps(trained, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
