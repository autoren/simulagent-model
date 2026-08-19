from __future__ import annotations

import json
from typing import Any


def json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def only_class_coverage_key_type_mismatch(persisted: dict[str, Any], reconstructed: dict[str, Any]) -> bool:
    if persisted == reconstructed:
        return False
    return json_normalize(reconstructed) == persisted and set(reconstructed["class_coverage_counts"]) == {1, 2, 3}


__all__ = ["json_normalize", "only_class_coverage_key_type_mismatch"]
