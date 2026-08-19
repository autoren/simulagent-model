#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def canonical_json(value: Any) -> Any:
    """Return the exact JSON data model that would be observed after persistence."""
    return json.loads(json.dumps(value, sort_keys=True))


def without_rank_counts(summary: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(summary)
    projected["metrics"].pop("rank_counts", None)
    return projected


def rank_count_key_type_contract(
    recomputed: dict[str, Any], persisted: dict[str, Any]
) -> bool:
    recomputed_counts = recomputed["metrics"]["rank_counts"]
    persisted_counts = persisted["metrics"]["rank_counts"]
    return bool(
        recomputed_counts
        and persisted_counts
        and all(type(key) is int for key in recomputed_counts)
        and all(type(key) is str for key in persisted_counts)
        and {str(key): value for key, value in recomputed_counts.items()} == persisted_counts
    )


def sole_json_key_type_mismatch(
    recomputed: dict[str, Any], persisted: dict[str, Any]
) -> bool:
    return bool(
        recomputed != persisted
        and rank_count_key_type_contract(recomputed, persisted)
        and without_rank_counts(recomputed) == without_rank_counts(persisted)
        and canonical_json(recomputed) == persisted
    )
