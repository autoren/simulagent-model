from __future__ import annotations

from copy import deepcopy
from typing import Any


def without_population_build_count(value: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(value)
    projected["access"].pop("population_build_count", None)
    return projected


def sole_population_build_count_alias_mismatch(
    reconstructed: dict[str, Any], persisted: dict[str, Any]
) -> bool:
    return bool(
        reconstructed != persisted
        and "population_build_count" not in reconstructed["access"]
        and persisted["access"].get("population_build_count") == 1
        and without_population_build_count(persisted) == reconstructed
    )


__all__ = [
    "sole_population_build_count_alias_mismatch",
    "without_population_build_count",
]
