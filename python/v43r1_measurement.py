"""Canonical, duplicate-safe graph comparison registered for V43r1."""
from __future__ import annotations
from typing import Any, Sequence
from v22_relational import canonical_json

def canonical_graph(rows: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(canonical_json(row) for row in rows))

def duplicate_free(rows: Sequence[dict[str, Any]]) -> bool:
    encoded=canonical_graph(rows)
    return len(encoded)==len(set(encoded))

def graph_equal(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> bool:
    return duplicate_free(left) and duplicate_free(right) and canonical_graph(left)==canonical_graph(right)
