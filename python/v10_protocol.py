#!/usr/bin/env python3
"""Shared locked data and fold utilities for V10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TEMPORAL_ORDER = ["CURRENT", "UNKNOWN_CURRENT", "STALE_ONLY", "CONFLICTING_CURRENT"]
RELATION_ORDER = ["CONTRADICTED", "ENTAILED", "UNKNOWN"]
VALUE_ORDER = ["inactive", "active"]
ALLOWED_VALUE_SETS = [["inactive"], ["active"], ["inactive", "active"]]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_locked_records(lock: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = Path(lock["dataset_manifest"])
    if file_sha256(manifest_path) != lock["dataset_manifest_sha256"]:
        raise RuntimeError("V10 manifest changed after lock")
    records: list[dict[str, Any]] = []
    for relative, expected in lock["dataset_artifact_sha256"].items():
        path = manifest_path.parent / relative
        if file_sha256(path) != expected:
            raise RuntimeError(f"V10 data changed after lock: {relative}")
        records.extend(read_jsonl(path))
    return records


def derive_allowed_values(temporal: str, relations: tuple[str, str] | list[str]) -> list[str]:
    if temporal != "CURRENT":
        return ["inactive", "active"]
    if list(relations) == ["ENTAILED", "CONTRADICTED"]:
        return ["active"]
    if list(relations) == ["CONTRADICTED", "ENTAILED"]:
        return ["inactive"]
    return ["inactive", "active"]


def folds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mechanics = sorted({record["mechanic"] for record in records})
    templates = sorted({record["template_family"] for record in records})
    lexicons = sorted({record["state_lexicon_family"] for record in records})
    operators = sorted({record["operator_family"] for record in records})
    train_split = np.asarray([record["split"] == "train" for record in records])
    evaluation_split = ~train_split
    result: list[dict[str, Any]] = [{
        "name": "context",
        "kind": "context",
        "train": train_split.copy(),
        "evaluation": evaluation_split.copy(),
    }]
    result.extend({
        "name": f"mechanic:{heldout}",
        "kind": "mechanic",
        "train": np.asarray([
            record["split"] == "train" and record["mechanic"] != heldout
            for record in records
        ]),
        "evaluation": np.asarray([record["mechanic"] == heldout for record in records]),
    } for heldout in mechanics)
    result.extend({
        "name": f"template:{heldout}",
        "kind": "template",
        "train": np.asarray([
            record["split"] == "train" and record["template_family"] != heldout
            for record in records
        ]),
        "evaluation": np.asarray([
            record["split"] == "evaluation" and record["template_family"] == heldout
            for record in records
        ]),
    } for heldout in templates)
    result.extend({
        "name": f"lexicon:{heldout}",
        "kind": "lexicon",
        "train": np.asarray([
            record["split"] == "train" and record["state_lexicon_family"] != heldout
            for record in records
        ]),
        "evaluation": np.asarray([
            record["split"] == "evaluation" and record["state_lexicon_family"] == heldout
            for record in records
        ]),
    } for heldout in lexicons)
    result.extend({
        "name": f"operator:{heldout}",
        "kind": "operator",
        "train": np.asarray([
            record["split"] == "train"
            and record["operator_family"] != heldout
            and record["state_lexicon_family"] == "entity_renamed"
            for record in records
        ]),
        "evaluation": np.asarray([
            record["operator_family"] == heldout
            and record["state_lexicon_family"] == "entity_renamed"
            for record in records
        ]),
    } for heldout in operators)
    result.extend({
        "name": f"combined:{operator}:{lexicon}",
        "kind": "combined",
        "train": np.asarray([
            record["split"] == "train"
            and record["operator_family"] != operator
            and record["state_lexicon_family"] != lexicon
            for record in records
        ]),
        "evaluation": np.asarray([
            record["operator_family"] == operator
            and record["state_lexicon_family"] == lexicon
            for record in records
        ]),
    } for operator in operators for lexicon in lexicons)
    if len(result) != 24:
        raise RuntimeError(f"V10 expected 24 folds, found {len(result)}")
    for fold in result:
        if np.any(fold["train"] & fold["evaluation"]):
            raise RuntimeError(f"V10 fold overlaps train and evaluation: {fold['name']}")
        if not fold["train"].any() or not fold["evaluation"].any():
            raise RuntimeError(f"V10 fold is empty: {fold['name']}")
    return result
