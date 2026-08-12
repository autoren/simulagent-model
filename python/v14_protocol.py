#!/usr/bin/env python3
"""Locked record and fold utilities for the V14 operator-supported corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from v10_protocol import file_sha256, read_jsonl


def load_records_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    records = []
    for relative, expected in manifest["artifact_sha256"].items():
        path = manifest_path.parent / relative
        if file_sha256(path) != expected:
            raise RuntimeError(f"V14 artifact changed: {relative}")
        records.extend(read_jsonl(path))
    return records


def primary_folds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mechanics = sorted({record["mechanic"] for record in records})
    templates = sorted({record["template_family"] for record in records})
    lexicons = sorted({record["state_lexicon_family"] for record in records})
    operators = sorted({record["operator_family"] for record in records})
    train_split = np.asarray([record["split"] == "train" for record in records])
    evaluation_split = ~train_split
    result = [{"name": "context", "kind": "context", "train": train_split, "evaluation": evaluation_split}]
    result.extend({
        "name": f"mechanic:{heldout}", "kind": "mechanic",
        "train": np.asarray([record["split"] == "train" and record["mechanic"] != heldout for record in records]),
        "evaluation": np.asarray([record["mechanic"] == heldout for record in records]),
    } for heldout in mechanics)
    result.extend({
        "name": f"surface:{heldout}", "kind": "surface",
        "train": np.asarray([record["split"] == "train" and record["template_family"] != heldout for record in records]),
        "evaluation": np.asarray([record["split"] == "evaluation" and record["template_family"] == heldout for record in records]),
    } for heldout in templates)
    result.extend({
        "name": f"lexicon:{heldout}", "kind": "lexicon",
        "train": np.asarray([record["split"] == "train" and record["state_lexicon_family"] != heldout for record in records]),
        "evaluation": np.asarray([record["split"] == "evaluation" and record["state_lexicon_family"] == heldout for record in records]),
    } for heldout in lexicons)
    result.extend({
        "name": f"operator:{heldout}", "kind": "operator",
        "train": np.asarray([
            record["split"] == "train" and record["operator_family"] != heldout
            and record["state_lexicon_family"] == "entity_renamed" for record in records
        ]),
        "evaluation": np.asarray([
            record["operator_family"] == heldout and record["state_lexicon_family"] == "entity_renamed"
            for record in records
        ]),
    } for heldout in operators)
    result.extend({
        "name": f"combined:{operator}:{lexicon}", "kind": "combined",
        "train": np.asarray([
            record["split"] == "train" and record["operator_family"] != operator
            and record["state_lexicon_family"] != lexicon for record in records
        ]),
        "evaluation": np.asarray([
            record["operator_family"] == operator and record["state_lexicon_family"] == lexicon
            for record in records
        ]),
    } for operator in operators for lexicon in lexicons)
    if len(result) != 27:
        raise RuntimeError(f"V14 expected 27 primary folds, found {len(result)}")
    validate_folds(result)
    return result


def zero_shot_operator_folds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operators = sorted({record["semantic_operator_family"] for record in records})
    result = [{
        "name": f"semantic_operator:{heldout}", "kind": "semantic_operator_diagnostic",
        "train": np.asarray([
            record["split"] == "train" and record["semantic_operator_family"] != heldout for record in records
        ]),
        "evaluation": np.asarray([
            record["split"] == "evaluation" and record["semantic_operator_family"] == heldout for record in records
        ]),
    } for heldout in operators]
    validate_folds(result)
    return result


def validate_folds(values: list[dict[str, Any]]) -> None:
    for fold in values:
        if np.any(fold["train"] & fold["evaluation"]):
            raise RuntimeError(f"V14 fold overlaps: {fold['name']}")
        if not fold["train"].any() or not fold["evaluation"].any():
            raise RuntimeError(f"V14 fold is empty: {fold['name']}")
