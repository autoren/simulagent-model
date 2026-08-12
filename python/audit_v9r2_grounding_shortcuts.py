#!/usr/bin/env python3
"""Audit V9r2 after removal of the synthetic context identifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from audit_v9_grounding_shortcuts import classifier, file_sha256, folds, metric, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v9r2-grounding-lock.json")
    parser.add_argument("--output", default="outputs/v9r2-pre-model/shortcut-audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError(f"V9r2 shortcut audit already exists: {output_path}")
    lock = json.loads(lock_path.read_text())
    manifest_path = Path(lock["expected_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["grounding_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V9r2 corpus does not share the grounding lock")
    records: list[dict[str, Any]] = []
    for relative, expected in manifest["artifact_sha256"].items():
        path = manifest_path.parent / relative
        if file_sha256(path) != expected:
            raise RuntimeError(f"V9r2 data artifact changed: {relative}")
        records.extend(read_jsonl(path))
    record_folds = folds(records)

    pair_record_indices = []
    metadata = []
    position = []
    text = []
    gold = []
    values = []
    temporal = []
    value_order = ["inactive", "active", "active|inactive"]
    temporal_order = ["CURRENT", "UNKNOWN_CURRENT", "STALE_ONLY", "CONFLICTING_CURRENT"]
    for record_index, record in enumerate(records):
        units = record["evidence_units"]
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            positive = next(index for index, unit in enumerate(units) if (
                unit["start"] == target["evidence_span"]["start"]
                and unit["end"] == target["evidence_span"]["end"]
            ))
            query = record["agent_input"]["transition_determinants"][determinant_index]
            for evidence_index, unit in enumerate(units):
                pair_record_indices.append(record_index)
                base = {
                    "determinant_position": str(determinant_index),
                    "evidence_position": str(evidence_index),
                    "determinant_count": str(len(record["agent_input"]["transition_determinants"])),
                }
                position.append(base)
                metadata.append({
                    **base,
                    "mechanic": record["mechanic"],
                    "operator": record["operator_family"],
                    "template": record["template_family"],
                    "surface": record["surface_variant"],
                })
                text.append(
                    f"Action: {record['agent_input']['candidate_action']}\n"
                    f"Query: {query['label']}\nEvidence: {unit['text']}"
                )
                matched = evidence_index == positive
                gold.append(matched)
                values.append(value_order.index("|".join(sorted(target["allowed_values"]))) if matched else -1)
                temporal.append(temporal_order.index(target["temporal_status"]) if matched else -1)
    pair_records = np.asarray(pair_record_indices)
    gold_np = np.asarray(gold, dtype=bool)
    values_np = np.asarray(values, dtype=np.int32)
    temporal_np = np.asarray(temporal, dtype=np.int32)

    metadata_metrics = {}
    position_metrics = {}
    for fold in record_folds:
        train = fold["train"][pair_records]
        evaluation = fold["evaluation"][pair_records]
        for source, destination in ((metadata, metadata_metrics), (position, position_metrics)):
            model = make_pipeline(DictVectorizer(), classifier())
            model.fit([source[index] for index in np.flatnonzero(train)], gold_np[train])
            scores = model.decision_function([source[index] for index in np.flatnonzero(evaluation)])
            destination[fold["name"]] = metric(gold_np[evaluation], scores)

    context = record_folds[0]
    train = context["train"][pair_records]
    evaluation = context["evaluation"][pair_records]
    lexical = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000),
        classifier(),
    )
    lexical.fit([text[index] for index in np.flatnonzero(train)], gold_np[train])
    scores = lexical.decision_function([text[index] for index in np.flatnonzero(evaluation)])
    linguistic: dict[str, Any] = {"context_match": metric(gold_np[evaluation], scores)}
    for name, targets in (("allowed_values", values_np), ("temporal_status", temporal_np)):
        train_positive = train & gold_np
        evaluation_positive = evaluation & gold_np
        model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000),
            classifier(),
        )
        model.fit([text[index] for index in np.flatnonzero(train_positive)], targets[train_positive])
        predicted = model.predict([text[index] for index in np.flatnonzero(evaluation_positive)])
        linguistic[f"context_{name}"] = {
            "accuracy": float(np.mean(predicted == targets[evaluation_positive])),
            "examples": int(evaluation_positive.sum()),
        }

    gate_config = lock["config"]["shortcutGates"]
    checks = [
        {
            "name": "metadata_match_maximum_fold_balanced_accuracy",
            "value": max(value["balanced_accuracy"] for value in metadata_metrics.values()),
            "maximum": gate_config["maximumMetadataMatchBalancedAccuracy"],
        },
        {
            "name": "position_match_maximum_fold_balanced_accuracy",
            "value": max(value["balanced_accuracy"] for value in position_metrics.values()),
            "maximum": gate_config["maximumPositionMatchBalancedAccuracy"],
        },
    ]
    for check in checks:
        check["passed"] = check["value"] <= check["maximum"]
    validation = manifest["validation"]
    structural_passed = (
        not validation["errors"]
        and validation["synthetic_context_identifiers"] == 0
        and validation["context_cross_split_overlaps"] == 0
        and validation["conflicting_duplicate_prompts"] == 0
        and validation["cross_split_duplicate_prompts"] == 0
        and validation["malformed_spans"] == 0
        and validation["symbolic_mismatches"] == 0
    )
    passed = structural_passed and all(check["passed"] for check in checks)
    report = {
        "schema_version": 9,
        "revision": 2,
        "experiment": "v9r2_grounding_pre_model_shortcut_audit",
        "grounding_lock": str(lock_path),
        "grounding_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "records": len(records),
        "pair_examples": len(gold),
        "folds": [fold["name"] for fold in record_folds],
        "audits": {
            "metadata_match": metadata_metrics,
            "position_match": position_metrics,
            "reported_linguistic_baselines": linguistic,
        },
        "structural_passed": structural_passed,
        "gates": {"passed": passed, "checks": checks},
        "decision": "authorize_frozen_grounding" if passed else "stop_before_model_access",
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
            "final_v9_mechanic_records_read": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
