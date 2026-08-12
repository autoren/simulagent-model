#!/usr/bin/env python3
"""Audit V9 grounding data for structural and pre-model shortcuts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v9-grounding-lock.json")
    parser.add_argument("--output", default="outputs/v9-pre-model/shortcut-audit.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def classifier(c_value: float = 1.0) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=3000,
        random_state=0,
        solver="lbfgs",
    )


def folds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mechanics = sorted({record["mechanic"] for record in records})
    templates = sorted({record["template_family"] for record in records})
    operators = sorted({record["operator_family"] for record in records})
    result = [{
        "name": "context",
        "kind": "context",
        "train": np.asarray([record["split"] == "train" for record in records]),
        "evaluation": np.asarray([record["split"] == "calibration" for record in records]),
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
            record["split"] == "calibration" and record["template_family"] == heldout
            for record in records
        ]),
    } for heldout in templates)
    result.extend({
        "name": f"operator:{heldout}",
        "kind": "operator",
        "train": np.asarray([
            record["split"] == "train" and record["operator_family"] != heldout
            for record in records
        ]),
        "evaluation": np.asarray([record["operator_family"] == heldout for record in records]),
    } for heldout in operators)
    return result


def metric(gold: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predicted = scores >= 0
    return {
        "balanced_accuracy": float(balanced_accuracy_score(gold, predicted)),
        "roc_auc": float(roc_auc_score(gold, scores)),
    }


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError(f"V9 shortcut audit already exists: {output_path}")
    lock = json.loads(lock_path.read_text())
    manifest_path = Path(lock["expected_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["grounding_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V9 corpus does not share the grounding lock")
    root = manifest_path.parent
    records: list[dict[str, Any]] = []
    for relative, expected in manifest["artifact_sha256"].items():
        path = root / relative
        if file_sha256(path) != expected:
            raise RuntimeError(f"V9 data artifact changed: {relative}")
        records.extend(read_jsonl(path))
    record_folds = folds(records)

    pair_record_indices = []
    pair_metadata = []
    pair_position = []
    pair_text = []
    pair_gold = []
    pair_value = []
    pair_temporal = []
    value_order = ["inactive", "active", "active|inactive"]
    temporal_order = ["CURRENT", "UNKNOWN_CURRENT", "STALE_ONLY", "CONFLICTING_CURRENT"]
    for record_index, record in enumerate(records):
        units = record["evidence_units"]
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            positive_index = next(
                index for index, unit in enumerate(units)
                if unit["start"] == target["evidence_span"]["start"]
                and unit["end"] == target["evidence_span"]["end"]
            )
            query = record["agent_input"]["transition_determinants"][determinant_index]
            for evidence_index, unit in enumerate(units):
                pair_record_indices.append(record_index)
                pair_metadata.append({
                    "determinant_position": str(determinant_index),
                    "evidence_position": str(evidence_index),
                    "determinant_count": str(len(record["agent_input"]["transition_determinants"])),
                    "mechanic": record["mechanic"],
                    "operator": record["operator_family"],
                    "template": record["template_family"],
                    "surface": record["surface_variant"],
                })
                pair_position.append({
                    "determinant_position": str(determinant_index),
                    "evidence_position": str(evidence_index),
                    "determinant_count": str(len(record["agent_input"]["transition_determinants"])),
                })
                pair_text.append(
                    f"Action: {record['agent_input']['candidate_action']}\n"
                    f"Query: {query['label']}\nEvidence: {unit['text']}"
                )
                matched = evidence_index == positive_index
                pair_gold.append(matched)
                pair_value.append(value_order.index("|".join(sorted(target["allowed_values"]))) if matched else -1)
                pair_temporal.append(temporal_order.index(target["temporal_status"]) if matched else -1)
    pair_record_indices_np = np.asarray(pair_record_indices)
    pair_gold_np = np.asarray(pair_gold, dtype=bool)
    pair_value_np = np.asarray(pair_value, dtype=np.int32)
    pair_temporal_np = np.asarray(pair_temporal, dtype=np.int32)

    metadata_metrics = {}
    position_metrics = {}
    context_code_metrics = {}
    ambiguous = np.asarray([not record["target"]["identifiable"] for record in records], dtype=bool)
    scene_codes = [re.search(r"Audit scene ([0-9a-f]+)", record["agent_input"]["observation"]).group(1) for record in records]
    for fold in record_folds:
        train_pairs = fold["train"][pair_record_indices_np]
        evaluation_pairs = fold["evaluation"][pair_record_indices_np]
        metadata_model = make_pipeline(DictVectorizer(), classifier())
        metadata_model.fit(
            [pair_metadata[index] for index in np.flatnonzero(train_pairs)],
            pair_gold_np[train_pairs],
        )
        metadata_scores = metadata_model.decision_function(
            [pair_metadata[index] for index in np.flatnonzero(evaluation_pairs)]
        )
        metadata_metrics[fold["name"]] = metric(pair_gold_np[evaluation_pairs], metadata_scores)

        position_model = make_pipeline(DictVectorizer(), classifier())
        position_model.fit(
            [pair_position[index] for index in np.flatnonzero(train_pairs)],
            pair_gold_np[train_pairs],
        )
        position_scores = position_model.decision_function(
            [pair_position[index] for index in np.flatnonzero(evaluation_pairs)]
        )
        position_metrics[fold["name"]] = metric(pair_gold_np[evaluation_pairs], position_scores)

        code_model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False),
            classifier(),
        )
        code_model.fit(
            [scene_codes[index] for index in np.flatnonzero(fold["train"])],
            ambiguous[fold["train"]],
        )
        code_scores = code_model.decision_function(
            [scene_codes[index] for index in np.flatnonzero(fold["evaluation"])]
        )
        context_code_metrics[fold["name"]] = metric(ambiguous[fold["evaluation"]], code_scores)

    context_fold = record_folds[0]
    train_pairs = context_fold["train"][pair_record_indices_np]
    evaluation_pairs = context_fold["evaluation"][pair_record_indices_np]
    lexical_match = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000),
        classifier(),
    )
    lexical_match.fit(
        [pair_text[index] for index in np.flatnonzero(train_pairs)],
        pair_gold_np[train_pairs],
    )
    lexical_match_scores = lexical_match.decision_function(
        [pair_text[index] for index in np.flatnonzero(evaluation_pairs)]
    )
    linguistic = {"context_match": metric(pair_gold_np[evaluation_pairs], lexical_match_scores)}
    for name, targets in (("allowed_values", pair_value_np), ("temporal_status", pair_temporal_np)):
        train_positive = train_pairs & pair_gold_np
        evaluation_positive = evaluation_pairs & pair_gold_np
        model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000),
            classifier(),
        )
        model.fit(
            [pair_text[index] for index in np.flatnonzero(train_positive)],
            targets[train_positive],
        )
        predicted = model.predict([pair_text[index] for index in np.flatnonzero(evaluation_positive)])
        linguistic[f"context_{name}"] = {
            "accuracy": float(np.mean(predicted == targets[evaluation_positive])),
            "examples": int(evaluation_positive.sum()),
        }

    gates_config = lock["config"]["shortcutGates"]
    checks = [
        {
            "name": "metadata_match_maximum_fold_balanced_accuracy",
            "value": max(value["balanced_accuracy"] for value in metadata_metrics.values()),
            "maximum": gates_config["maximumMetadataMatchBalancedAccuracy"],
        },
        {
            "name": "position_match_maximum_fold_balanced_accuracy",
            "value": max(value["balanced_accuracy"] for value in position_metrics.values()),
            "maximum": gates_config["maximumPositionMatchBalancedAccuracy"],
        },
        {
            "name": "context_code_maximum_fold_balanced_accuracy",
            "value": max(value["balanced_accuracy"] for value in context_code_metrics.values()),
            "maximum": gates_config["maximumContextCodeBalancedAccuracy"],
        },
    ]
    for check in checks:
        check["passed"] = check["value"] <= check["maximum"]
    structural = manifest["validation"]
    structural_passed = (
        not structural["errors"]
        and structural["context_cross_split_overlaps"] == 0
        and structural["conflicting_duplicate_prompts"] == 0
        and structural["cross_split_duplicate_prompts"] == 0
        and structural["malformed_spans"] == 0
        and structural["symbolic_mismatches"] == 0
        and structural["determinant_ids_in_observation"] == 0
        and structural["literal_value_labels_in_observation"] == 0
    )
    report = {
        "schema_version": 9,
        "experiment": "v9_grounding_pre_model_shortcut_audit",
        "grounding_lock": str(lock_path),
        "grounding_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "records": len(records),
        "pair_examples": len(pair_gold),
        "folds": [fold["name"] for fold in record_folds],
        "audits": {
            "metadata_match": metadata_metrics,
            "position_match": position_metrics,
            "context_code": context_code_metrics,
            "reported_linguistic_baselines": linguistic,
        },
        "structural_passed": structural_passed,
        "gates": {"passed": structural_passed and all(check["passed"] for check in checks), "checks": checks},
        "decision": "authorize_frozen_grounding" if structural_passed and all(check["passed"] for check in checks) else "stop_before_model_access",
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
