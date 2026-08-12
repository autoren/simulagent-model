#!/usr/bin/env python3
"""Run V10 structural and pre-model shortcut gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from v10_protocol import TEMPORAL_ORDER, file_sha256, folds, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v10-grounding-lock.json")
    parser.add_argument("--output", default="outputs/v10-pre-model/shortcut-audit.json")
    return parser.parse_args()


def classifier() -> LogisticRegression:
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=0, solver="lbfgs")


def metric(gold: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(gold, scores >= 0)),
        "roc_auc": float(roc_auc_score(gold, scores)),
    }


def fit_dict_metric(features: list[dict[str, str]], gold: np.ndarray, train: np.ndarray, evaluation: np.ndarray) -> dict[str, float]:
    model = make_pipeline(DictVectorizer(), classifier())
    model.fit([features[index] for index in np.flatnonzero(train)], gold[train])
    scores = model.decision_function([features[index] for index in np.flatnonzero(evaluation)])
    return metric(gold[evaluation], scores)


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError(f"V10 shortcut audit already exists: {output_path}")
    lock = json.loads(lock_path.read_text())
    manifest_path = Path(lock["expected_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["grounding_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V10 corpus does not share its grounding lock")
    records: list[dict[str, Any]] = []
    for relative, expected in manifest["artifact_sha256"].items():
        path = manifest_path.parent / relative
        if file_sha256(path) != expected:
            raise RuntimeError(f"V10 artifact changed: {relative}")
        records.extend(read_jsonl(path))
    record_folds = folds(records)

    pair_records: list[int] = []
    pair_metadata: list[dict[str, str]] = []
    pair_positions: list[dict[str, str]] = []
    pair_text: list[str] = []
    pair_match: list[bool] = []
    pair_temporal: list[int] = []
    pair_current_value: list[int] = []
    relation_pair_indices: list[int] = []
    relation_positions: list[dict[str, str]] = []
    relation_text: list[str] = []
    relation_gold: list[bool] = []
    for record_index, record in enumerate(records):
        units = record["evidence_units"]
        hypotheses = {value["determinant_id"]: value["statements"] for value in record["agent_input"]["state_hypotheses"]}
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            positive = next(index for index, unit in enumerate(units) if (
                unit["start"] == target["evidence_span"]["start"] and unit["end"] == target["evidence_span"]["end"]
            ))
            query = record["agent_input"]["transition_determinants"][determinant_index]
            for evidence_index, unit in enumerate(units):
                pair_index = len(pair_records)
                pair_records.append(record_index)
                base = {
                    "determinant_position": str(determinant_index),
                    "evidence_position": str(evidence_index),
                    "determinant_count": str(len(record["agent_input"]["transition_determinants"])),
                }
                pair_positions.append(base)
                pair_metadata.append({
                    **base,
                    "mechanic": record["mechanic"],
                    "operator": record["operator_family"],
                    "template": record["template_family"],
                    "lexicon": record["state_lexicon_family"],
                })
                pair_text.append(f"Action: {record['agent_input']['candidate_action']}\nQuery: {query['label']}\nEvidence: {unit['text']}")
                matched = evidence_index == positive
                pair_match.append(matched)
                pair_temporal.append(TEMPORAL_ORDER.index(target["temporal_status"]) if matched else -1)
                current = target["current_value"]
                pair_current_value.append((1 if current == "active" else 0) if matched and current is not None else -1)
                if matched and current is not None:
                    for hypothesis_index, statement in enumerate(hypotheses[target["determinant_id"]]):
                        relation_pair_indices.append(pair_index)
                        relation_positions.append({"hypothesis_position": str(hypothesis_index)})
                        relation_text.append(f"Evidence: {unit['text']}\nHypothesis: {statement}")
                        relation_gold.append(target["hypothesis_relations"][hypothesis_index] == "ENTAILED")

    pair_records_np = np.asarray(pair_records, dtype=np.int32)
    match_np = np.asarray(pair_match, dtype=bool)
    temporal_np = np.asarray(pair_temporal, dtype=np.int8)
    current_np = np.asarray(pair_current_value, dtype=np.int8)
    relation_pairs_np = np.asarray(relation_pair_indices, dtype=np.int32)
    relation_gold_np = np.asarray(relation_gold, dtype=bool)
    audits: dict[str, dict[str, Any]] = {
        "metadata_match": {},
        "position_match": {},
        "metadata_polarity": {},
        "hypothesis_position_relation": {},
    }
    for fold in record_folds:
        train_pairs = fold["train"][pair_records_np]
        eval_pairs = fold["evaluation"][pair_records_np]
        audits["metadata_match"][fold["name"]] = fit_dict_metric(pair_metadata, match_np, train_pairs, eval_pairs)
        audits["position_match"][fold["name"]] = fit_dict_metric(pair_positions, match_np, train_pairs, eval_pairs)
        current_train = train_pairs & (current_np >= 0)
        current_eval = eval_pairs & (current_np >= 0)
        audits["metadata_polarity"][fold["name"]] = fit_dict_metric(
            pair_metadata,
            current_np == 1,
            current_train,
            current_eval,
        )
        relation_train = fold["train"][pair_records_np[relation_pairs_np]]
        relation_eval = fold["evaluation"][pair_records_np[relation_pairs_np]]
        audits["hypothesis_position_relation"][fold["name"]] = fit_dict_metric(
            relation_positions,
            relation_gold_np,
            relation_train,
            relation_eval,
        )

    context = record_folds[0]
    train_pairs = context["train"][pair_records_np]
    eval_pairs = context["evaluation"][pair_records_np]
    lexical: dict[str, Any] = {}
    char_options = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000)
    match_model = make_pipeline(TfidfVectorizer(**char_options), classifier())
    match_model.fit([pair_text[index] for index in np.flatnonzero(train_pairs)], match_np[train_pairs])
    match_scores = match_model.decision_function([pair_text[index] for index in np.flatnonzero(eval_pairs)])
    lexical["context_match"] = metric(match_np[eval_pairs], match_scores)
    for name, targets, valid in (
        ("temporal", temporal_np, temporal_np >= 0),
        ("current_polarity", current_np, current_np >= 0),
    ):
        train = train_pairs & valid
        evaluation = eval_pairs & valid
        model = make_pipeline(TfidfVectorizer(**char_options), classifier())
        model.fit([pair_text[index] for index in np.flatnonzero(train)], targets[train])
        predicted = model.predict([pair_text[index] for index in np.flatnonzero(evaluation)])
        lexical[f"context_{name}"] = {"accuracy": float(np.mean(predicted == targets[evaluation])), "examples": int(evaluation.sum())}
    relation_train = context["train"][pair_records_np[relation_pairs_np]]
    relation_eval = context["evaluation"][pair_records_np[relation_pairs_np]]
    relation_model = make_pipeline(TfidfVectorizer(**char_options), classifier())
    relation_model.fit([relation_text[index] for index in np.flatnonzero(relation_train)], relation_gold_np[relation_train])
    relation_scores = relation_model.decision_function([relation_text[index] for index in np.flatnonzero(relation_eval)])
    lexical["context_relation"] = metric(relation_gold_np[relation_eval], relation_scores)
    audits["reported_linguistic_baselines"] = lexical

    gate_config = lock["config"]["shortcutGates"]
    specifications = [
        ("metadata_match", "maximumMetadataMatchBalancedAccuracy"),
        ("position_match", "maximumPositionMatchBalancedAccuracy"),
        ("metadata_polarity", "maximumMetadataPolarityBalancedAccuracy"),
        ("hypothesis_position_relation", "maximumHypothesisPositionRelationBalancedAccuracy"),
    ]
    checks = []
    for audit_name, config_name in specifications:
        value = max(item["balanced_accuracy"] for item in audits[audit_name].values())
        maximum = gate_config[config_name]
        checks.append({"name": f"{audit_name}_maximum_fold_balanced_accuracy", "value": value, "maximum": maximum, "passed": value <= maximum})
    validation = manifest["validation"]
    structural_passed = not validation["errors"] and all(validation[name] == 0 for name in (
        "malformed_spans",
        "malformed_hypotheses",
        "relation_mismatches",
        "allowed_value_derivation_mismatches",
        "symbolic_mismatches",
        "imbalanced_current_cells",
        "complement_cross_split_overlaps",
        "context_cross_split_overlaps",
        "duplicate_prompts",
        "cross_split_duplicate_prompts",
        "conflicting_duplicate_prompts",
    ))
    passed = structural_passed and all(check["passed"] for check in checks)
    report = {
        "schema_version": 10,
        "experiment": "v10_pre_model_shortcut_audit",
        "grounding_lock": str(lock_path),
        "grounding_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "records": len(records),
        "pair_examples": len(pair_records),
        "current_relation_examples": len(relation_gold),
        "folds": [{"name": fold["name"], "kind": fold["kind"]} for fold in record_folds],
        "audits": audits,
        "structural_passed": structural_passed,
        "gates": {"passed": passed, "checks": checks},
        "decision": "authorize_v10_frozen_extraction" if passed else "stop_before_model_access",
        "data_access": lock["data_access"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
