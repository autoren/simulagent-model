#!/usr/bin/env python3
"""Run the locked 13-fold V9r2 frozen evidence-grounding evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_v9_grounding_shortcuts import folds, read_jsonl
from binary_metrics import evaluate_binary
from v9_symbolic import evaluate_allowed_transitions


VALUE_SETS = [["inactive"], ["active"], ["inactive", "active"]]
TEMPORAL_ORDER = ["CURRENT", "UNKNOWN_CURRENT", "STALE_ONLY", "CONFLICTING_CURRENT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v9-frozen-grounding-lock.json")
    parser.add_argument("--features", default="outputs/v9-frozen-grounding/features")
    parser.add_argument("--output-dir", default="outputs/v9-frozen-grounding/evaluation")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def token_f1(left: str, right: str) -> float:
    left_tokens = re.findall(r"[a-z0-9]+", left.lower())
    right_tokens = re.findall(r"[a-z0-9]+", right.lower())
    left_counts = {token: left_tokens.count(token) for token in set(left_tokens)}
    right_counts = {token: right_tokens.count(token) for token in set(right_tokens)}
    overlap = sum(min(left_counts.get(token, 0), right_counts.get(token, 0)) for token in set(left_counts) | set(right_counts))
    precision = overlap / len(left_tokens) if left_tokens else 0.0
    recall = overlap / len(right_tokens) if right_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_fold_records(
    records: list[dict[str, Any]],
    evaluation_mask: np.ndarray,
    pair_record_indices: np.ndarray,
    determinant_indices: np.ndarray,
    evidence_indices: np.ndarray,
    match_targets: np.ndarray,
    match_scores: np.ndarray,
    predicted_values: np.ndarray,
    predicted_temporal: np.ndarray,
) -> tuple[dict[str, Any], dict[int, bool]]:
    pair_lookup: dict[tuple[int, int], list[int]] = {}
    for pair_index in np.flatnonzero(evaluation_mask[pair_record_indices]):
        key = (int(pair_record_indices[pair_index]), int(determinant_indices[pair_index]))
        pair_lookup.setdefault(key, []).append(int(pair_index))
    span_correct = []
    span_token_f1 = []
    value_correct = []
    temporal_correct = []
    ledger_correct = []
    possible_correct = []
    gold_ambiguous = []
    predicted_ambiguous = []
    record_predictions: dict[int, bool] = {}
    for record_index in np.flatnonzero(evaluation_mask):
        record = records[record_index]
        predicted_grounding = []
        record_exact = True
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            candidates = pair_lookup[(int(record_index), determinant_index)]
            selected = max(candidates, key=lambda index: float(match_scores[index]))
            gold_pair = next(index for index in candidates if bool(match_targets[index]))
            selected_evidence = int(evidence_indices[selected])
            gold_evidence = int(evidence_indices[gold_pair])
            span_ok = selected_evidence == gold_evidence
            value_ok = int(predicted_values[selected]) == int(
                {"inactive": 0, "active": 1, "active|inactive": 2}[
                    "|".join(sorted(target["allowed_values"]))
                ]
            )
            temporal_ok = TEMPORAL_ORDER[int(predicted_temporal[selected])] == target["temporal_status"]
            predicted_span = record["evidence_units"][selected_evidence]["text"]
            span_correct.append(span_ok)
            span_token_f1.append(token_f1(predicted_span, target["evidence_span"]["text"]))
            value_correct.append(value_ok)
            temporal_correct.append(temporal_ok)
            record_exact = record_exact and span_ok and value_ok and temporal_ok
            predicted_grounding.append({
                "determinant_id": target["determinant_id"],
                "allowed_values": VALUE_SETS[int(predicted_values[selected])],
            })
        symbolic = evaluate_allowed_transitions(record["action_dependency_schema"], predicted_grounding)
        ambiguous = not symbolic["identifiable"]
        record_predictions[int(record_index)] = ambiguous
        gold_ambiguous.append(not record["target"]["identifiable"])
        predicted_ambiguous.append(ambiguous)
        possible_correct.append(
            symbolic["possible_transition_codes"] == record["target"]["possible_transition_codes"]
        )
        ledger_correct.append(record_exact)

    flip_pairs = []
    groups: dict[tuple[str, str], list[int]] = {}
    for record_index in np.flatnonzero(evaluation_mask):
        record = records[record_index]
        if record["intervention_kind"] != "oracle_label_flip":
            continue
        groups.setdefault((record["intervention_group_id"], record["surface_variant"]), []).append(int(record_index))
    for indices in groups.values():
        unresolved = [index for index in indices if records[index]["intervention_member"] == "relevant_unresolved"]
        resolved = [index for index in indices if records[index]["intervention_member"] == "relevant_resolved"]
        if len(unresolved) != 1 or len(resolved) != 1:
            raise RuntimeError(f"Malformed V9 evaluation pair: {indices}")
        flip_pairs.append(record_predictions[unresolved[0]] and not record_predictions[resolved[0]])
    symbolic_metrics = evaluate_binary(
        gold_ambiguous,
        [float(value) for value in predicted_ambiguous],
        0.5,
    )
    evaluation_pairs = evaluation_mask[pair_record_indices]
    return {
        "records": int(evaluation_mask.sum()),
        "determinants": len(span_correct),
        "span_accuracy": float(np.mean(span_correct)),
        "span_token_f1": float(np.mean(span_token_f1)),
        "determinant_identification_accuracy": float(np.mean(span_correct)),
        "allowed_values_accuracy": float(np.mean(value_correct)),
        "temporal_accuracy": float(np.mean(temporal_correct)),
        "complete_ledger_accuracy": float(np.mean(ledger_correct)),
        "pair_match_auc": float(roc_auc_score(
            match_targets[evaluation_pairs],
            match_scores[evaluation_pairs],
        )),
        "symbolic_identifiability": symbolic_metrics,
        "possible_transition_set_exact_accuracy": float(np.mean(possible_correct)),
        "label_flip_pairs": len(flip_pairs),
        "complete_flip_pair_accuracy": float(np.mean(flip_pairs)),
    }, record_predictions


def gate_report(results: dict[str, dict[str, Any]], gates: dict[str, float]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for result in results.values():
        by_kind.setdefault(result["kind"], []).append(result["overall"])
    checks = [
        ("context_span_accuracy", by_kind["context"][0]["span_accuracy"], gates["minimumContextSpanAccuracy"]),
        ("minimum_mechanic_span_accuracy", min(value["span_accuracy"] for value in by_kind["mechanic"]), gates["minimumEveryMechanicSpanAccuracy"]),
        ("minimum_template_span_accuracy", min(value["span_accuracy"] for value in by_kind["template"]), gates["minimumEveryTemplateSpanAccuracy"]),
        ("minimum_operator_span_accuracy", min(value["span_accuracy"] for value in by_kind["operator"]), gates["minimumEveryOperatorSpanAccuracy"]),
        ("minimum_fold_allowed_values_accuracy", min(value["allowed_values_accuracy"] for values in by_kind.values() for value in values), gates["minimumEveryFoldAllowedValuesAccuracy"]),
        ("minimum_fold_temporal_accuracy", min(value["temporal_accuracy"] for values in by_kind.values() for value in values), gates["minimumEveryFoldTemporalAccuracy"]),
        ("minimum_fold_symbolic_balanced_accuracy", min(value["symbolic_identifiability"]["balanced_accuracy"] for values in by_kind.values() for value in values), gates["minimumEveryFoldSymbolicBalancedAccuracy"]),
        ("minimum_fold_complete_flip_pair_accuracy", min(value["complete_flip_pair_accuracy"] for values in by_kind.values() for value in values), gates["minimumEveryFoldCompleteFlipPairAccuracy"]),
    ]
    values = [
        {"name": name, "value": value, "minimum": minimum, "passed": value >= minimum}
        for name, value, minimum in checks
    ]
    return {"passed": all(value["passed"] for value in values), "checks": values}


def save_pipeline(path: Path, prefix: str, model: Any, payload: dict[str, np.ndarray]) -> None:
    scaler = model.named_steps["standardscaler"]
    head = model.named_steps["logisticregression"]
    payload[f"{prefix}_scaler_mean"] = scaler.mean_
    payload[f"{prefix}_scaler_scale"] = scaler.scale_
    payload[f"{prefix}_coef"] = head.coef_
    payload[f"{prefix}_intercept"] = head.intercept_
    payload[f"{prefix}_classes"] = head.classes_


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    feature_root = Path(args.features)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"V9 evaluation result already exists: {result_path}")
    lock = json.loads(lock_path.read_text())
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V9 features do not share the frozen-grounding lock")
    feature_path = Path(metadata["feature_artifact"])
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V9 feature artifact changed")
    manifest_path = Path(lock["dataset_manifest"])
    records = []
    for relative, expected in lock["dataset_artifact_sha256"].items():
        path = manifest_path.parent / relative
        if file_sha256(path) != expected:
            raise RuntimeError(f"V9 data changed after lock: {relative}")
        records.extend(read_jsonl(path))
    with np.load(feature_path, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    if arrays["record_ids"].tolist() != [record["id"] for record in records]:
        raise RuntimeError("V9 feature and record order differ")
    pair_features = arrays["unique_features"][arrays["pair_feature_indices"]].astype(np.float32)
    pair_records = arrays["pair_record_indices"]
    match_targets = arrays["match_targets"].astype(bool)
    all_folds = folds(records)
    results: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold_number, fold in enumerate(all_folds):
        train_pairs = fold["train"][pair_records]
        evaluation_pairs = fold["evaluation"][pair_records]
        positive_train = train_pairs & match_targets
        match_model = probe(lock["protocol"]["cValue"], lock["protocol"]["seed"] + fold_number)
        value_model = probe(lock["protocol"]["cValue"], lock["protocol"]["seed"] + fold_number)
        temporal_model = probe(lock["protocol"]["cValue"], lock["protocol"]["seed"] + fold_number)
        match_model.fit(pair_features[train_pairs], match_targets[train_pairs])
        value_model.fit(pair_features[positive_train], arrays["value_targets"][positive_train])
        temporal_model.fit(pair_features[positive_train], arrays["temporal_targets"][positive_train])
        match_scores = np.full(len(pair_features), np.nan, dtype=np.float32)
        predicted_values = np.full(len(pair_features), -1, dtype=np.int8)
        predicted_temporal = np.full(len(pair_features), -1, dtype=np.int8)
        match_scores[evaluation_pairs] = match_model.decision_function(pair_features[evaluation_pairs]).astype(np.float32)
        predicted_values[evaluation_pairs] = value_model.predict(pair_features[evaluation_pairs]).astype(np.int8)
        predicted_temporal[evaluation_pairs] = temporal_model.predict(pair_features[evaluation_pairs]).astype(np.int8)
        overall, _ = evaluate_fold_records(
            records,
            fold["evaluation"],
            pair_records,
            arrays["determinant_indices"],
            arrays["evidence_indices"],
            match_targets,
            match_scores,
            predicted_values,
            predicted_temporal,
        )
        by_surface = {}
        for surface in sorted({record["surface_variant"] for record in records}):
            surface_mask = fold["evaluation"] & np.asarray([
                record["surface_variant"] == surface for record in records
            ])
            by_surface[surface], _ = evaluate_fold_records(
                records,
                surface_mask,
                pair_records,
                arrays["determinant_indices"],
                arrays["evidence_indices"],
                match_targets,
                match_scores,
                predicted_values,
                predicted_temporal,
            )
        artifact_path = output_dir / f"{fold['name'].replace(':', '-')}-heads.npz"
        payload: dict[str, np.ndarray] = {}
        save_pipeline(artifact_path, "match", match_model, payload)
        save_pipeline(artifact_path, "value", value_model, payload)
        save_pipeline(artifact_path, "temporal", temporal_model, payload)
        np.savez_compressed(artifact_path, **payload)
        results[fold["name"]] = {
            "kind": fold["kind"],
            "training_records": int(fold["train"].sum()),
            "evaluation_records": int(fold["evaluation"].sum()),
            "training_pair_examples": int(train_pairs.sum()),
            "head_artifact": str(artifact_path),
            "head_artifact_sha256": file_sha256(artifact_path),
            "overall": overall,
            "by_surface": by_surface,
        }
    gates = gate_report(results, lock["protocol"]["gates"])
    result = {
        "schema_version": 9,
        "revision": 2,
        "experiment": "v9_frozen_neuro_symbolic_grounding",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "model": lock["protocol"]["model"],
        "protocol": lock["protocol"],
        "folds": results,
        "gates": gates,
        "decision": "eligible_for_separate_final_mechanic_protocol" if gates["passed"] else "stop_before_lora_or_final",
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
            "final_v9_mechanic_records_read": 0,
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
