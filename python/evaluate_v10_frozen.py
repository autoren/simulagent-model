#!/usr/bin/env python3
"""Evaluate the locked V10 frozen representations and neuro-symbolic pipeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from binary_metrics import evaluate_binary
from v10_protocol import (
    RELATION_ORDER,
    TEMPORAL_ORDER,
    VALUE_ORDER,
    derive_allowed_values,
    file_sha256,
    folds,
    load_locked_records,
)
from v9_symbolic import evaluate_allowed_transitions


REPRESENTATIONS = {
    "mean_direct": {"pair_feature": "base_mean_features", "polarity": "direct"},
    "evidence_span_direct": {"pair_feature": "base_span_features", "polarity": "direct"},
    "nli_final": {"pair_feature": "base_span_features", "polarity": "nli"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v10-frozen-lock.json")
    parser.add_argument("--features", default="outputs/v10-frozen/features")
    parser.add_argument("--output-dir", default="outputs/v10-frozen/evaluation")
    return parser.parse_args()


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


def relations_to_current(relations: list[str] | tuple[str, str]) -> str | None:
    if list(relations) == ["ENTAILED", "CONTRADICTED"]:
        return "active"
    if list(relations) == ["CONTRADICTED", "ENTAILED"]:
        return "inactive"
    return None


def build_pair_lookup(
    pair_records: np.ndarray,
    determinant_indices: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for pair_index, (record_index, determinant_index) in enumerate(zip(pair_records, determinant_indices)):
        result.setdefault((int(record_index), int(determinant_index)), []).append(pair_index)
    return result


def selection_metrics(
    records: list[dict[str, Any]],
    mask: np.ndarray,
    pair_lookup: dict[tuple[int, int], list[int]],
    pair_records: np.ndarray,
    evidence_indices: np.ndarray,
    match_targets: np.ndarray,
    match_scores: np.ndarray,
    temporal_predictions: np.ndarray,
) -> dict[str, Any]:
    span_correct: list[bool] = []
    span_f1: list[float] = []
    temporal_oracle_span: list[bool] = []
    temporal_predicted_span: list[bool] = []
    for record_index in np.flatnonzero(mask):
        record = records[int(record_index)]
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            candidates = pair_lookup[(int(record_index), determinant_index)]
            selected = max(candidates, key=lambda index: float(match_scores[index]))
            gold_pair = next(index for index in candidates if bool(match_targets[index]))
            selected_evidence = int(evidence_indices[selected])
            gold_evidence = int(evidence_indices[gold_pair])
            span_correct.append(selected_evidence == gold_evidence)
            span_f1.append(token_f1(
                record["evidence_units"][selected_evidence]["text"],
                record["evidence_units"][gold_evidence]["text"],
            ))
            temporal_oracle_span.append(TEMPORAL_ORDER[int(temporal_predictions[gold_pair])] == target["temporal_status"])
            temporal_predicted_span.append(TEMPORAL_ORDER[int(temporal_predictions[selected])] == target["temporal_status"])
    evaluation_pairs = mask[pair_records]
    return {
        "records": int(mask.sum()),
        "determinants": len(span_correct),
        "span_accuracy": float(np.mean(span_correct)),
        "span_token_f1": float(np.mean(span_f1)),
        "pair_match_auc": float(roc_auc_score(match_targets[evaluation_pairs], match_scores[evaluation_pairs])),
        "temporal_accuracy_oracle_span": float(np.mean(temporal_oracle_span)),
        "temporal_accuracy_predicted_span": float(np.mean(temporal_predicted_span)),
    }


def evaluate_ablation(
    records: list[dict[str, Any]],
    mask: np.ndarray,
    pair_lookup: dict[tuple[int, int], list[int]],
    evidence_indices: np.ndarray,
    match_targets: np.ndarray,
    match_scores: np.ndarray,
    temporal_predictions: np.ndarray,
    relation_predictions: np.ndarray,
    oracle_span: bool,
    oracle_temporal: bool,
) -> tuple[dict[str, Any], dict[int, bool]]:
    polarity_correct: list[bool] = []
    relation_correct: list[bool] = []
    pair_consistent: list[bool] = []
    allowed_correct: list[bool] = []
    temporal_correct: list[bool] = []
    ledger_correct: list[bool] = []
    grounding_correct: list[bool] = []
    possible_correct: list[bool] = []
    gold_ambiguous: list[bool] = []
    predicted_ambiguous: list[bool] = []
    record_predictions: dict[int, bool] = {}
    for record_index in np.flatnonzero(mask):
        record = records[int(record_index)]
        predicted_grounding = []
        record_allowed_exact = True
        record_grounding_exact = True
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            candidates = pair_lookup[(int(record_index), determinant_index)]
            predicted_pair = max(candidates, key=lambda index: float(match_scores[index]))
            gold_pair = next(index for index in candidates if bool(match_targets[index]))
            selected = gold_pair if oracle_span else predicted_pair
            predicted_temporal = target["temporal_status"] if oracle_temporal else TEMPORAL_ORDER[int(temporal_predictions[selected])]
            relations = [RELATION_ORDER[int(value)] for value in relation_predictions[selected]]
            allowed = derive_allowed_values(predicted_temporal, relations)
            allowed_ok = allowed == target["allowed_values"]
            temporal_ok = predicted_temporal == target["temporal_status"]
            span_ok = selected == gold_pair
            allowed_correct.append(allowed_ok)
            temporal_correct.append(temporal_ok)
            record_allowed_exact = record_allowed_exact and allowed_ok
            record_grounding_exact = record_grounding_exact and span_ok and temporal_ok and allowed_ok
            if target["temporal_status"] == "CURRENT":
                polarity_correct.append(relations_to_current(relations) == target["current_value"])
                pair_consistent.append(relations in (["ENTAILED", "CONTRADICTED"], ["CONTRADICTED", "ENTAILED"]))
                relation_correct.extend([
                    relations[index] == target["hypothesis_relations"][index] for index in range(2)
                ])
            predicted_grounding.append({"determinant_id": target["determinant_id"], "allowed_values": allowed})
        symbolic = evaluate_allowed_transitions(record["action_dependency_schema"], predicted_grounding)
        ambiguous = not symbolic["identifiable"]
        record_predictions[int(record_index)] = ambiguous
        gold_ambiguous.append(not record["target"]["identifiable"])
        predicted_ambiguous.append(ambiguous)
        possible_correct.append(symbolic["possible_transition_codes"] == record["target"]["possible_transition_codes"])
        ledger_correct.append(record_allowed_exact)
        grounding_correct.append(record_grounding_exact)

    flip_pairs: list[bool] = []
    pair_groups: dict[tuple[str, str], list[int]] = {}
    intervention_groups: dict[str, list[int]] = {}
    for record_index in np.flatnonzero(mask):
        record = records[int(record_index)]
        intervention_groups.setdefault(record["intervention_group_id"], []).append(int(record_index))
        if record["intervention_kind"] == "oracle_label_flip":
            pair_groups.setdefault((record["intervention_group_id"], record["state_lexicon_family"]), []).append(int(record_index))
    for indices in pair_groups.values():
        if len(indices) != 2:
            continue
        flip_pairs.append(all(record_predictions[index] == (not records[index]["target"]["identifiable"]) for index in indices))
    complete_groups = [
        all(record_predictions[index] == (not records[index]["target"]["identifiable"]) for index in indices)
        for indices in intervention_groups.values() if len(indices) == 6
    ]
    symbolic_metrics = evaluate_binary(gold_ambiguous, [float(value) for value in predicted_ambiguous], 0.5)
    return {
        "polarity_examples": len(polarity_correct),
        "polarity_accuracy": float(np.mean(polarity_correct)),
        "current_hypothesis_relation_accuracy": float(np.mean(relation_correct)),
        "hypothesis_pair_consistency": float(np.mean(pair_consistent)),
        "allowed_values_accuracy": float(np.mean(allowed_correct)),
        "temporal_accuracy": float(np.mean(temporal_correct)),
        "complete_allowed_ledger_accuracy": float(np.mean(ledger_correct)),
        "complete_grounding_accuracy": float(np.mean(grounding_correct)),
        "symbolic_identifiability": symbolic_metrics,
        "possible_transition_set_exact_accuracy": float(np.mean(possible_correct)),
        "label_flip_pairs": len(flip_pairs),
        "complete_flip_pair_accuracy": float(np.mean(flip_pairs)) if flip_pairs else None,
        "complete_intervention_groups": len(complete_groups),
        "complete_intervention_group_accuracy": float(np.mean(complete_groups)) if complete_groups else None,
    }, record_predictions


def group_scope_mask(records: list[dict[str, Any]], fold: dict[str, Any]) -> np.ndarray:
    if fold["kind"] in {"context", "mechanic", "template"}:
        return fold["evaluation"].copy()
    if fold["kind"] == "lexicon":
        return np.asarray([record["split"] == "evaluation" for record in records])
    held_operator = fold["name"].split(":")[1]
    return np.asarray([record["operator_family"] == held_operator for record in records])


def evaluate_cell(
    records: list[dict[str, Any]],
    mask: np.ndarray,
    pair_lookup: dict[tuple[int, int], list[int]],
    pair_records: np.ndarray,
    evidence_indices: np.ndarray,
    match_targets: np.ndarray,
    match_scores: np.ndarray,
    temporal_predictions: np.ndarray,
    relation_predictions: np.ndarray,
) -> dict[str, Any]:
    result = selection_metrics(
        records, mask, pair_lookup, pair_records, evidence_indices, match_targets, match_scores, temporal_predictions,
    )
    ablations = {}
    for name, oracle_span, oracle_temporal in (
        ("oracle_span_oracle_temporal", True, True),
        ("predicted_span_oracle_temporal", False, True),
        ("oracle_span_predicted_temporal", True, False),
        ("fully_predicted", False, False),
    ):
        ablations[name], _ = evaluate_ablation(
            records,
            mask,
            pair_lookup,
            evidence_indices,
            match_targets,
            match_scores,
            temporal_predictions,
            relation_predictions,
            oracle_span,
            oracle_temporal,
        )
    result["ablations"] = ablations
    return result


def save_pipeline(prefix: str, model: Any, payload: dict[str, np.ndarray]) -> None:
    scaler = model.named_steps["standardscaler"]
    head = model.named_steps["logisticregression"]
    payload[f"{prefix}_scaler_mean"] = scaler.mean_
    payload[f"{prefix}_scaler_scale"] = scaler.scale_
    payload[f"{prefix}_coef"] = head.coef_
    payload[f"{prefix}_intercept"] = head.intercept_
    payload[f"{prefix}_classes"] = head.classes_


def gate_report(fold_results: dict[str, dict[str, Any]], gates: dict[str, float]) -> dict[str, Any]:
    overall = [value["overall"] for value in fold_results.values()]
    surfaces = [cell for value in fold_results.values() for cell in value["by_surface"].values()]
    checks = [
        ("minimum_fold_span_accuracy", min(value["span_accuracy"] for value in overall), gates["minimumEveryFoldSpanAccuracy"]),
        ("minimum_surface_span_accuracy", min(value["span_accuracy"] for value in surfaces), gates["minimumEverySurfaceSpanAccuracy"]),
        ("minimum_fold_temporal_accuracy", min(value["temporal_accuracy_predicted_span"] for value in overall), gates["minimumEveryFoldTemporalAccuracy"]),
        ("minimum_surface_temporal_accuracy", min(value["temporal_accuracy_predicted_span"] for value in surfaces), gates["minimumEverySurfaceTemporalAccuracy"]),
        ("minimum_fold_oracle_polarity_accuracy", min(value["ablations"]["oracle_span_oracle_temporal"]["polarity_accuracy"] for value in overall), gates["minimumEveryFoldOraclePolarityAccuracy"]),
        ("minimum_surface_oracle_polarity_accuracy", min(value["ablations"]["oracle_span_oracle_temporal"]["polarity_accuracy"] for value in surfaces), gates["minimumEverySurfaceOraclePolarityAccuracy"]),
        ("minimum_fold_nli_pair_consistency", min(value["ablations"]["oracle_span_oracle_temporal"]["hypothesis_pair_consistency"] for value in overall), gates["minimumEveryFoldNliPairConsistency"]),
        ("minimum_surface_nli_pair_consistency", min(value["ablations"]["oracle_span_oracle_temporal"]["hypothesis_pair_consistency"] for value in surfaces), gates["minimumEverySurfaceNliPairConsistency"]),
        ("minimum_fold_allowed_values_accuracy", min(value["ablations"]["fully_predicted"]["allowed_values_accuracy"] for value in overall), gates["minimumEveryFoldAllowedValuesAccuracy"]),
        ("minimum_surface_allowed_values_accuracy", min(value["ablations"]["fully_predicted"]["allowed_values_accuracy"] for value in surfaces), gates["minimumEverySurfaceAllowedValuesAccuracy"]),
        ("minimum_fold_symbolic_balanced_accuracy", min(value["ablations"]["fully_predicted"]["symbolic_identifiability"]["balanced_accuracy"] for value in overall), gates["minimumEveryFoldSymbolicBalancedAccuracy"]),
        ("minimum_surface_symbolic_balanced_accuracy", min(value["ablations"]["fully_predicted"]["symbolic_identifiability"]["balanced_accuracy"] for value in surfaces), gates["minimumEverySurfaceSymbolicBalancedAccuracy"]),
        ("minimum_fold_complete_flip_pair_accuracy", min(value["ablations"]["fully_predicted"]["complete_flip_pair_accuracy"] for value in overall), gates["minimumEveryFoldCompleteFlipPairAccuracy"]),
        ("minimum_fold_complete_intervention_group_accuracy", min(value["group_scope"]["complete_intervention_group_accuracy"] for value in fold_results.values()), gates["minimumEveryFoldCompleteInterventionGroupAccuracy"]),
    ]
    values = [
        {"name": name, "value": float(value), "minimum": minimum, "passed": value >= minimum}
        for name, value, minimum in checks
    ]
    return {"passed": all(value["passed"] for value in values), "checks": values}


def decision_from_gates(report: dict[str, Any]) -> str:
    if report["passed"]:
        return "stop_at_frozen_0.8b_success"
    failures = {value["name"] for value in report["checks"] if not value["passed"]}
    if any("oracle_polarity" in value or "nli_pair" in value for value in failures):
        return "authorize_separately_locked_larger_frozen_capacity_diagnostic"
    if any("span" in value or "temporal" in value for value in failures):
        return "revise_upstream_grounding_only"
    return "revise_neuro_symbolic_composition_without_lora"


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    feature_root = Path(args.features)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"V10 result already exists: {result_path}")
    lock = json.loads(lock_path.read_text())
    records = load_locked_records(lock)
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V10 features do not share the frozen lock")
    feature_path = Path(metadata["feature_artifact"])
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V10 feature artifact changed")
    with np.load(feature_path, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    if arrays["record_ids"].tolist() != [record["id"] for record in records]:
        raise RuntimeError("V10 feature and record order differ")

    symbolic_mismatches = 0
    for record in records:
        symbolic = evaluate_allowed_transitions(record["action_dependency_schema"], record["target"]["determinant_grounding"])
        if symbolic["identifiable"] != record["target"]["identifiable"] or symbolic["possible_transition_codes"] != record["target"]["possible_transition_codes"]:
            symbolic_mismatches += 1
    if symbolic_mismatches:
        raise RuntimeError(f"V10 Python symbolic audit found {symbolic_mismatches} mismatches")

    pair_records = arrays["pair_record_indices"].astype(np.int32)
    determinant_indices = arrays["determinant_indices"].astype(np.int8)
    evidence_indices = arrays["evidence_indices"].astype(np.int8)
    match_targets = arrays["match_targets"].astype(bool)
    temporal_targets = arrays["temporal_targets"].astype(np.int8)
    current_targets = arrays["current_value_targets"].astype(np.int8)
    relation_targets = arrays["relation_targets"].astype(np.int8)
    pair_base = arrays["pair_base_indices"].astype(np.int32)
    pair_nli = arrays["pair_nli_indices"].astype(np.int32)
    pair_lookup = build_pair_lookup(pair_records, determinant_indices)
    all_folds = folds(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    representation_results: dict[str, Any] = {name: {} for name in REPRESENTATIONS}

    for fold_number, fold in enumerate(all_folds):
        train_pairs = fold["train"][pair_records]
        positive_train = train_pairs & match_targets
        current_train = positive_train & (current_targets >= 0)
        feature_cache: dict[str, tuple[np.ndarray, Any, Any, np.ndarray, np.ndarray]] = {}
        for feature_name in {value["pair_feature"] for value in REPRESENTATIONS.values()}:
            unique_features = arrays[feature_name].astype(np.float32)
            pair_features = unique_features[pair_base]
            match_model = probe(lock["protocol"]["cValue"], lock["protocol"]["seed"] + fold_number)
            temporal_model = probe(lock["protocol"]["cValue"], lock["protocol"]["seed"] + fold_number)
            match_model.fit(pair_features[train_pairs], match_targets[train_pairs])
            temporal_model.fit(pair_features[positive_train], temporal_targets[positive_train])
            match_scores = match_model.decision_function(pair_features).astype(np.float32)
            temporal_predictions = temporal_model.predict(pair_features).astype(np.int8)
            feature_cache[feature_name] = (pair_features, match_model, temporal_model, match_scores, temporal_predictions)

        for representation, specification in REPRESENTATIONS.items():
            pair_features, match_model, temporal_model, match_scores, temporal_predictions = feature_cache[specification["pair_feature"]]
            payload: dict[str, np.ndarray] = {}
            save_pipeline("match", match_model, payload)
            save_pipeline("temporal", temporal_model, payload)
            if specification["polarity"] == "direct":
                polarity_model = probe(lock["protocol"]["cValue"], lock["protocol"]["seed"] + fold_number)
                polarity_model.fit(pair_features[current_train], current_targets[current_train])
                current_predictions = polarity_model.predict(pair_features).astype(np.int8)
                relation_predictions = np.asarray([
                    [RELATION_ORDER.index("ENTAILED"), RELATION_ORDER.index("CONTRADICTED")]
                    if value == VALUE_ORDER.index("active")
                    else [RELATION_ORDER.index("CONTRADICTED"), RELATION_ORDER.index("ENTAILED")]
                    for value in current_predictions
                ], dtype=np.int8)
                save_pipeline("polarity", polarity_model, payload)
            else:
                nli_features = arrays["nli_final_features"].astype(np.float32)
                train_pair_indices = np.flatnonzero(positive_train)
                train_feature_indices = pair_nli[train_pair_indices].reshape(-1)
                train_relation_targets = relation_targets[train_pair_indices].reshape(-1)
                relation_model = probe(lock["protocol"]["cValue"], lock["protocol"]["seed"] + fold_number)
                relation_model.fit(nli_features[train_feature_indices], train_relation_targets)
                unique_relation_predictions = relation_model.predict(nli_features).astype(np.int8)
                relation_predictions = unique_relation_predictions[pair_nli]
                save_pipeline("relation", relation_model, payload)
            fold_slug = fold["name"].replace(":", "-")
            artifact_path = output_dir / f"{representation}-{fold_slug}-heads.npz"
            np.savez_compressed(artifact_path, **payload)
            overall = evaluate_cell(
                records,
                fold["evaluation"],
                pair_lookup,
                pair_records,
                evidence_indices,
                match_targets,
                match_scores,
                temporal_predictions,
                relation_predictions,
            )
            by_surface = {}
            for surface in sorted({record["state_lexicon_family"] for record in records}):
                surface_mask = fold["evaluation"] & np.asarray([
                    record["state_lexicon_family"] == surface for record in records
                ])
                if surface_mask.any():
                    by_surface[surface] = evaluate_cell(
                        records,
                        surface_mask,
                        pair_lookup,
                        pair_records,
                        evidence_indices,
                        match_targets,
                        match_scores,
                        temporal_predictions,
                        relation_predictions,
                    )
            group_mask = group_scope_mask(records, fold)
            group_metrics, _ = evaluate_ablation(
                records,
                group_mask,
                pair_lookup,
                evidence_indices,
                match_targets,
                match_scores,
                temporal_predictions,
                relation_predictions,
                False,
                False,
            )
            representation_results[representation][fold["name"]] = {
                "kind": fold["kind"],
                "training_records": int(fold["train"].sum()),
                "evaluation_records": int(fold["evaluation"].sum()),
                "training_pair_examples": int(train_pairs.sum()),
                "head_artifact": str(artifact_path),
                "head_artifact_sha256": file_sha256(artifact_path),
                "overall": overall,
                "by_surface": by_surface,
                "group_scope": {
                    "records": int(group_mask.sum()),
                    "complete_intervention_groups": group_metrics["complete_intervention_groups"],
                    "complete_intervention_group_accuracy": group_metrics["complete_intervention_group_accuracy"],
                },
            }

    primary = lock["protocol"]["primaryRepresentation"]
    gates = gate_report(representation_results[primary], lock["protocol"]["gates"])
    decision = decision_from_gates(gates)
    result = {
        "schema_version": 10,
        "experiment": "v10_frozen_current_state_polarity",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "model": lock["protocol"]["model"],
        "protocol": lock["protocol"],
        "symbolic_audit": {"records": len(records), "mismatches": symbolic_mismatches, "passed": symbolic_mismatches == 0},
        "representations": representation_results,
        "primary_representation": primary,
        "primary_gates": gates,
        "decision": decision,
        "lora_authorized": False,
        "larger_frozen_model_run_in_v10": False,
        "data_access": lock["data_access"],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
