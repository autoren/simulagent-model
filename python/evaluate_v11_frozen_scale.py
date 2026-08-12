#!/usr/bin/env python3
"""Run the unchanged V10 evaluation over both locked V11 larger backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_v10_frozen import (
    REPRESENTATIONS,
    build_pair_lookup,
    evaluate_ablation,
    evaluate_cell,
    gate_report,
    group_scope_mask,
    probe,
    save_pipeline,
)
from v10_protocol import RELATION_ORDER, TEMPORAL_ORDER, VALUE_ORDER, file_sha256, folds, load_locked_records
from v9_symbolic import evaluate_allowed_transitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v11-frozen-scale-lock.json")
    parser.add_argument("--features-root", default="outputs/v11-frozen-scale/features")
    parser.add_argument("--output-dir", default="outputs/v11-frozen-scale/evaluation")
    return parser.parse_args()


def oracle_failed(gates: dict[str, Any]) -> bool:
    return any(
        not value["passed"] and ("oracle_polarity" in value["name"] or "nli_pair" in value["name"])
        for value in gates["checks"]
    )


def upstream_failed(gates: dict[str, Any]) -> bool:
    return any(
        not value["passed"] and ("span" in value["name"] or "temporal" in value["name"])
        for value in gates["checks"]
    )


def scale_decision(model_results: dict[str, dict[str, Any]]) -> str:
    four = model_results["qwen35_4b"]["primary_gates"]
    nine = model_results["qwen35_9b"]["primary_gates"]
    if four["passed"] and nine["passed"]:
        return "transferable_polarity_emerges_by_4b_no_lora_authorized"
    if four["passed"] and not nine["passed"]:
        return "nonmonotonic_scale_result_stop_for_inconsistency_audit"
    if not four["passed"] and nine["passed"]:
        return "transferable_polarity_emerges_at_9b_prefer_9b_grounding"
    if oracle_failed(four) and oracle_failed(nine):
        return "frozen_scale_insufficient_test_nonlinear_token_aware_readout"
    if upstream_failed(four) and upstream_failed(nine):
        return "oracle_polarity_available_revise_upstream_grounding"
    return "mixed_gate_failures_stop_before_lora"


def evaluate_model(
    model_key: str,
    model_spec: dict[str, Any],
    lock: dict[str, Any],
    lock_path: Path,
    records: list[dict[str, Any]],
    reference: dict[str, np.ndarray],
    feature_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    model_output = output_dir / model_key
    result_path = model_output / "result.json"
    metadata_path = feature_root / model_key / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError(f"V11 {model_key} features do not share the scale lock")
    if metadata["model"] != model_spec["model"] or metadata["revision"] != model_spec["revision"]:
        raise RuntimeError(f"V11 {model_key} feature identity differs from lock")
    feature_path = Path(metadata["feature_artifact"])
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError(f"V11 {model_key} feature artifact changed")
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if (
            saved["protocol_lock_sha256"] != file_sha256(lock_path)
            or saved["feature_artifact_sha256"] != metadata["feature_artifact_sha256"]
        ):
            raise RuntimeError(f"V11 saved {model_key} evaluation differs from current lock or features")
        return saved
    with np.load(feature_path, allow_pickle=False) as values:
        feature_arrays = {key: values[key] for key in values.files}

    pair_records = reference["pair_record_indices"].astype(np.int32)
    determinant_indices = reference["determinant_indices"].astype(np.int8)
    evidence_indices = reference["evidence_indices"].astype(np.int8)
    match_targets = reference["match_targets"].astype(bool)
    temporal_targets = reference["temporal_targets"].astype(np.int8)
    current_targets = reference["current_value_targets"].astype(np.int8)
    relation_targets = reference["relation_targets"].astype(np.int8)
    pair_base = reference["pair_base_indices"].astype(np.int32)
    pair_nli = reference["pair_nli_indices"].astype(np.int32)
    pair_lookup = build_pair_lookup(pair_records, determinant_indices)
    all_folds = folds(records)
    representation_results: dict[str, Any] = {name: {} for name in REPRESENTATIONS}
    model_output.mkdir(parents=True, exist_ok=True)

    for fold_number, fold in enumerate(all_folds):
        train_pairs = fold["train"][pair_records]
        positive_train = train_pairs & match_targets
        current_train = positive_train & (current_targets >= 0)
        feature_cache: dict[str, tuple[np.ndarray, Any, Any, np.ndarray, np.ndarray]] = {}
        for feature_name in {value["pair_feature"] for value in REPRESENTATIONS.values()}:
            unique_features = feature_arrays[feature_name].astype(np.float32)
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
                nli_features = feature_arrays["nli_final_features"].astype(np.float32)
                train_pair_indices = np.flatnonzero(positive_train)
                train_feature_indices = pair_nli[train_pair_indices].reshape(-1)
                train_relation_targets = relation_targets[train_pair_indices].reshape(-1)
                relation_model = probe(lock["protocol"]["cValue"], lock["protocol"]["seed"] + fold_number)
                relation_model.fit(nli_features[train_feature_indices], train_relation_targets)
                unique_relation_predictions = relation_model.predict(nli_features).astype(np.int8)
                relation_predictions = unique_relation_predictions[pair_nli]
                save_pipeline("relation", relation_model, payload)
            artifact_path = model_output / f"{representation}-{fold['name'].replace(':', '-')}-heads.npz"
            np.savez_compressed(artifact_path, **payload)
            overall = evaluate_cell(
                records, fold["evaluation"], pair_lookup, pair_records, evidence_indices,
                match_targets, match_scores, temporal_predictions, relation_predictions,
            )
            by_surface = {}
            for surface in sorted({record["state_lexicon_family"] for record in records}):
                surface_mask = fold["evaluation"] & np.asarray([
                    record["state_lexicon_family"] == surface for record in records
                ])
                if surface_mask.any():
                    by_surface[surface] = evaluate_cell(
                        records, surface_mask, pair_lookup, pair_records, evidence_indices,
                        match_targets, match_scores, temporal_predictions, relation_predictions,
                    )
            group_mask = group_scope_mask(records, fold)
            group_metrics, _ = evaluate_ablation(
                records, group_mask, pair_lookup, evidence_indices, match_targets, match_scores,
                temporal_predictions, relation_predictions, False, False,
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
    symbolic_mismatches = 0
    for record in records:
        symbolic = evaluate_allowed_transitions(record["action_dependency_schema"], record["target"]["determinant_grounding"])
        if symbolic["identifiable"] != record["target"]["identifiable"] or symbolic["possible_transition_codes"] != record["target"]["possible_transition_codes"]:
            symbolic_mismatches += 1
    if symbolic_mismatches:
        raise RuntimeError(f"V11 {model_key} symbolic audit found {symbolic_mismatches} mismatches")
    result = {
        "schema_version": 11,
        "experiment": "v11_frozen_scale_model_evaluation",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "model_key": model_key,
        "model_spec": model_spec,
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "source_v10_feature_artifact_sha256": lock["source_v10"]["feature_artifact_sha256"],
        "protocol": lock["protocol"],
        "symbolic_audit": {"records": len(records), "mismatches": symbolic_mismatches, "passed": True},
        "representations": representation_results,
        "primary_representation": primary,
        "primary_gates": gates,
        "data_access": lock["data_access"],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    feature_root = Path(args.features_root)
    output_dir = Path(args.output_dir)
    combined_path = output_dir / "result.json"
    if combined_path.exists():
        raise RuntimeError(f"V11 combined result already exists: {combined_path}")
    lock = json.loads(lock_path.read_text())
    v10_lock_path = Path(lock["source_v10"]["frozen_lock"])
    if file_sha256(v10_lock_path) != lock["source_v10"]["frozen_lock_sha256"]:
        raise RuntimeError("V11 source V10 lock changed")
    v10_lock = json.loads(v10_lock_path.read_text())
    records = load_locked_records(v10_lock)
    reference_path = Path(lock["source_v10"]["feature_artifact"])
    if file_sha256(reference_path) != lock["source_v10"]["feature_artifact_sha256"]:
        raise RuntimeError("V11 V10 feature reference changed")
    reference_names = [
        "pair_record_indices", "determinant_indices", "evidence_indices", "match_targets",
        "temporal_targets", "current_value_targets", "relation_targets", "pair_base_indices", "pair_nli_indices",
    ]
    with np.load(reference_path, allow_pickle=False) as values:
        reference = {name: values[name] for name in reference_names}
    model_results = {}
    for model_key in lock["run_order"]:
        model_results[model_key] = evaluate_model(
            model_key, lock["models"][model_key], lock, lock_path, records,
            reference, feature_root, output_dir,
        )
    result = {
        "schema_version": 11,
        "experiment": "v11_frozen_scale_capacity_diagnostic",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "source_v10_result_sha256": lock["source_v10"]["result_sha256"],
        "model_results": {
            key: {
                "result": str(output_dir / key / "result.json"),
                "result_sha256": file_sha256(output_dir / key / "result.json"),
                "primary_gates": value["primary_gates"],
            }
            for key, value in model_results.items()
        },
        "decision": scale_decision(model_results),
        "lora_authorized": False,
        "final_mechanic_authorized": False,
        "data_access": lock["data_access"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
