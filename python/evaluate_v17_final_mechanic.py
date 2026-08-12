#!/usr/bin/env python3
"""Fit the frozen V15 deployment heads on development and score V17 once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_v10_frozen import (
    build_pair_lookup, evaluate_ablation, evaluate_cell, gate_report, probe, save_pipeline,
)
from evaluate_v15_full_pipeline import nli_pairs_by_base, unique_current_targets
from v10_protocol import RELATION_ORDER, VALUE_ORDER, file_sha256
from v14_protocol import load_records_from_manifest
from v17_protocol import load_v17_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v17-final-evaluation-lock.json")
    parser.add_argument("--features", default="outputs/v17-final/features")
    parser.add_argument("--output-dir", default="outputs/v17-final/evaluation")
    return parser.parse_args()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def template_result(
    template: str,
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    match_scores: np.ndarray,
    temporal_predictions: np.ndarray,
    relation_predictions: np.ndarray,
) -> dict[str, Any]:
    pair_records = arrays["pair_record_indices"].astype(np.int32)
    determinant_indices = arrays["determinant_indices"].astype(np.int8)
    evidence_indices = arrays["evidence_indices"].astype(np.int8)
    match_targets = arrays["match_targets"].astype(bool)
    pair_lookup = build_pair_lookup(pair_records, determinant_indices)
    mask = np.asarray([record["template_family"] == template for record in records])
    overall = evaluate_cell(
        records, mask, pair_lookup, pair_records, evidence_indices,
        match_targets, match_scores, temporal_predictions, relation_predictions,
    )
    by_surface = {}
    for lexicon in sorted({record["state_lexicon_family"] for record in records}):
        cell = mask & np.asarray([record["state_lexicon_family"] == lexicon for record in records])
        by_surface[lexicon] = evaluate_cell(
            records, cell, pair_lookup, pair_records, evidence_indices,
            match_targets, match_scores, temporal_predictions, relation_predictions,
        )
    group_metrics, _ = evaluate_ablation(
        records, mask, pair_lookup, evidence_indices, match_targets,
        match_scores, temporal_predictions, relation_predictions, False, False,
    )
    return {
        "kind": "final_template",
        "overall": overall,
        "by_surface": by_surface,
        "group_scope": {
            "records": int(mask.sum()),
            "complete_intervention_groups": group_metrics["complete_intervention_groups"],
            "complete_intervention_group_accuracy": group_metrics["complete_intervention_group_accuracy"],
        },
    }


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    feature_root = Path(args.features)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"V17 evaluation directory already exists; refusing a retry: {output_dir}")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["final_mechanic_evaluations_permitted"] != 1 or lock["limits"]["development_linear_fits_permitted"] != 3:
        raise RuntimeError("V17 seal does not authorize the locked one-shot fit/evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"V17 locked implementation changed: {path}")

    final_records = load_v17_records(lock)
    final_metadata_path = feature_root / "metadata.json"
    final_metadata = json.loads(final_metadata_path.read_text())
    if final_metadata["evaluation_lock_sha256"] != file_sha256(lock_path) or final_metadata["final_feature_extraction_number"] != 1:
        raise RuntimeError("V17 final features do not share the sealed evaluation lock")
    final_feature_path = Path(final_metadata["feature_artifact"])
    if file_sha256(final_feature_path) != final_metadata["feature_artifact_sha256"]:
        raise RuntimeError("V17 final feature artifact changed")
    final = load_arrays(final_feature_path)
    if final["record_ids"].tolist() != [record["id"] for record in final_records]:
        raise RuntimeError("V17 final record/feature order differs")

    dev_manifest_path = Path(lock["source"]["v14_manifest"])
    if file_sha256(dev_manifest_path) != lock["source"]["v14_manifest_sha256"]:
        raise RuntimeError("V17 development manifest changed")
    dev_records = load_records_from_manifest(dev_manifest_path)
    dev_metadata_path = Path(lock["source"]["v15_features"])
    if file_sha256(dev_metadata_path) != lock["source"]["v15_features_sha256"]:
        raise RuntimeError("V17 V15 metadata changed")
    dev_metadata = json.loads(dev_metadata_path.read_text())
    dev_feature_path = Path(dev_metadata["feature_artifact"])
    if file_sha256(dev_feature_path) != lock["source"]["v15_feature_artifact_sha256"]:
        raise RuntimeError("V17 V15 development feature artifact changed")
    dev = load_arrays(dev_feature_path)
    if dev["record_ids"].tolist() != [record["id"] for record in dev_records]:
        raise RuntimeError("V17 V15 development record/feature order differs")

    dev_base = dev["base_span_features"].astype(np.float32)
    dev_match = dev["unique_base_match_targets"].astype(bool)
    dev_temporal = dev["unique_base_temporal_targets"].astype(np.int8)
    dev_pair_base = dev["pair_base_indices"].astype(np.int32)
    dev_unique_current = unique_current_targets(
        dev_pair_base, dev["current_value_targets"].astype(np.int8), len(dev_base)
    )
    dev_nli_by_base = nli_pairs_by_base(
        dev_pair_base, dev["pair_nli_indices"].astype(np.int32), len(dev_base)
    )
    dev_nli = dev["nli_hypothesis_mean_features"].astype(np.float32)
    dev_polarity = dev_nli[dev_nli_by_base[:, 0]] - dev_nli[dev_nli_by_base[:, 1]]
    current_train = dev_unique_current >= 0
    positive_train = dev_match

    match_model = probe(lock["c_value"], lock["seed"])
    temporal_model = probe(lock["c_value"], lock["seed"])
    polarity_model = probe(lock["c_value"], lock["seed"])
    match_model.fit(dev_base, dev_match)
    temporal_model.fit(dev_base[positive_train], dev_temporal[positive_train])
    polarity_model.fit(dev_polarity[current_train], dev_unique_current[current_train])

    final_base = final["base_span_features"].astype(np.float32)
    final_pair_base = final["pair_base_indices"].astype(np.int32)
    final_nli_by_base = nli_pairs_by_base(
        final_pair_base, final["pair_nli_indices"].astype(np.int32), len(final_base)
    )
    final_nli = final["nli_hypothesis_mean_features"].astype(np.float32)
    final_polarity = final_nli[final_nli_by_base[:, 0]] - final_nli[final_nli_by_base[:, 1]]
    unique_match_scores = match_model.decision_function(final_base).astype(np.float32)
    unique_temporal = temporal_model.predict(final_base).astype(np.int8)
    unique_current = polarity_model.predict(final_polarity).astype(np.int8)
    match_scores = unique_match_scores[final_pair_base]
    temporal_predictions = unique_temporal[final_pair_base]
    current_predictions = unique_current[final_pair_base]
    relation_predictions = np.asarray([
        [RELATION_ORDER.index("ENTAILED"), RELATION_ORDER.index("CONTRADICTED")]
        if value == VALUE_ORDER.index("active")
        else [RELATION_ORDER.index("CONTRADICTED"), RELATION_ORDER.index("ENTAILED")]
        for value in current_predictions
    ], dtype=np.int8)

    output_dir.mkdir(parents=True, exist_ok=False)
    payload: dict[str, np.ndarray] = {}
    save_pipeline("match", match_model, payload)
    save_pipeline("temporal", temporal_model, payload)
    save_pipeline("polarity", polarity_model, payload)
    head_path = output_dir / "deployment-heads.npz"
    np.savez_compressed(head_path, **payload)

    templates = sorted({record["template_family"] for record in final_records})
    template_results = {
        template: template_result(
            template, final_records, final, match_scores,
            temporal_predictions, relation_predictions,
        )
        for template in templates
    }
    gates = gate_report(template_results, lock["gates"])
    pair_records = final["pair_record_indices"].astype(np.int32)
    determinant_indices = final["determinant_indices"].astype(np.int8)
    evidence_indices = final["evidence_indices"].astype(np.int8)
    match_targets = final["match_targets"].astype(bool)
    pair_lookup = build_pair_lookup(pair_records, determinant_indices)
    overall = evaluate_cell(
        final_records, np.ones(len(final_records), dtype=bool), pair_lookup,
        pair_records, evidence_indices, match_targets, match_scores,
        temporal_predictions, relation_predictions,
    )
    decision = (
        "final_mechanic_generalization_passes"
        if gates["passed"] else "final_mechanic_generalization_fails"
    )
    result = {
        "schema_version": 17,
        "experiment": "v17_one_shot_final_mechanic_evaluation",
        "evaluation_lock": str(lock_path), "evaluation_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "feature_artifact_sha256": final_metadata["feature_artifact_sha256"],
        "development_feature_artifact_sha256": lock["source"]["v15_feature_artifact_sha256"],
        "mechanic": "beacon_console_diagnostic", "operator_family": "multiway_partition",
        "final_evaluation_number": 1, "development_linear_fits": 3,
        "training_final_records": 0,
        "training_unique_base_prompts": len(dev_base),
        "training_unique_positive_prompts": int(positive_train.sum()),
        "training_unique_current_prompts": int(current_train.sum()),
        "head_artifact": str(head_path), "head_artifact_sha256": file_sha256(head_path),
        "overall": overall, "template_folds": template_results,
        "final_gates": gates, "decision": decision,
        "lora_authorized": False, "final_retry_authorized": False,
        "data_access": {
            **final_metadata["data_access"],
            "final_v17_mechanic_records_read": len(final_records),
            "final_v17_model_scores_read": len(final_records),
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": decision, "final_gates": gates, "overall": overall,
        "head_artifact_sha256": result["head_artifact_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
