#!/usr/bin/env python3
"""Evaluate the locked threshold-free V8 ledger-derived ambiguity decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from binary_metrics import evaluate_binary
from run_v8_lomo_diagnostics import direction_metrics, pair_indices
from train_v8_action_conditioned_head import (
    ActionConditionedHead,
    SENSITIVE_INDEX,
    gate_report,
    model_inputs,
    structured_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v8-structured-decision-lock.json")
    parser.add_argument("--output", default="outputs/v8-structured-decision/result.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ledger_scores(status_logits: np.ndarray) -> np.ndarray:
    if status_logits.ndim != 3 or status_logits.shape[-1] != 5:
        raise ValueError("V8 ledger logits must have shape [records, rows, 5]")
    sensitive = status_logits[:, :, SENSITIVE_INDEX]
    nonsensitive = np.delete(status_logits, SENSITIVE_INDEX, axis=-1)
    row_margins = sensitive - np.max(nonsensitive, axis=-1)
    return np.max(row_margins, axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError(f"V8 structured-decision result already exists: {output_path}")
    lock = json.loads(lock_path.read_text())
    config = lock["decision_config"]

    stage4_result_path = Path(lock["stage4_result"]["path"])
    if file_sha256(stage4_result_path) != lock["stage4_result"]["sha256"]:
        raise RuntimeError("V8 Stage 4 result changed after decision lock")
    stage4_result = json.loads(stage4_result_path.read_text())
    component_metadata_path = Path(lock["components"]["metadata"])
    if file_sha256(component_metadata_path) != lock["components"]["metadata_sha256"]:
        raise RuntimeError("V8 component metadata changed after decision lock")
    component_metadata = json.loads(component_metadata_path.read_text())
    component_path = Path(component_metadata["artifact"])
    if file_sha256(component_path) != lock["components"]["artifact_sha256"]:
        raise RuntimeError("V8 component artifact changed after decision lock")
    feature_metadata = json.loads(Path(lock["stage3_features"]["metadata"]).read_text())
    feature_path = Path(feature_metadata["feature_artifact"])
    if file_sha256(feature_path) != lock["stage3_features"]["artifact_sha256"]:
        raise RuntimeError("V8 Stage 3 feature artifact changed after decision lock")

    with np.load(feature_path, allow_pickle=False) as values:
        data = {key: values[key] for key in values.files}
    with np.load(component_path, allow_pickle=False) as values:
        components = {key: values[key] for key in values.files}
    global_features = data["layer_06_mean"].astype(np.float32)
    component_embeddings = components["embeddings"].astype(np.float32)

    folds: dict[str, Any] = {}
    for heldout in lock["mechanics"]:
        saved_fold = stage4_result["folds"][heldout]
        parameter_path = Path(saved_fold["parameter_artifact"])
        if file_sha256(parameter_path) != lock["parameters"][heldout]["sha256"]:
            raise RuntimeError(f"V8 saved head changed for {heldout}")
        model = ActionConditionedHead(global_features.shape[1], lock["projection_width"])
        model.load_weights(str(parameter_path))
        model.eval()

        heldout_indices = np.flatnonzero(data["mechanics"] == heldout)
        inputs = model_inputs(
            heldout_indices,
            global_features,
            component_embeddings,
            components,
        )
        status_logits_mx, _ = model(*inputs)
        mx.eval(status_logits_mx)
        status_logits = np.asarray(status_logits_mx, dtype=np.float32)
        scores = ledger_scores(status_logits)

        by_surface: dict[str, Any] = {}
        heldout_data = {
            key: value[heldout_indices]
            for key, value in data.items()
            if len(value.shape) > 0 and value.shape[0] == len(global_features)
        }
        for surface in lock["surfaces"]:
            local_mask = heldout_data["surface_variants"] == surface
            pairs = pair_indices(heldout_data, local_mask)
            by_surface[surface] = {
                "pointwise": evaluate_binary(
                    heldout_data["gold_ambiguous"][local_mask].astype(bool).tolist(),
                    scores[local_mask].tolist(),
                    config["threshold"],
                ),
                "pair_direction": direction_metrics(
                    pairs,
                    scores,
                    heldout_data["primary_resolved_values"],
                ),
                "structured": structured_metrics(
                    components["status_targets"][heldout_indices][local_mask],
                    status_logits[local_mask],
                ),
            }
        folds[heldout] = {
            "parameter_artifact": str(parameter_path),
            "parameter_artifact_sha256": file_sha256(parameter_path),
            "heldout_records": len(heldout_indices),
            "by_surface": by_surface,
        }
        mx.clear_cache()

    gates = gate_report(folds, config["gates"])
    result = {
        "schema_version": 8,
        "experiment": "v8_ledger_derived_decision_lomo",
        "decision_lock": str(lock_path),
        "decision_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "stage4_result_sha256": lock["stage4_result"]["sha256"],
        "decision_config": config,
        "folds": folds,
        "gates": gates,
        "decision": (
            "eligible_for_new_final_mechanic_protocol"
            if gates["passed"] else "ledger_derived_decision_not_ready"
        ),
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
