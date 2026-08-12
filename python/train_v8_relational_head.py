#!/usr/bin/env python3
"""Train/evaluate the locked normalized V8 query-conditioned relational head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from binary_metrics import evaluate_binary
from run_v8_lomo_diagnostics import direction_metrics, pair_indices
from train_v8_action_conditioned_head import (
    STATUS_ORDER,
    SENSITIVE_INDEX,
    class_weights,
    gate_report,
    local_pairs,
    local_surface_groups,
    model_inputs,
    structured_metrics,
)


class QueryConditionedRelationalHead(nn.Module):
    def __init__(self, input_dims: int, width: int):
        super().__init__()
        self.global_projection = nn.Linear(input_dims, width)
        self.action_projection = nn.Linear(input_dims, width)
        self.table_projection = nn.Linear(input_dims, width)
        self.role_projection = nn.Linear(input_dims, width)
        self.evidence_projection = nn.Linear(input_dims, width)
        self.query_norm = nn.LayerNorm(width)
        self.role_norm = nn.LayerNorm(width)
        self.evidence_norm = nn.LayerNorm(width)
        self.relation_projection = nn.Linear(width * 6, width)
        self.status_head = nn.Linear(width, len(STATUS_ORDER))
        self.sensitivity_head = nn.Linear(width, 1)

    def __call__(
        self,
        global_features: mx.array,
        action_features: mx.array,
        table_features: mx.array,
        role_features: mx.array,
        evidence_features: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        query = self.query_norm(
            self.global_projection(global_features)
            + self.action_projection(action_features)
            + self.table_projection(table_features)
        )
        roles = self.role_norm(self.role_projection(role_features))
        evidence = self.evidence_norm(self.evidence_projection(evidence_features))
        expanded_query = mx.broadcast_to(query[:, None, :], roles.shape)
        relations = mx.concatenate(
            [
                expanded_query,
                roles,
                evidence,
                expanded_query * roles,
                expanded_query * evidence,
                roles * evidence,
            ],
            axis=-1,
        )
        row_hidden = nn.gelu(self.relation_projection(relations))
        status_logits = self.status_head(row_hidden)
        row_sensitivity_logits = mx.squeeze(self.sensitivity_head(row_hidden), axis=-1)
        ambiguity_logits = mx.max(row_sensitivity_logits, axis=1)
        return status_logits, row_sensitivity_logits, ambiguity_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v8-relational-head-lock.json")
    parser.add_argument("--output-dir", default="outputs/v8-relational-head/lomo")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_loss(
    config: dict[str, Any],
    record_targets: mx.array,
    status_targets: mx.array,
    sensitivity_targets: mx.array,
    status_weights: mx.array,
    sensitivity_weights: mx.array,
    record_weights: mx.array,
    pairs: mx.array,
    surface_groups: mx.array,
) -> Any:
    weights = config["lossWeights"]

    def loss_fn(
        model: QueryConditionedRelationalHead,
        *inputs: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        status_logits, row_logits, ambiguity_logits = model(*inputs)
        determinant_losses = nn.losses.cross_entropy(
            status_logits.reshape(-1, len(STATUS_ORDER)),
            status_targets.reshape(-1),
            reduction="none",
        )
        determinant_loss = mx.mean(determinant_losses * status_weights.reshape(-1))
        row_losses = nn.losses.binary_cross_entropy(
            row_logits,
            sensitivity_targets,
            with_logits=True,
            reduction="none",
        )
        row_sensitivity_loss = mx.mean(row_losses * sensitivity_weights)
        point_losses = nn.losses.binary_cross_entropy(
            ambiguity_logits,
            record_targets,
            with_logits=True,
            reduction="none",
        )
        pointwise_loss = mx.mean(point_losses * record_weights)
        pair_margins = ambiguity_logits[pairs[:, 0]] - ambiguity_logits[pairs[:, 1]]
        pairwise_loss = mx.mean(mx.logaddexp(mx.zeros_like(pair_margins), -pair_margins))
        surface_loss = mx.mean(mx.var(ambiguity_logits[surface_groups], axis=1))
        total = (
            weights["determinant"] * determinant_loss
            + weights["rowSensitivity"] * row_sensitivity_loss
            + weights["pointwise"] * pointwise_loss
            + weights["pairwise"] * pairwise_loss
            + weights["surface"] * surface_loss
        )
        return total, {
            "determinant": determinant_loss,
            "row_sensitivity": row_sensitivity_loss,
            "pointwise": pointwise_loss,
            "pairwise": pairwise_loss,
            "surface": surface_loss,
        }

    return loss_fn


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"V8 relational-head result already exists: {result_path}")
    lock = json.loads(lock_path.read_text())
    config = lock["head_config"]

    feature_metadata = json.loads(Path(lock["stage3_features"]["metadata"]).read_text())
    feature_path = Path(feature_metadata["feature_artifact"])
    if file_sha256(feature_path) != lock["stage3_features"]["artifact_sha256"]:
        raise RuntimeError("V8 Stage 3 features changed after relational-head lock")
    component_metadata = json.loads(Path(lock["components"]["metadata"]).read_text())
    component_path = Path(component_metadata["artifact"])
    if file_sha256(component_path) != lock["components"]["artifact_sha256"]:
        raise RuntimeError("V8 components changed after relational-head lock")
    with np.load(feature_path, allow_pickle=False) as values:
        data = {key: values[key] for key in values.files}
    with np.load(component_path, allow_pickle=False) as values:
        components = {key: values[key] for key in values.files}
    global_features = data["layer_06_mean"].astype(np.float32)
    component_embeddings = components["embeddings"].astype(np.float32)

    folds: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold_number, heldout in enumerate(lock["mechanics"]):
        mx.random.seed(config["seed"] + fold_number)
        train_mask = (data["mechanics"] != heldout) & (data["splits"] == "train")
        train_indices = np.flatnonzero(train_mask)
        inputs = model_inputs(train_indices, global_features, component_embeddings, components)
        record_gold_np = data["gold_ambiguous"][train_indices].astype(np.float32)
        status_gold_np = components["status_targets"][train_indices].astype(np.int32)
        sensitivity_gold_np = (status_gold_np == SENSITIVE_INDEX).astype(np.float32)
        status_class_weights = class_weights(status_gold_np, len(STATUS_ORDER))
        sensitivity_class_weights = class_weights(sensitivity_gold_np.astype(np.int32), 2)
        record_class_weights = class_weights(record_gold_np.astype(np.int32), 2)
        status_weights = mx.array(status_class_weights[status_gold_np])
        sensitivity_weights = mx.array(
            sensitivity_class_weights[sensitivity_gold_np.astype(np.int32)]
        )
        record_weights = mx.array(record_class_weights[record_gold_np.astype(np.int32)])
        record_gold = mx.array(record_gold_np)
        status_gold = mx.array(status_gold_np)
        sensitivity_gold = mx.array(sensitivity_gold_np)
        pairs_np = local_pairs(data, train_mask, train_indices)
        surface_groups_np = local_surface_groups(data, train_indices)
        pairs = mx.array(pairs_np)
        surface_groups = mx.array(surface_groups_np)

        model = QueryConditionedRelationalHead(
            global_features.shape[1], config["projectionWidth"]
        )
        optimizer = optim.Adam(learning_rate=config["learningRate"])
        loss_fn = make_loss(
            config,
            record_gold,
            status_gold,
            sensitivity_gold,
            status_weights,
            sensitivity_weights,
            record_weights,
            pairs,
            surface_groups,
        )
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        final_losses: dict[str, float] = {}
        for _ in range(config["trainingSteps"]):
            (loss, parts), gradients = loss_and_grad(model, *inputs)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss, parts)
            final_losses = {
                "total": float(loss),
                **{name: float(value) for name, value in parts.items()},
            }

        all_indices = np.arange(len(global_features))
        all_inputs = model_inputs(all_indices, global_features, component_embeddings, components)
        status_logits_mx, row_logits_mx, ambiguity_logits_mx = model(*all_inputs)
        mx.eval(status_logits_mx, row_logits_mx, ambiguity_logits_mx)
        status_logits = np.asarray(status_logits_mx, dtype=np.float32)
        row_logits = np.asarray(row_logits_mx, dtype=np.float32)
        ambiguity_logits = np.asarray(ambiguity_logits_mx, dtype=np.float32)

        by_surface: dict[str, Any] = {}
        heldout_mask = data["mechanics"] == heldout
        for surface in lock["surfaces"]:
            surface_mask = heldout_mask & (data["surface_variants"] == surface)
            surface_pairs = pair_indices(data, surface_mask)
            by_surface[surface] = {
                "pointwise": evaluate_binary(
                    data["gold_ambiguous"][surface_mask].astype(bool).tolist(),
                    ambiguity_logits[surface_mask].tolist(),
                    0.0,
                ),
                "pair_direction": direction_metrics(
                    surface_pairs,
                    ambiguity_logits,
                    data["primary_resolved_values"],
                ),
                "structured": structured_metrics(
                    components["status_targets"][surface_mask],
                    status_logits[surface_mask],
                ),
                "row_sensitivity_accuracy": float(np.mean(
                    (row_logits[surface_mask] > 0)
                    == (components["status_targets"][surface_mask] == SENSITIVE_INDEX)
                )),
            }
        parameter_path = output_dir / f"{heldout}-head.npz"
        np.savez_compressed(
            parameter_path,
            **{name: np.asarray(value) for name, value in tree_flatten(model.parameters())},
        )
        folds[heldout] = {
            "training_records": len(train_indices),
            "training_pairs": len(pairs_np),
            "surface_groups": len(surface_groups_np),
            "final_losses": final_losses,
            "parameter_artifact": str(parameter_path),
            "parameter_artifact_sha256": file_sha256(parameter_path),
            "by_surface": by_surface,
        }
        mx.clear_cache()

    gates = gate_report(folds, config["gates"])
    result = {
        "schema_version": 8,
        "experiment": "v8_query_conditioned_relational_head_lomo",
        "relational_head_lock": str(lock_path),
        "relational_head_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "component_artifact_sha256": lock["components"]["artifact_sha256"],
        "head_config": config,
        "folds": folds,
        "gates": gates,
        "decision": (
            "eligible_for_new_final_mechanic_protocol"
            if gates["passed"] else "relational_head_not_ready_stop_v8_development"
        ),
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
