#!/usr/bin/env python3
"""Train/evaluate the locked V8 action-conditioned structured head in LOMO folds."""

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

from binary_metrics import evaluate_binary, fit_threshold
from run_v8_lomo_diagnostics import direction_metrics, pair_indices


STATUS_ORDER = [
    "RESOLVED_TRUE",
    "RESOLVED_FALSE",
    "UNRESOLVED_OUTCOME_SENSITIVE",
    "UNRESOLVED_OUTCOME_INVARIANT",
    "IRRELEVANT",
]
SENSITIVE_INDEX = STATUS_ORDER.index("UNRESOLVED_OUTCOME_SENSITIVE")


class ActionConditionedHead(nn.Module):
    def __init__(self, input_dims: int, width: int):
        super().__init__()
        self.global_projection = nn.Linear(input_dims, width)
        self.action_projection = nn.Linear(input_dims, width)
        self.table_projection = nn.Linear(input_dims, width)
        self.role_projection = nn.Linear(input_dims, width)
        self.evidence_projection = nn.Linear(input_dims, width)
        self.status_head = nn.Linear(width, len(STATUS_ORDER))
        self.ambiguity_head = nn.Linear(width * 5, 1)

    def __call__(
        self,
        global_features: mx.array,
        action_features: mx.array,
        table_features: mx.array,
        role_features: mx.array,
        evidence_features: mx.array,
    ) -> tuple[mx.array, mx.array]:
        global_hidden = nn.gelu(self.global_projection(global_features))
        action_hidden = nn.gelu(self.action_projection(action_features))
        table_hidden = nn.gelu(self.table_projection(table_features))
        row_hidden = nn.gelu(
            global_hidden[:, None, :]
            + action_hidden[:, None, :]
            + table_hidden[:, None, :]
            + self.role_projection(role_features)
            + self.evidence_projection(evidence_features)
        )
        status_logits = self.status_head(row_hidden)
        row_mean = mx.mean(row_hidden, axis=1)
        row_max = mx.max(row_hidden, axis=1)
        record_hidden = mx.concatenate(
            [global_hidden, action_hidden, table_hidden, row_mean, row_max], axis=1
        )
        pointwise = mx.squeeze(self.ambiguity_head(record_hidden), axis=-1)
        sensitive_evidence = mx.logsumexp(status_logits[:, :, SENSITIVE_INDEX], axis=1)
        ambiguity_logits = pointwise + 0.25 * sensitive_evidence
        return status_logits, ambiguity_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v8-structured-head-lock.json")
    parser.add_argument("--components", default="outputs/v8-structured-head/components")
    parser.add_argument("--output-dir", default="outputs/v8-structured-head/lomo")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_weights(targets: np.ndarray, classes: int) -> np.ndarray:
    counts = np.bincount(targets.reshape(-1), minlength=classes).astype(np.float32)
    weights = counts.sum() / (classes * np.maximum(counts, 1.0))
    return weights.astype(np.float32)


def local_pairs(data: dict[str, np.ndarray], mask: np.ndarray, indices: np.ndarray) -> np.ndarray:
    lookup = {int(global_index): local for local, global_index in enumerate(indices)}
    return np.asarray([
        [lookup[ambiguous], lookup[identifiable]]
        for ambiguous, identifiable in pair_indices(data, mask)
    ], dtype=np.int32)


def local_surface_groups(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for local, global_index in enumerate(indices):
        groups.setdefault(str(data["surface_group_ids"][global_index]), []).append(local)
    triples = [sorted(values) for values in groups.values()]
    if any(len(values) != 3 for values in triples):
        raise RuntimeError("V8 training surface group is incomplete")
    return np.asarray(sorted(triples), dtype=np.int32)


def model_inputs(
    indices: np.ndarray,
    global_features: np.ndarray,
    component_embeddings: np.ndarray,
    components: dict[str, np.ndarray],
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    return (
        mx.array(global_features[indices]),
        mx.array(component_embeddings[components["action_indices"][indices]]),
        mx.array(component_embeddings[components["table_indices"][indices]]),
        mx.array(component_embeddings[components["role_indices"][indices]]),
        mx.array(component_embeddings[components["evidence_indices"][indices]]),
    )


def make_loss(
    config: dict[str, Any],
    record_targets: mx.array,
    status_targets: mx.array,
    status_weights: mx.array,
    record_weights: mx.array,
    pairs: mx.array,
    surface_groups: mx.array,
) -> Any:
    weights = config["lossWeights"]

    def loss_fn(model: ActionConditionedHead, *inputs: mx.array) -> tuple[mx.array, dict[str, mx.array]]:
        status_logits, ambiguity_logits = model(*inputs)
        determinant_losses = nn.losses.cross_entropy(
            status_logits.reshape(-1, len(STATUS_ORDER)),
            status_targets.reshape(-1),
            reduction="none",
        )
        determinant_loss = mx.mean(determinant_losses * status_weights.reshape(-1))
        point_losses = nn.losses.binary_cross_entropy(
            ambiguity_logits,
            record_targets,
            with_logits=True,
            reduction="none",
        )
        pointwise_loss = mx.mean(point_losses * record_weights)
        pair_margins = ambiguity_logits[pairs[:, 0]] - ambiguity_logits[pairs[:, 1]]
        pairwise_loss = mx.mean(mx.logaddexp(mx.zeros_like(pair_margins), -pair_margins))
        surface_scores = ambiguity_logits[surface_groups]
        surface_loss = mx.mean(mx.var(surface_scores, axis=1))
        total = (
            weights["determinant"] * determinant_loss
            + weights["pointwise"] * pointwise_loss
            + weights["pairwise"] * pairwise_loss
            + weights["surface"] * surface_loss
        )
        return total, {
            "determinant": determinant_loss,
            "pointwise": pointwise_loss,
            "pairwise": pairwise_loss,
            "surface": surface_loss,
        }

    return loss_fn


def macro_f1(gold: np.ndarray, predicted: np.ndarray, classes: int) -> float:
    values = []
    for class_index in range(classes):
        tp = int(np.sum((gold == class_index) & (predicted == class_index)))
        fp = int(np.sum((gold != class_index) & (predicted == class_index)))
        fn = int(np.sum((gold == class_index) & (predicted != class_index)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def structured_metrics(gold: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    predicted = np.argmax(logits, axis=-1)
    ambiguous_records = np.flatnonzero(np.any(gold == SENSITIVE_INDEX, axis=1))
    decisive_correct = 0
    for index in ambiguous_records:
        gold_row = int(np.flatnonzero(gold[index] == SENSITIVE_INDEX)[0])
        predicted_row = int(np.argmax(logits[index, :, SENSITIVE_INDEX]))
        decisive_correct += gold_row == predicted_row
    return {
        "rows": int(gold.size),
        "status_macro_f1": macro_f1(gold.reshape(-1), predicted.reshape(-1), len(STATUS_ORDER)),
        "status_accuracy": float(np.mean(gold == predicted)),
        "exact_ledger_accuracy": float(np.mean(np.all(gold == predicted, axis=1))),
        "decisive_records": len(ambiguous_records),
        "decisive_determinant_accuracy": decisive_correct / len(ambiguous_records),
    }


def gate_report(folds: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    balanced = [
        fold["by_surface"][surface]["pointwise"]["balanced_accuracy"]
        for fold in folds.values() for surface in fold["by_surface"]
    ]
    direction = [
        fold["by_surface"][surface]["pair_direction"]["accuracy"]
        for fold in folds.values() for surface in fold["by_surface"]
    ]
    status = [
        fold["by_surface"][surface]["structured"]["status_macro_f1"]
        for fold in folds.values() for surface in fold["by_surface"]
    ]
    decisive = [
        fold["by_surface"][surface]["structured"]["decisive_determinant_accuracy"]
        for fold in folds.values() for surface in fold["by_surface"]
    ]
    checks = [
        ("minimum_fold_surface_balanced_accuracy", min(balanced), gates["minimumEveryFoldSurfaceBalancedAccuracy"]),
        ("mean_fold_surface_balanced_accuracy", float(np.mean(balanced)), gates["minimumMeanFoldSurfaceBalancedAccuracy"]),
        ("minimum_fold_surface_pair_direction", min(direction), gates["minimumEveryFoldSurfacePairDirection"]),
        ("minimum_fold_surface_status_macro_f1", min(status), gates["minimumEveryFoldSurfaceStatusMacroF1"]),
        ("minimum_fold_surface_decisive_determinant_accuracy", min(decisive), gates["minimumEveryFoldSurfaceDecisiveDeterminantAccuracy"]),
    ]
    result = [
        {"name": name, "value": value, "minimum": minimum, "passed": value >= minimum}
        for name, value, minimum in checks
    ]
    return {"passed": all(check["passed"] for check in result), "checks": result}


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    component_root = Path(args.components)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"V8 structured-head result already exists: {result_path}")
    lock = json.loads(lock_path.read_text())
    config = lock["head_config"]
    component_metadata_path = component_root / "metadata.json"
    component_metadata = json.loads(component_metadata_path.read_text())
    if component_metadata["structured_head_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V8 components do not share the structured-head lock")
    component_path = Path(component_metadata["artifact"])
    if file_sha256(component_path) != component_metadata["artifact_sha256"]:
        raise RuntimeError("V8 component artifact changed")
    stage3_metadata = json.loads(Path(lock["stage3_features"]["metadata"]).read_text())
    stage3_path = Path(stage3_metadata["feature_artifact"])
    if file_sha256(stage3_path) != lock["stage3_features"]["artifact_sha256"]:
        raise RuntimeError("V8 Stage 3 features changed")
    with np.load(stage3_path, allow_pickle=False) as values:
        data = {key: values[key] for key in values.files}
    with np.load(component_path, allow_pickle=False) as values:
        components = {key: values[key] for key in values.files}
    global_features = data["layer_06_mean"].astype(np.float32)
    component_embeddings = components["embeddings"].astype(np.float32)
    if len(global_features) != len(components["action_indices"]):
        raise RuntimeError("V8 global and component features differ in record count")

    mechanics = list(lock["mechanics"])
    surfaces = list(lock["surfaces"])
    folds: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold_number, heldout in enumerate(mechanics):
        mx.random.seed(config["seed"] + fold_number)
        train_mask = (data["mechanics"] != heldout) & (data["splits"] == "train")
        train_indices = np.flatnonzero(train_mask)
        inputs = model_inputs(train_indices, global_features, component_embeddings, components)
        record_gold_np = data["gold_ambiguous"][train_indices].astype(np.float32)
        status_gold_np = components["status_targets"][train_indices].astype(np.int32)
        record_gold = mx.array(record_gold_np)
        status_gold = mx.array(status_gold_np)
        status_weight_values = class_weights(status_gold_np, len(STATUS_ORDER))
        record_weight_values = class_weights(record_gold_np.astype(np.int32), 2)
        status_weights = mx.array(status_weight_values[status_gold_np])
        record_weights = mx.array(record_weight_values[record_gold_np.astype(np.int32)])
        pairs_np = local_pairs(data, train_mask, train_indices)
        surface_groups_np = local_surface_groups(data, train_indices)
        pairs = mx.array(pairs_np)
        surface_groups = mx.array(surface_groups_np)
        model = ActionConditionedHead(global_features.shape[1], config["projectionWidth"])
        optimizer = optim.Adam(learning_rate=config["learningRate"])
        loss_fn = make_loss(
            config,
            record_gold,
            status_gold,
            status_weights,
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
            final_losses = {"total": float(loss), **{name: float(value) for name, value in parts.items()}}

        all_indices = np.arange(len(global_features))
        all_inputs = model_inputs(all_indices, global_features, component_embeddings, components)
        status_logits_mx, ambiguity_logits_mx = model(*all_inputs)
        mx.eval(status_logits_mx, ambiguity_logits_mx)
        status_logits = np.asarray(status_logits_mx, dtype=np.float32)
        ambiguity_logits = np.asarray(ambiguity_logits_mx, dtype=np.float32)

        calibration_mask = (
            (data["mechanics"] != heldout)
            & (data["splits"] == "calibration")
            & (data["surface_variants"] == "canonical")
        )
        threshold_report = fit_threshold(
            data["gold_ambiguous"][calibration_mask].astype(bool).tolist(),
            ambiguity_logits[calibration_mask].tolist(),
        )
        threshold = threshold_report["threshold"]
        by_surface: dict[str, Any] = {}
        for surface in surfaces:
            surface_mask = (data["mechanics"] == heldout) & (data["surface_variants"] == surface)
            surface_pairs = pair_indices(data, surface_mask)
            by_surface[surface] = {
                "pointwise": evaluate_binary(
                    data["gold_ambiguous"][surface_mask].astype(bool).tolist(),
                    ambiguity_logits[surface_mask].tolist(),
                    threshold,
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
            "threshold": threshold,
            "threshold_calibration": threshold_report,
            "parameter_artifact": str(parameter_path),
            "parameter_artifact_sha256": file_sha256(parameter_path),
            "by_surface": by_surface,
        }
        mx.clear_cache()

    gates = gate_report(folds, config["gates"])
    result = {
        "schema_version": 8,
        "experiment": "v8_action_conditioned_structured_head_lomo",
        "structured_head_lock": str(lock_path),
        "structured_head_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "stage3_result_sha256": lock["stage3_result"]["sha256"],
        "component_artifact_sha256": component_metadata["artifact_sha256"],
        "model": config["model"],
        "head_config": config,
        "status_order": STATUS_ORDER,
        "folds": folds,
        "gates": gates,
        "decision": "eligible_for_new_final_mechanic_protocol" if gates["passed"] else "structured_head_not_ready",
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
