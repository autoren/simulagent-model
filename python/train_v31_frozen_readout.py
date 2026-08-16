#!/usr/bin/env python3
"""Train the three locked V31 frozen structured-readout seeds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from audit_v31_signed_fact_adaptation import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v31_evaluation import summarize_seed
from v31_structured_model import StructuredPointerHead, class_weights, make_loss, select_predictions


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def predict(
    model: StructuredPointerHead, rows: list[dict], arrays: dict[str, np.ndarray],
    indices: np.ndarray, config: dict, batch_size: int = 128,
) -> list[dict]:
    result = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        outputs = model(
            mx.array(arrays["clause_features"][selected]),
            mx.array(arrays["entity_features"][selected]),
            mx.array(arrays["entity_mask"][selected]),
        )
        mx.eval(*outputs)
        values = tuple(np.asarray(value, dtype=np.float32) for value in outputs)
        result.extend(select_predictions([rows[index] for index in selected], values, config))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v31-signed-fact-adaptation-lock.json")
    parser.add_argument("--features", default="outputs/v31-signed-fact-adaptation/fit-calibration-features/metadata.json")
    parser.add_argument("--output-dir", default="outputs/v31-signed-fact-adaptation/frozen-readout")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    metadata_path = (PROJECT_ROOT / args.features).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "frozen-training-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V31 frozen-readout training was already attempted")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    seeds = config["training"]["seeds"]
    if lock["limits"]["frozenReadoutTrainingRuns"] != len(seeds):
        raise RuntimeError("V31 lock does not authorize the registered frozen seeds")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V31 locked implementation changed: {path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V31 frozen features do not share the protocol lock")
    artifact = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(artifact) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V31 frozen features changed")
    arrays = load_arrays(artifact)
    rows = sorted(read_rows(
        PROJECT_ROOT / lock["source"]["corpus"],
        ("adaptation_fit", "adaptation_calibration"),
    ), key=lambda row: row["id"])
    if arrays["record_ids"].tolist() != [row["id"] for row in rows]:
        raise RuntimeError("V31 feature/record order mismatch")
    fit_indices = np.flatnonzero(arrays["splits"] == "adaptation_fit")
    calibration_indices = np.flatnonzero(arrays["splits"] == "adaptation_calibration")
    predicate_weight_values, truth_weight_values = class_weights(
        [rows[index] for index in fit_indices], config
    )
    predicate_weights, truth_weights = mx.array(predicate_weight_values), mx.array(truth_weight_values)
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 31, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path), "status": "started",
        "registered_seeds": seeds, "evaluation_records_read": 0,
    }, indent=2, sort_keys=True) + "\n")
    output_dir.mkdir(parents=True, exist_ok=False)
    ledgers = {}
    for seed in seeds:
        mx.random.seed(seed)
        model = StructuredPointerHead(
            config["model"]["hiddenSize"], config["sharedStructuredHead"]["width"],
            len(config["sharedStructuredHead"]["predicateClasses"]),
            len(config["sharedStructuredHead"]["truthClasses"]),
        )
        optimizer = optim.Adam(learning_rate=config["training"]["learningRate"])
        order = np.random.default_rng(seed).permutation(fit_indices)
        accumulation = config["training"]["gradientAccumulationSteps"]
        if len(order) % accumulation:
            raise RuntimeError("V31 fit population is not divisible by gradient accumulation")
        started = time.perf_counter()
        losses = []
        for start in range(0, len(order), accumulation):
            selected = order[start:start + accumulation]
            accumulated_gradients = None
            micro_losses = []
            for index in selected:
                loss_fn = make_loss([rows[index]], config, predicate_weights, truth_weights)
                loss_and_grad = nn.value_and_grad(model, loss_fn)
                (loss, parts), gradients = loss_and_grad(
                    mx.array(arrays["clause_features"][index:index + 1]),
                    mx.array(arrays["entity_features"][index:index + 1]),
                    mx.array(arrays["entity_mask"][index:index + 1]),
                )
                mx.eval(loss, parts, gradients)
                accumulated_gradients = (
                    gradients if accumulated_gradients is None else
                    tree_map(lambda left, right: left + right, accumulated_gradients, gradients)
                )
                micro_losses.append({
                    "total": float(loss),
                    **{name: float(value) for name, value in parts.items()},
                })
            gradients = tree_map(lambda value: value / accumulation, accumulated_gradients)
            gradients, gradient_norm = optim.clip_grad_norm(
                gradients, config["training"]["gradientClipNorm"]
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, gradient_norm)
            losses.append({
                **{
                    name: float(np.mean([row[name] for row in micro_losses]))
                    for name in micro_losses[0]
                },
                "gradient_norm": float(gradient_norm),
            })
        model.eval()
        calibration_predictions = predict(model, rows, arrays, calibration_indices, config)
        calibration = summarize_seed(
            [rows[index] for index in calibration_indices], calibration_predictions,
            config, apply_gates=False,
        )
        parameter_path = output_dir / f"seed-{seed}.safetensors"
        mx.save_safetensors(parameter_path, dict(tree_flatten(model.parameters())))
        ledger = {
            "schema_version": 31, "system": "frozen_readout", "seed": seed,
            "training_records": len(fit_indices), "epochs": config["training"]["epochs"],
            "optimizer_steps": len(order) // accumulation,
            "examples_seen": len(order), "final_loss": losses[-1],
            "mean_last_20_loss": {
                key: float(np.mean([row[key] for row in losses[-20:]])) for key in losses[-1]
            },
            "trainable_parameters": int(sum(value.size for _, value in tree_flatten(model.parameters()))),
            "calibration": calibration,
            "parameter_artifact": str(parameter_path.relative_to(PROJECT_ROOT)),
            "parameter_artifact_sha256": file_sha256(parameter_path),
            "runtime_seconds": time.perf_counter() - started,
            "data_access": {
                "fit_records_read": len(fit_indices), "calibration_records_read": len(calibration_indices),
                "evaluation_records_read": 0, "evaluation_features_read": 0,
                "backbone_trainable_parameters": 0, "lora_training_runs": 0,
                "checkpoint_selections": 0, "hyperparameter_selections": 0,
            },
        }
        ledger_path = output_dir / f"seed-{seed}-ledger.json"
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        ledgers[str(seed)] = {
            "ledger": str(ledger_path.relative_to(PROJECT_ROOT)),
            "ledger_sha256": file_sha256(ledger_path),
            "parameters": ledger["parameter_artifact"],
            "parameters_sha256": ledger["parameter_artifact_sha256"],
        }
        mx.clear_cache()
    manifest = {
        "schema_version": 31, "system": "frozen_readout",
        "protocol_lock_sha256": file_sha256(lock_path), "training_runs": len(seeds),
        "seeds": ledgers, "feature_metadata_sha256": file_sha256(metadata_path),
        "evaluation_records_read": 0, "evaluation_features_read": 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "manifest_sha256": file_sha256(manifest_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__": main()
