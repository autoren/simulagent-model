#!/usr/bin/env python3
"""Train the registered V32 monolithic and joint-auxiliary heads."""

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

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_evaluation import summarize_seed
from v32_structured_model import class_weights, make_head, make_loss, select_predictions


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def predict(model, rows, arrays, indices, config, decoding="direct_truth_head", batch_size=128):
    result = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        outputs = model(mx.array(arrays["clause_features"][selected]), mx.array(arrays["entity_features"][selected]), mx.array(arrays["entity_mask"][selected]))
        mx.eval(*outputs)
        result.extend(select_predictions(
            [rows[index] for index in selected], tuple(np.asarray(value, dtype=np.float32) for value in outputs), config, decoding,
        ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v32-factorized-semantics-lock.json")
    parser.add_argument("--features", default="outputs/v32-factorized-semantics/fit-calibration-features/metadata.json")
    parser.add_argument("--output-dir", default="outputs/v32-factorized-semantics/training")
    args = parser.parse_args()
    lock_path, metadata_path = (PROJECT_ROOT / args.lock).resolve(), (PROJECT_ROOT / args.features).resolve()
    output_dir, attempt_path = (PROJECT_ROOT / args.output_dir).resolve(), PROJECT_ROOT / "outputs/v32-factorized-semantics/training-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V32 head training was already attempted")
    lock, metadata = json.loads(lock_path.read_text()), json.loads(metadata_path.read_text())
    config, seeds = lock["config_payload"], lock["config_payload"]["training"]["seeds"]
    if lock["limits"]["monolithicTrainingRuns"] != len(seeds) or lock["limits"]["jointAuxiliaryTrainingRuns"] != len(seeds):
        raise RuntimeError("V32 lock does not authorize exactly six registered runs")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V32 locked implementation changed: {path}")
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V32 features do not share the protocol lock")
    artifact = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(artifact) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V32 feature artifact changed")
    arrays = load_arrays(artifact)
    rows = sorted(read_rows(PROJECT_ROOT / lock["source"]["corpus"], ("factor_fit", "factor_calibration")), key=lambda row: row["id"])
    if arrays["record_ids"].tolist() != [row["id"] for row in rows]:
        raise RuntimeError("V32 feature/record order mismatch")
    fit_indices, calibration_indices = np.flatnonzero(arrays["splits"] == "factor_fit"), np.flatnonzero(arrays["splits"] == "factor_calibration")
    weights = {key: mx.array(value) for key, value in class_weights([rows[index] for index in fit_indices], config).items()}
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 32, "attempt_number": 1, "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started", "registered_seeds": seeds, "registered_artifacts": ["monolithic", "joint_auxiliary"],
        "evaluation_records_read": 0,
    }, indent=2, sort_keys=True) + "\n")
    output_dir.mkdir(parents=True, exist_ok=False)
    systems = {
        "monolithic": config["systems"]["monolithic"]["trainingObjective"],
        "joint_auxiliary": config["systems"]["auxiliaryDirect"]["trainingObjective"],
    }
    manifests = {}
    for system, objective in systems.items():
        system_dir = output_dir / system
        system_dir.mkdir()
        ledgers = {}
        for seed in seeds:
            mx.random.seed(seed)
            model = make_head(config)
            optimizer = optim.Adam(learning_rate=config["training"]["learningRate"])
            order = np.random.default_rng(seed).permutation(fit_indices)
            accumulation = config["training"]["gradientAccumulationSteps"]
            if len(order) % accumulation:
                raise RuntimeError("V32 fit population is not divisible by accumulation")
            started, losses = time.perf_counter(), []
            for start in range(0, len(order), accumulation):
                selected = order[start:start + accumulation]
                accumulated, micro = None, []
                for index in selected:
                    loss_fn = make_loss(model, [rows[index]], config, weights, objective)
                    loss_and_grad = nn.value_and_grad(model, loss_fn)
                    (loss, parts), gradients = loss_and_grad(
                        mx.array(arrays["clause_features"][index:index + 1]),
                        mx.array(arrays["entity_features"][index:index + 1]), mx.array(arrays["entity_mask"][index:index + 1]),
                    )
                    mx.eval(loss, parts, gradients)
                    accumulated = gradients if accumulated is None else tree_map(lambda left, right: left + right, accumulated, gradients)
                    micro.append({"total": float(loss), **{key: float(value) for key, value in parts.items()}})
                gradients = tree_map(lambda value: value / accumulation, accumulated)
                gradients, norm = optim.clip_grad_norm(gradients, config["training"]["gradientClipNorm"])
                optimizer.update(model, gradients)
                mx.eval(model.parameters(), optimizer.state, norm)
                losses.append({**{key: float(np.mean([row[key] for row in micro])) for key in micro[0]}, "gradient_norm": float(norm)})
            model.eval()
            calibration_predictions = predict(model, rows, arrays, calibration_indices, config)
            calibration = summarize_seed([rows[index] for index in calibration_indices], calibration_predictions, config, apply_gates=False)
            compiled_predictions = predict(model, rows, arrays, calibration_indices, config, "fixed_registered_truth_compiler")
            calibration_compiled = summarize_seed([rows[index] for index in calibration_indices], compiled_predictions, config, apply_gates=False)
            parameter_path = system_dir / f"seed-{seed}.safetensors"
            mx.save_safetensors(parameter_path, dict(tree_flatten(model.parameters())))
            ledger = {
                "schema_version": 32, "system": system, "seed": seed, "objective": objective,
                "training_records": len(fit_indices), "epochs": config["training"]["epochs"],
                "optimizer_steps": len(order) // accumulation, "examples_seen": len(order),
                "final_loss": losses[-1], "mean_last_20_loss": {key: float(np.mean([row[key] for row in losses[-20:]])) for key in losses[-1]},
                "trainable_parameters": int(sum(value.size for _, value in tree_flatten(model.parameters()))),
                "calibration_direct": calibration, "calibration_compiled": calibration_compiled,
                "parameter_artifact": str(parameter_path.relative_to(PROJECT_ROOT)), "parameter_artifact_sha256": file_sha256(parameter_path),
                "runtime_seconds": time.perf_counter() - started,
                "data_access": {"fit_records_read": len(fit_indices), "calibration_records_read": len(calibration_indices), "evaluation_records_read": 0, "evaluation_features_read": 0, "checkpoint_selections": 0, "hyperparameter_selections": 0},
            }
            ledger_path = system_dir / f"seed-{seed}-ledger.json"
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
            ledgers[str(seed)] = {"ledger": str(ledger_path.relative_to(PROJECT_ROOT)), "ledger_sha256": file_sha256(ledger_path), "parameters": ledger["parameter_artifact"], "parameters_sha256": ledger["parameter_artifact_sha256"]}
            del model
            mx.clear_cache()
        manifest = {"schema_version": 32, "system": system, "protocol_lock_sha256": file_sha256(lock_path), "training_runs": len(seeds), "seeds": ledgers, "feature_metadata_sha256": file_sha256(metadata_path), "evaluation_records_read": 0, "evaluation_features_read": 0}
        manifest_path = system_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifests[system] = {"manifest": str(manifest_path.relative_to(PROJECT_ROOT)), "manifest_sha256": file_sha256(manifest_path), "seeds": ledgers}
    overall = {"schema_version": 32, "experiment": "v32_registered_head_training", "protocol_lock_sha256": file_sha256(lock_path), "training_runs": 6, "systems": manifests}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(overall, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "manifest_sha256": file_sha256(manifest_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__": main()
