#!/usr/bin/env python3
"""Run the bounded V33 fit/calibration learning curves and confirmation study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from audit_v32_factorized_semantics import read_rows
from train_v32_heads import load_arrays
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_structured_model import class_weights, make_head, make_loss
from v33_development import (
    combine_outputs, decode_outputs, score_development, select_qualified_system,
    select_search_checkpoint, system_qualification,
)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def output_arrays(model, arrays: dict[str, np.ndarray], indices: np.ndarray, batch_size: int = 128) -> tuple[np.ndarray, ...]:
    chunks: list[list[np.ndarray]] = [[] for _ in range(6)]
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        outputs = model(
            mx.array(arrays["clause_features"][selected]),
            mx.array(arrays["entity_features"][selected]), mx.array(arrays["entity_mask"][selected]),
        )
        mx.eval(*outputs)
        for target, value in zip(chunks, outputs, strict=True):
            target.append(np.asarray(value, dtype=np.float32))
    return tuple(np.concatenate(values, axis=0) for values in chunks)


def train_epoch(
    model, optimizer, rows, arrays, fit_indices, objective, weights,
    seed: int, epoch: int, batch_size: int, clip_norm: float, v32_config: dict[str, Any],
) -> dict[str, float]:
    order = np.random.default_rng(seed * 1000 + epoch).permutation(fit_indices)
    losses, norms = [], []
    for start in range(0, len(order), batch_size):
        selected = order[start:start + batch_size]
        selected_rows = [rows[index] for index in selected]
        loss_fn = make_loss(model, selected_rows, v32_config, weights, objective)
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        (loss, parts), gradients = loss_and_grad(
            mx.array(arrays["clause_features"][selected]),
            mx.array(arrays["entity_features"][selected]), mx.array(arrays["entity_mask"][selected]),
        )
        gradients, norm = optim.clip_grad_norm(gradients, clip_norm)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss, parts, norm)
        losses.append(float(loss)); norms.append(float(norm))
    return {"mean_total_loss": float(np.mean(losses)), "mean_gradient_norm": float(np.mean(norms)), "optimizer_steps": len(losses)}


def train_path(
    objective_name: str, objective: Sequence[str], learning_rate: float, epochs: int,
    seed: int, rows: list[dict], arrays: dict[str, np.ndarray], fit_indices: np.ndarray,
    calibration_indices: np.ndarray, config: dict[str, Any], weights: dict[str, mx.array],
    checkpoints: set[int] | None = None,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, float]]]:
    mx.random.seed(seed)
    model = make_head(config["v32_config"])
    optimizer = optim.Adam(learning_rate=learning_rate)
    curve, epoch_ledgers = [], []
    checkpoint_set = checkpoints or {epochs}
    for epoch in range(1, epochs + 1):
        ledger = train_epoch(
            model, optimizer, rows, arrays, fit_indices, objective, weights,
            seed, epoch, config["search"]["batchSize"], config["search"]["gradientClipNorm"],
            config["v32_config"],
        )
        epoch_ledgers.append({"epoch": epoch, **ledger})
        if epoch in checkpoint_set:
            model.eval()
            fit_outputs = output_arrays(model, arrays, fit_indices)
            calibration_outputs = output_arrays(model, arrays, calibration_indices)
            fit_rows = [rows[index] for index in fit_indices]
            calibration_rows = [rows[index] for index in calibration_indices]
            curve.append({
                "objective": objective_name, "learning_rate": learning_rate, "epoch": epoch,
                "fit": score_development(
                    fit_rows, decode_outputs(fit_rows, fit_outputs, config["v32_config"], compiled=False), config["v32_config"]
                ),
                "calibration": score_development(
                    calibration_rows, decode_outputs(calibration_rows, calibration_outputs, config["v32_config"], compiled=False), config["v32_config"]
                ),
                "epoch_training": ledger,
            })
            model.train()
    model.eval()
    return model, curve, epoch_ledgers


def system_mean(seed_values: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    metrics = (
        "atom_exact_accuracy", "relation_order_accuracy", "lexical_sign_accuracy",
        "outer_operation_accuracy", "direct_truth_accuracy", "compiled_truth_accuracy",
        "direct_exact_fact_accuracy", "compiled_exact_fact_accuracy",
    )
    return {
        split: {key: float(np.mean([row[split][key] for row in seed_values.values()])) for key in metrics}
        for split in ("fit", "calibration")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v33-development-adequacy-lock.json")
    parser.add_argument("--output-dir", default="outputs/v33-development-adequacy")
    args = parser.parse_args()
    lock_path, output_dir = (PROJECT_ROOT / args.lock).resolve(), (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir / "run-attempt.json"
    if (output_dir / "result.json").exists() or attempt_path.exists():
        raise RuntimeError("V33 development study was already attempted")
    lock = json.loads(lock_path.read_text())
    config, v32_config = lock["config_payload"], lock["v32_config_payload"]
    runtime_config = {**config, "v32_config": v32_config}
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V33 locked implementation changed: {path}")
    metadata_path = PROJECT_ROOT / config["sourceV32FeatureMetadata"]
    metadata = json.loads(metadata_path.read_text())
    if file_sha256(metadata_path) != lock["source"]["feature_metadata_sha256"]:
        raise RuntimeError("V33 source feature metadata changed")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != lock["source"]["feature_artifact_sha256"]:
        raise RuntimeError("V33 source features changed")
    arrays = load_arrays(feature_path)
    rows = sorted(read_rows(PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])), key=lambda row: row["id"])
    if arrays["record_ids"].tolist() != [row["id"] for row in rows]:
        raise RuntimeError("V33 allowed-record and feature order differs")
    fit_indices = np.flatnonzero(arrays["splits"] == "factor_fit")
    calibration_indices = np.flatnonzero(arrays["splits"] == "factor_calibration")
    fit_rows = [rows[index] for index in fit_indices]
    calibration_rows = [rows[index] for index in calibration_indices]
    weight_values = class_weights(fit_rows, v32_config)
    weights = {key: mx.array(value) for key, value in weight_values.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 33, "attempt_number": 1, "status": "started",
        "protocol_lock_sha256": file_sha256(lock_path), "fit_records": len(fit_indices),
        "calibration_records": len(calibration_indices), "v32_evaluation_records_read": 0,
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    search_reports, path_ledgers = [], []
    maximum_epoch = max(config["search"]["checkpointEpochs"])
    checkpoints = set(config["search"]["checkpointEpochs"])
    for objective_name, objective in config["search"]["objectives"].items():
        for learning_rate in config["search"]["learningRates"]:
            model, curve, ledgers = train_path(
                objective_name, objective, learning_rate, maximum_epoch, config["search"]["seed"],
                rows, arrays, fit_indices, calibration_indices, runtime_config, weights, checkpoints,
            )
            search_reports.extend(curve)
            path_ledgers.append({
                "objective": objective_name, "learning_rate": learning_rate,
                "epochs": maximum_epoch, "epoch_training": ledgers,
            })
            del model
            mx.clear_cache()
    selected = {
        objective: select_search_checkpoint(
            objective, [row for row in search_reports if row["objective"] == objective], config
        )
        for objective in config["search"]["objectives"]
    }
    search = {
        "schema_version": 33, "training_paths": len(path_ledgers),
        "checkpoint_evaluations": len(search_reports), "reports": search_reports,
        "selected": selected, "path_ledgers": path_ledgers,
    }
    search_path = output_dir / "search.json"
    search_path.write_text(json.dumps(search, indent=2, sort_keys=True) + "\n")
    confirmation_root = output_dir / "confirmation"
    confirmation_root.mkdir()
    candidate_seed_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        name: {} for name in config["confirmation"]["candidateSystems"]
    }
    parameter_artifacts, prediction_artifacts = {}, {}
    for seed in config["confirmation"]["seeds"]:
        models, outputs = {}, {"fit": {}, "calibration": {}}
        seed_root = confirmation_root / f"seed-{seed}"
        seed_root.mkdir()
        for objective_name, objective in config["search"]["objectives"].items():
            chosen = selected[objective_name]
            model, _, ledgers = train_path(
                objective_name, objective, chosen["learning_rate"], chosen["epoch"], seed,
                rows, arrays, fit_indices, calibration_indices, runtime_config, weights, {chosen["epoch"]},
            )
            models[objective_name] = model
            outputs["fit"][objective_name] = output_arrays(model, arrays, fit_indices)
            outputs["calibration"][objective_name] = output_arrays(model, arrays, calibration_indices)
            parameter_path = seed_root / f"{objective_name}.safetensors"
            mx.save_safetensors(parameter_path, dict(tree_flatten(model.parameters())))
            ledger_path = seed_root / f"{objective_name}-ledger.json"
            ledger_path.write_text(json.dumps({
                "schema_version": 33, "objective": objective_name, "objective_fields": objective,
                "seed": seed, "learning_rate": chosen["learning_rate"], "epochs": chosen["epoch"],
                "epoch_training": ledgers, "parameter_artifact": str(parameter_path.relative_to(PROJECT_ROOT)),
                "parameter_artifact_sha256": file_sha256(parameter_path),
                "data_access": {"fit_records_read": len(fit_indices), "calibration_records_read": len(calibration_indices), "v32_evaluation_records_read": 0, "v32_evaluation_features_read": 0},
            }, indent=2, sort_keys=True) + "\n")
            parameter_artifacts[str(parameter_path.relative_to(output_dir))] = file_sha256(parameter_path)
            parameter_artifacts[str(ledger_path.relative_to(output_dir))] = file_sha256(ledger_path)
        for split, selected_rows in (("fit", fit_rows), ("calibration", calibration_rows)):
            independent_outputs = combine_outputs(
                outputs[split]["atom"], outputs[split]["truth"],
                outputs[split]["lexicalSign"], outputs[split]["outerOperation"],
            )
            candidate_outputs = {
                "independentDirect": (independent_outputs, False),
                "independentCompiled": (independent_outputs, True),
                "jointDirect": (outputs[split]["jointAuxiliary"], False),
                "jointCompiled": (outputs[split]["jointAuxiliary"], True),
            }
            for candidate, (values, compiled) in candidate_outputs.items():
                predictions = decode_outputs(selected_rows, values, v32_config, compiled)
                candidate_seed_metrics[candidate].setdefault(str(seed), {})[split] = score_development(
                    selected_rows, predictions, v32_config
                )
                prediction_path = seed_root / f"{candidate}-{split}-predictions.jsonl"
                write_jsonl(prediction_path, predictions)
                prediction_artifacts[str(prediction_path.relative_to(output_dir))] = file_sha256(prediction_path)
        for model in models.values():
            del model
        mx.clear_cache()
    system_summaries = {name: {"seeds": values, "mean": system_mean(values)} for name, values in candidate_seed_metrics.items()}
    qualification = {
        name: system_qualification(candidate_seed_metrics[name], config)
        for name in config["qualification"]["eligibleSystems"]
    }
    selected_system, selection = select_qualified_system(qualification, config)
    if qualification["jointCompiled"]["passed"]:
        diagnosis = config["diagnosis"]["jointPass"]
    elif qualification["independentCompiled"]["passed"]:
        diagnosis = config["diagnosis"]["independentPassJointFailure"]
    else:
        fit_pass_any = any(
            all(value for key, value in seed["checks"].items() if key.startswith("fit_"))
            for system in qualification.values() for seed in system["seeds"].values()
        )
        diagnosis = config["diagnosis"]["fitPassCalibrationFailure"] if fit_pass_any else config["diagnosis"]["fitFailure"]
    result = {
        "schema_version": 33, "experiment": config["experiment"],
        "protocol_lock_sha256": file_sha256(lock_path), "search": str(search_path.relative_to(PROJECT_ROOT)),
        "search_sha256": file_sha256(search_path), "selected_search_configurations": selected,
        "systems": system_summaries, "qualification": qualification,
        "selection": selection, "selected_system": selected_system,
        "development_qualified": selected_system is not None, "diagnosis": diagnosis,
        "fresh_suite_preregistration_authorized": selected_system is not None,
        "v32_evaluation_reuse_authorized": False, "v28_authorized": False,
        "parameter_artifacts": dict(sorted(parameter_artifacts.items())),
        "prediction_artifacts": dict(sorted(prediction_artifacts.items())),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "fit_records_read": len(fit_indices), "calibration_records_read": len(calibration_indices),
            "search_training_paths": len(path_ledgers), "search_checkpoint_evaluations": len(search_reports),
            "confirmation_training_runs": len(config["search"]["objectives"]) * len(config["confirmation"]["seeds"]),
            "v32_evaluation_records_read": 0, "v32_evaluation_features_read": 0,
            "v32_evaluation_predictions_read": 0, "backbone_forward_passes": 0,
            "adapter_training_runs": 0, "v28_integration_replays": 0,
            "fresh_suite_constructions": 0,
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "result_sha256": file_sha256(result_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_search_configurations": {key: {"learning_rate": value["learning_rate"], "epoch": value["epoch"]} for key, value in selected.items()},
        "qualification": {key: {"passed": value["passed"], "passing_seeds": value["passing_seeds"], "fit_mean": value["fit_mean"], "calibration_mean": value["calibration_mean"]} for key, value in qualification.items()},
        "selected_system": selected_system, "diagnosis": diagnosis,
        "fresh_suite_preregistration_authorized": selected_system is not None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()
