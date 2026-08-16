#!/usr/bin/env python3
"""Train the three locked V31 LoRA-plus-structured-readout seeds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from audit_v31_signed_fact_adaptation import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v31_evaluation import summarize_seed
from v31_structured_model import (
    AdaptedStructuredGrounder, StructuredPointerHead, class_weights,
    loss_from_outputs, prompt_tokens_and_entity_spans, select_predictions,
)


def load_seed_parameters(model: nn.Module, path: Path) -> None:
    values = mx.load(path)
    expected = {name for name, _ in tree_flatten(model.trainable_parameters())}
    if set(values) != expected:
        raise RuntimeError("V31 saved LoRA/readout keys do not match the registered trainable keys")
    model.load_weights(list(values.items()), strict=False)
    mx.eval(model.parameters())


def predict_rows(
    model: AdaptedStructuredGrounder, tokenizer: Any, rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    predictions = []
    model.eval()
    for row in rows:
        tokens, spans, _ = prompt_tokens_and_entity_spans(row, config, tokenizer)
        outputs = model(mx.array([tokens]), spans)
        mx.eval(*outputs)
        values = tuple(np.asarray(value, dtype=np.float32) for value in outputs)
        predictions.extend(select_predictions([row], values, config))
        mx.clear_cache()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v31-signed-fact-adaptation-lock.json")
    parser.add_argument("--output-dir", default="outputs/v31-signed-fact-adaptation/lora-readout")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "lora-training-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V31 LoRA-readout training was already attempted")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    seeds = config["training"]["seeds"]
    if lock["limits"]["loraTrainingRuns"] != len(seeds):
        raise RuntimeError("V31 lock does not authorize the registered LoRA seeds")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V31 locked implementation changed: {path}")
    for name, expected in lock["source"]["corpus_file_sha256"].items():
        if file_sha256(PROJECT_ROOT / lock["source"]["corpus"] / name) != expected:
            raise RuntimeError(f"V31 corpus changed after lock: {name}")
    rows = sorted(read_rows(
        PROJECT_ROOT / lock["source"]["corpus"],
        ("adaptation_fit", "adaptation_calibration"),
    ), key=lambda row: row["id"])
    fit_rows = [row for row in rows if row["split"] == "adaptation_fit"]
    calibration_rows = [row for row in rows if row["split"] == "adaptation_calibration"]
    if len(rows) != lock["planned_training"]["fit_calibration_records"]:
        raise RuntimeError("V31 LoRA fit/calibration population differs from lock")
    predicate_weight_values, truth_weight_values = class_weights(fit_rows, config)
    predicate_weights = mx.array(predicate_weight_values)
    truth_weights = mx.array(truth_weight_values)
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 31, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path), "status": "started",
        "registered_seeds": seeds, "evaluation_records_read": 0,
    }, indent=2, sort_keys=True) + "\n")
    output_dir.mkdir(parents=True, exist_ok=False)
    ledgers = {}
    specification = config["model"]
    lora = config["systems"]["loraReadout"]
    accumulation = config["training"]["gradientAccumulationSteps"]
    if len(fit_rows) % accumulation:
        raise RuntimeError("V31 fit population is not divisible by gradient accumulation")
    for seed in seeds:
        started = time.perf_counter()
        base, tokenizer, model_config = load(
            specification["model"], revision=specification["revision"], return_config=True
        )
        text_config = model_config["text_config"]
        if text_config["num_hidden_layers"] != specification["totalLayers"] or text_config["hidden_size"] != specification["hiddenSize"]:
            raise RuntimeError("V31 LoRA model architecture differs from lock")
        core = base.language_model.model
        core.freeze()
        mx.random.seed(seed)
        linear_to_lora_layers(core, lora["lastLayers"], {
            "rank": lora["rank"], "scale": lora["scale"],
            "dropout": lora["dropout"], "keys": lora["moduleKeys"],
        })
        # Reset before head construction so its initialization is identical to the frozen arm.
        mx.random.seed(seed)
        head = StructuredPointerHead(
            specification["hiddenSize"], config["sharedStructuredHead"]["width"],
            len(config["sharedStructuredHead"]["predicateClasses"]),
            len(config["sharedStructuredHead"]["truthClasses"]),
        )
        model = AdaptedStructuredGrounder(
            core, head, max(config["construction"]["entityCounts"])
        )
        trainable = dict(tree_flatten(model.trainable_parameters()))
        backbone_names = [name for name in trainable if name.startswith("backbone.")]
        if not backbone_names or any("lora_" not in name for name in backbone_names):
            raise RuntimeError("V31 LoRA trainability escaped the registered adapter boundary")
        optimizer = optim.Adam(learning_rate=config["training"]["learningRate"])
        order = np.random.default_rng(seed).permutation(len(fit_rows))
        losses = []
        model.train()
        for start in range(0, len(order), accumulation):
            selected = order[start:start + accumulation]
            accumulated_gradients = None
            micro_losses = []
            for fit_index in selected:
                row = fit_rows[int(fit_index)]
                tokens, spans, _ = prompt_tokens_and_entity_spans(row, config, tokenizer)
                if len(tokens) > specification["maxSequenceLength"]:
                    raise RuntimeError(f"V31 LoRA prompt exceeds maximum: {row['id']}")

                def loss_fn(current: AdaptedStructuredGrounder):
                    return loss_from_outputs(
                        current(mx.array([tokens]), spans), [row], config,
                        predicate_weights, truth_weights,
                    )

                loss_and_grad = nn.value_and_grad(model, loss_fn)
                (loss, parts), gradients = loss_and_grad()
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
            step = start // accumulation + 1
            if args.progress_every and (step % args.progress_every == 0 or start + accumulation == len(order)):
                print(
                    f"v31 LoRA seed {seed}: optimizer step {step}/{len(order) // accumulation}",
                    flush=True,
                )
            mx.clear_cache()
        calibration_predictions = predict_rows(model, tokenizer, calibration_rows, config)
        calibration = summarize_seed(
            calibration_rows, calibration_predictions, config, apply_gates=False
        )
        parameter_path = output_dir / f"seed-{seed}.safetensors"
        mx.save_safetensors(parameter_path, dict(tree_flatten(model.trainable_parameters())))
        ledger = {
            "schema_version": 31, "system": "lora_readout", "seed": seed,
            "training_records": len(fit_rows), "epochs": config["training"]["epochs"],
            "optimizer_steps": len(order) // accumulation, "examples_seen": len(order),
            "final_loss": losses[-1],
            "mean_last_20_loss": {
                key: float(np.mean([row[key] for row in losses[-20:]])) for key in losses[-1]
            },
            "trainable_parameters": int(sum(value.size for value in trainable.values())),
            "trainable_parameter_names": sorted(trainable),
            "calibration": calibration,
            "parameter_artifact": str(parameter_path.relative_to(PROJECT_ROOT)),
            "parameter_artifact_sha256": file_sha256(parameter_path),
            "runtime_seconds": time.perf_counter() - started,
            "data_access": {
                "fit_records_read": len(fit_rows),
                "calibration_records_read": len(calibration_rows),
                "evaluation_records_read": 0, "evaluation_features_read": 0,
                "lora_training_runs": 1, "checkpoint_selections": 0,
                "hyperparameter_selections": 0,
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
        del model, head, core, base
        mx.clear_cache()
    manifest = {
        "schema_version": 31, "system": "lora_readout",
        "protocol_lock_sha256": file_sha256(lock_path), "training_runs": len(seeds),
        "seeds": ledgers, "evaluation_records_read": 0, "evaluation_features_read": 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "manifest_sha256": file_sha256(manifest_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
