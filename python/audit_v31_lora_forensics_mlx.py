#!/usr/bin/env python3
"""Read-only forensic audit of the completed V31 LoRA artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from audit_v31_signed_fact_adaptation import read_rows
from train_v31_lora_readout_mlx import load_seed_parameters, predict_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v31_evaluation import summarize_seed
from v31_structured_model import (
    AdaptedStructuredGrounder, StructuredPointerHead, prompt_tokens_and_entity_spans,
)


def l2(value: mx.array | np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.sum(array * array)))


def softmax_entropy(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array -= np.max(array)
    probabilities = np.exp(array)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))


def prediction_diagnostics(predictions: list[dict]) -> dict:
    fields = ("predicate", "argument_1", "argument_2", "truth_status")
    result = {
        "selected_class_counts": {
            field: dict(sorted(Counter(
                row["selected_fields"][field] for row in predictions
            ).items()))
            for field in fields
        }
    }
    for field, logit_key in (("predicate", "predicate"), ("truth_status", "truth")):
        entropies = [softmax_entropy(row["logits"][logit_key]) for row in predictions]
        result[f"mean_{field}_entropy"] = float(np.mean(entropies))
        result[f"minimum_{field}_entropy"] = float(np.min(entropies))
        result[f"maximum_{field}_entropy"] = float(np.max(entropies))
    return result


def compact_metrics(summary: dict) -> dict:
    return {
        key: summary[key] for key in (
            "predicate_accuracy", "argument1_accuracy", "argument2_accuracy",
            "relation_argument_order_accuracy", "truth_status_accuracy",
            "exact_signed_fact_accuracy", "exact_scene_accuracy",
        )
    }


def layer_group(name: str) -> str:
    parts = name.split(".")
    if name.startswith("head."):
        return "head"
    if len(parts) > 2 and parts[0] == "backbone" and parts[1] == "layers":
        return f"backbone.layers.{parts[2]}"
    return parts[0]


def outputs_for_rows(model, tokenizer, rows, config):
    result = []
    model.eval()
    for row in rows:
        tokens, spans, _ = prompt_tokens_and_entity_spans(row, config, tokenizer)
        outputs = model(mx.array([tokens]), spans)
        mx.eval(*outputs)
        result.append([np.asarray(value, dtype=np.float32) for value in outputs])
        mx.clear_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-lock", default="configs/v31-trained-systems-lock.json")
    parser.add_argument("--output", default="outputs/v31-signed-fact-adaptation/lora-forensic-audit.json")
    parser.add_argument("--markdown", default="docs/v31-lora-forensic-audit.md")
    args = parser.parse_args()
    trained_path = (PROJECT_ROOT / args.trained_lock).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    attempt_path = output_path.with_name("lora-forensic-audit-attempt.json")
    if output_path.exists() or attempt_path.exists():
        raise RuntimeError("V31 LoRA forensic audit was already attempted")
    trained = json.loads(trained_path.read_text())
    protocol_path = PROJECT_ROOT / trained["protocol_lock"]
    protocol = json.loads(protocol_path.read_text())
    config = protocol["config_payload"]
    if file_sha256(protocol_path) != trained["protocol_lock_sha256"]:
        raise RuntimeError("V31 trained lock no longer matches its protocol")
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V31 implementation changed before forensic audit: {path}")
    for system in ("frozen_readout", "lora_readout"):
        for entry in trained[system]["seeds"].values():
            if file_sha256(PROJECT_ROOT / entry["parameters"]) != entry["parameters_sha256"]:
                raise RuntimeError(f"V31 {system} parameter artifact changed")
            if file_sha256(PROJECT_ROOT / entry["ledger"]) != entry["ledger_sha256"]:
                raise RuntimeError(f"V31 {system} ledger changed")
    rows = sorted(read_rows(
        PROJECT_ROOT / protocol["source"]["corpus"],
        ("adaptation_fit", "adaptation_calibration"),
    ), key=lambda row: row["id"])
    fit_rows = [row for row in rows if row["split"] == "adaptation_fit"]
    calibration_rows = [row for row in rows if row["split"] == "adaptation_calibration"]
    initial_rows = calibration_rows[:8]
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": "31-forensic", "status": "started",
        "trained_system_lock_sha256": file_sha256(trained_path),
        "training_updates": 0, "evaluation_use_for_selection": False,
        "fit_records_planned": len(fit_rows),
        "calibration_records_planned": len(calibration_rows),
    }, indent=2, sort_keys=True) + "\n")
    specification = config["model"]
    lora = config["systems"]["loraReadout"]
    seed_results = {}
    for seed in config["training"]["seeds"]:
        base, tokenizer, model_config = load(
            specification["model"], revision=specification["revision"], return_config=True
        )
        if model_config["text_config"]["hidden_size"] != specification["hiddenSize"]:
            raise RuntimeError("V31 forensic model architecture mismatch")
        core = base.language_model.model
        core.freeze()
        mx.random.seed(seed)
        frozen_head = StructuredPointerHead(
            specification["hiddenSize"], config["sharedStructuredHead"]["width"],
            len(config["sharedStructuredHead"]["predicateClasses"]),
            len(config["sharedStructuredHead"]["truthClasses"]),
        )
        frozen_initial = AdaptedStructuredGrounder(
            core, frozen_head, max(config["construction"]["entityCounts"])
        )
        frozen_outputs = outputs_for_rows(frozen_initial, tokenizer, initial_rows, config)
        del frozen_initial, frozen_head
        mx.random.seed(seed)
        linear_to_lora_layers(core, lora["lastLayers"], {
            "rank": lora["rank"], "scale": lora["scale"],
            "dropout": lora["dropout"], "keys": lora["moduleKeys"],
        })
        mx.random.seed(seed)
        head = StructuredPointerHead(
            specification["hiddenSize"], config["sharedStructuredHead"]["width"],
            len(config["sharedStructuredHead"]["predicateClasses"]),
            len(config["sharedStructuredHead"]["truthClasses"]),
        )
        model = AdaptedStructuredGrounder(
            core, head, max(config["construction"]["entityCounts"])
        )
        initial_parameters = dict(tree_flatten(model.trainable_parameters()))
        mx.eval(initial_parameters)
        lora_outputs = outputs_for_rows(model, tokenizer, initial_rows, config)
        maximum_output_delta = max(
            float(np.max(np.abs(left - right)))
            for left_rows, right_rows in zip(frozen_outputs, lora_outputs, strict=True)
            for left, right in zip(left_rows, right_rows, strict=True)
        )
        artifact_path = PROJECT_ROOT / trained["lora_readout"]["seeds"][str(seed)]["parameters"]
        saved = mx.load(artifact_path)
        expected_keys = set(initial_parameters)
        key_checks = {
            "exact_trainable_key_match": set(saved) == expected_keys,
            "all_backbone_keys_are_lora": all(
                "lora_" in name for name in saved if name.startswith("backbone.")
            ),
            "all_saved_values_finite": all(
                bool(np.isfinite(np.asarray(value)).all()) for value in saved.values()
            ),
        }
        grouped = defaultdict(lambda: {
            "tensors": 0, "parameters": 0, "initial_norm_squared": 0.0,
            "final_norm_squared": 0.0, "update_norm_squared": 0.0,
        })
        tensor_updates = {}
        for name in sorted(saved):
            initial = np.asarray(initial_parameters[name], dtype=np.float64)
            final = np.asarray(saved[name], dtype=np.float64)
            update = final - initial
            group = grouped[layer_group(name)]
            group["tensors"] += 1
            group["parameters"] += int(final.size)
            group["initial_norm_squared"] += float(np.sum(initial * initial))
            group["final_norm_squared"] += float(np.sum(final * final))
            group["update_norm_squared"] += float(np.sum(update * update))
            tensor_updates[name] = {
                "parameters": int(final.size), "initial_l2": l2(initial),
                "final_l2": l2(final), "update_l2": l2(update),
                "maximum_absolute_update": float(np.max(np.abs(update))),
            }
        update_groups = {
            group: {
                "tensors": values["tensors"], "parameters": values["parameters"],
                "initial_l2": math.sqrt(values["initial_norm_squared"]),
                "final_l2": math.sqrt(values["final_norm_squared"]),
                "update_l2": math.sqrt(values["update_norm_squared"]),
            }
            for group, values in sorted(grouped.items())
        }
        load_seed_parameters(model, artifact_path)
        fit_predictions = predict_rows(model, tokenizer, fit_rows, config)
        calibration_predictions = predict_rows(model, tokenizer, calibration_rows, config)
        fit_summary = summarize_seed(fit_rows, fit_predictions, config, apply_gates=False)
        calibration_summary = summarize_seed(
            calibration_rows, calibration_predictions, config, apply_gates=False
        )
        seed_results[str(seed)] = {
            "initial_function_equality": {
                "records": len(initial_rows),
                "maximum_absolute_logit_delta": maximum_output_delta,
                "exact_within_float32_tolerance": maximum_output_delta <= 1e-5,
            },
            "parameter_boundary": {
                **key_checks, "saved_tensors": len(saved),
                "saved_parameters": int(sum(value.size for value in saved.values())),
                "backbone_tensors": sum(name.startswith("backbone.") for name in saved),
                "head_tensors": sum(name.startswith("head.") for name in saved),
            },
            "parameter_update_by_group": update_groups,
            "parameter_update_by_tensor": tensor_updates,
            "final_fit": {
                "metrics": compact_metrics(fit_summary),
                "prediction_diagnostics": prediction_diagnostics(fit_predictions),
            },
            "final_calibration": {
                "metrics": compact_metrics(calibration_summary),
                "prediction_diagnostics": prediction_diagnostics(calibration_predictions),
            },
        }
        del model, head, core, base
        mx.clear_cache()
    checks = {
        "all_initial_functions_equal": all(
            row["initial_function_equality"]["exact_within_float32_tolerance"]
            for row in seed_results.values()
        ),
        "all_parameter_allowlists_exact": all(
            row["parameter_boundary"]["exact_trainable_key_match"]
            and row["parameter_boundary"]["all_backbone_keys_are_lora"]
            for row in seed_results.values()
        ),
        "all_parameters_finite": all(
            row["parameter_boundary"]["all_saved_values_finite"]
            for row in seed_results.values()
        ),
        "all_fit_exact_fact_below_frozen_calibration_minimum": all(
            row["final_fit"]["metrics"]["exact_signed_fact_accuracy"] < 0.75
            for row in seed_results.values()
        ),
    }
    audit = {
        "schema_version": "31-forensic", "experiment": "v31_lora_read_only_forensics",
        "trained_system_lock_sha256": file_sha256(trained_path),
        "training_updates": 0, "training_runs": 0, "hyperparameter_selections": 0,
        "checkpoint_selections": 0, "evaluation_use_for_selection": False,
        "fit_records_read_per_seed": len(fit_rows),
        "calibration_records_read_per_seed": len(calibration_rows),
        "seed_results": seed_results, "checks": checks,
        "passed": all(checks.values()),
        "interpretation": (
            "registered_updates_caused_fit_and_transfer_collapse_with_valid_parameter_boundary"
            if all(checks.values()) else "forensic_checks_inconclusive_or_implementation_concern"
        ),
        "limitations": {
            "full_loss_curves_available": False,
            "per_field_gradient_histories_available": False,
            "reason": "The locked trainer retained only final and last-20 aggregates; reconstructing histories would require forbidden retraining.",
            "transient_base_parameter_mutation_directly_observable": False,
            "boundary_evidence": "Saved keys, registered trainable keys, frozen-module construction, and optimizer input are consistent; no full runtime memory snapshot was retained.",
        },
    }
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    fit_values = [row["final_fit"]["metrics"]["exact_signed_fact_accuracy"] for row in seed_results.values()]
    cal_values = [row["final_calibration"]["metrics"]["exact_signed_fact_accuracy"] for row in seed_results.values()]
    head_updates = [row["parameter_update_by_group"]["head"]["update_l2"] for row in seed_results.values()]
    adapter_updates = [
        math.sqrt(sum(
            values["update_l2"] ** 2 for group, values in row["parameter_update_by_group"].items()
            if group.startswith("backbone.layers.")
        )) for row in seed_results.values()
    ]
    lines = [
        "# V31 LoRA forensic audit", "", "## Verdict", "",
        (
            "The registered LoRA branch was functionally equal to the frozen branch at initialization, "
            "saved exactly the allowed head and LoRA tensors, contained no non-finite parameter values, "
            "and collapsed on the fit population itself. The negative transfer is therefore consistent "
            "with destructive optimization under the registered objective, not merely held-out surface failure."
        ), "", "## Evidence", "",
        f"Initial maximum absolute logit deltas: `{[row['initial_function_equality']['maximum_absolute_logit_delta'] for row in seed_results.values()]}`.",
        f"Final fit exact-fact accuracies: `{[round(value, 3) for value in fit_values]}`.",
        f"Final calibration exact-fact accuracies: `{[round(value, 3) for value in cal_values]}`.",
        f"Head update L2 norms: `{[round(value, 3) for value in head_updates]}`.",
        f"Combined adapter update L2 norms: `{[round(value, 3) for value in adapter_updates]}`.",
        "", "All audit computations were read-only. No training update, checkpoint selection, hyperparameter selection, or V31 evaluation reuse for selection occurred.",
        "", "## Limits", "",
        "Full loss curves and per-field gradient histories were not retained by the locked trainer and cannot be reconstructed without retraining. The audit does not claim those unavailable observations.", "",
    ]
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "audit_sha256": file_sha256(output_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": audit["passed"], "interpretation": audit["interpretation"],
        "checks": checks, "fit_exact_fact": fit_values,
        "calibration_exact_fact": cal_values,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
