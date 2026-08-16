#!/usr/bin/env python3
"""Open and score the V31 sealed families once after the trained-system lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from audit_v31_signed_fact_adaptation import read_rows
from evaluate_v30_signed_fact_language_mlx import dequantized_label_rows, score_prompt
from train_v31_lora_readout_mlx import load_seed_parameters, predict_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import LABEL_TOKENS, select_option
from v31_evaluation import family_bootstrap_delta, summarize_seed, system_summary
from v31_language import zero_shot_field_prompt
from v31_structured_model import (
    AdaptedStructuredGrounder, StructuredPointerHead, features_from_hidden,
    prompt_tokens_and_entity_spans, select_predictions,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ))


def load_frozen_head(path: Path, config: dict) -> StructuredPointerHead:
    head = StructuredPointerHead(
        config["model"]["hiddenSize"], config["sharedStructuredHead"]["width"],
        len(config["sharedStructuredHead"]["predicateClasses"]),
        len(config["sharedStructuredHead"]["truthClasses"]),
    )
    values = mx.load(path)
    expected = {name for name, _ in tree_flatten(head.parameters())}
    if set(values) != expected:
        raise RuntimeError("V31 frozen-readout artifact keys changed")
    head.load_weights(list(values.items()), strict=True)
    head.eval()
    mx.eval(head.parameters())
    return head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-lock", default="configs/v31-trained-systems-lock.json")
    parser.add_argument("--output-dir", default="outputs/v31-signed-fact-adaptation/sealed-evaluation")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    trained_path = (PROJECT_ROOT / args.trained_lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V31 sealed evaluation was already attempted")
    trained = json.loads(trained_path.read_text())
    protocol_path = PROJECT_ROOT / trained["protocol_lock"]
    if file_sha256(protocol_path) != trained["protocol_lock_sha256"]:
        raise RuntimeError("V31 protocol lock changed after training")
    protocol = json.loads(protocol_path.read_text())
    config = protocol["config_payload"]
    if protocol["limits"]["sealedEvaluations"] != 1 or trained["proof"]["trained_systems"] != 6:
        raise RuntimeError("V31 locks do not authorize one sealed evaluation of six systems")
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V31 locked implementation changed: {path}")
    for name, expected in protocol["source"]["corpus_file_sha256"].items():
        if file_sha256(PROJECT_ROOT / protocol["source"]["corpus"] / name) != expected:
            raise RuntimeError(f"V31 corpus changed after lock: {name}")
    for system in ("frozen_readout", "lora_readout"):
        entry = trained[system]
        if file_sha256(PROJECT_ROOT / entry["manifest"]) != entry["manifest_sha256"]:
            raise RuntimeError(f"V31 trained {system} manifest changed")
        for seed_entry in entry["seeds"].values():
            if file_sha256(PROJECT_ROOT / seed_entry["parameters"]) != seed_entry["parameters_sha256"]:
                raise RuntimeError(f"V31 trained {system} parameters changed")
            if file_sha256(PROJECT_ROOT / seed_entry["ledger"]) != seed_entry["ledger_sha256"]:
                raise RuntimeError(f"V31 trained {system} ledger changed")
    rows = sorted(read_rows(
        PROJECT_ROOT / protocol["source"]["corpus"], ("adaptation_evaluation",)
    ), key=lambda row: row["id"])
    if len(rows) != protocol["planned_evaluation"]["records"]:
        raise RuntimeError("V31 sealed evaluation population differs from lock")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 31, "attempt_number": 1,
        "trained_system_lock_sha256": file_sha256(trained_path),
        "status": "started_before_model_load", "evaluation_records": len(rows),
    }, indent=2, sort_keys=True) + "\n")
    output_dir.mkdir(parents=True, exist_ok=False)
    specification = config["model"]
    base, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    base.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != specification["totalLayers"] or text_config["hidden_size"] != specification["hiddenSize"]:
        raise RuntimeError("V31 evaluation model architecture differs from lock")
    encoded = {token: tokenizer.encode(token, add_special_tokens=False) for token in LABEL_TOKENS}
    if any(len(values) != 1 for values in encoded.values()):
        raise RuntimeError(f"V31 label inventory is not single-token: {encoded}")
    label_rows = dequantized_label_rows(base, [encoded[token][0] for token in LABEL_TOKENS])
    mx.eval(label_rows)
    max_entities = max(config["construction"]["entityCounts"])
    clauses, entities, masks = [], [], []
    prompt_lengths = []
    prompt_payload = hashlib.sha256()
    zero_shot_predictions = []
    fields = ("predicate", "argument_1", "argument_2", "truth_status")
    for index, row in enumerate(rows, start=1):
        tokens, spans, content = prompt_tokens_and_entity_spans(row, config, tokenizer)
        hidden = base.language_model.model(mx.array([tokens]))[0]
        clause, entity, mask = features_from_hidden(hidden, spans, max_entities)
        mx.eval(clause, entity, mask)
        clauses.append(np.asarray(clause, dtype=np.float32))
        entities.append(np.asarray(entity, dtype=np.float32))
        masks.append(np.asarray(mask, dtype=bool))
        prompt_lengths.append(len(tokens))
        prompt_payload.update(content.encode())
        selected_fields = {}
        for field in fields:
            prompt, options = zero_shot_field_prompt(row, field, config)
            logits, length, _ = score_prompt(
                prompt, specification["systemPrompt"], len(options), base, tokenizer,
                label_rows, specification["maxSequenceLength"],
            )
            selected_fields[field] = select_option(logits, options)["value"]
            prompt_lengths.append(length)
        zero_shot_predictions.append({
            "id": row["id"], "scene_id": row["scene_id"], "split": row["split"],
            "selected_fields": selected_fields,
        })
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"v31 sealed reference+features: {index}/{len(rows)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    feature_path = output_dir / "evaluation-features.npz"
    np.savez_compressed(
        feature_path, record_ids=np.asarray([row["id"] for row in rows]),
        clause_features=np.stack(clauses).astype(np.float32),
        entity_features=np.stack(entities).astype(np.float32), entity_mask=np.stack(masks),
    )
    feature_sha = file_sha256(feature_path)
    clause_values = np.stack(clauses).astype(np.float32)
    entity_values = np.stack(entities).astype(np.float32)
    mask_values = np.stack(masks)
    frozen_predictions = {}
    frozen_seed_results = {}
    for seed in config["training"]["seeds"]:
        parameter_path = PROJECT_ROOT / trained["frozen_readout"]["seeds"][str(seed)]["parameters"]
        head = load_frozen_head(parameter_path, config)
        collected = []
        for start in range(0, len(rows), 128):
            outputs = head(
                mx.array(clause_values[start:start + 128]),
                mx.array(entity_values[start:start + 128]), mx.array(mask_values[start:start + 128]),
            )
            mx.eval(*outputs)
            values = tuple(np.asarray(value, dtype=np.float32) for value in outputs)
            collected.extend(select_predictions(rows[start:start + 128], values, config))
        frozen_predictions[str(seed)] = collected
        frozen_seed_results[str(seed)] = summarize_seed(rows, collected, config, apply_gates=True)
        write_jsonl(output_dir / f"frozen-seed-{seed}-predictions.jsonl", collected)
        del head
        mx.clear_cache()
    zero_summary = summarize_seed(rows, zero_shot_predictions, config, apply_gates=False)
    write_jsonl(output_dir / "zero-shot-reference-predictions.jsonl", zero_shot_predictions)
    del base
    mx.clear_cache()
    lora_predictions = {}
    lora_seed_results = {}
    lora_config = config["systems"]["loraReadout"]
    for seed in config["training"]["seeds"]:
        base, tokenizer, model_config = load(
            specification["model"], revision=specification["revision"], return_config=True
        )
        core = base.language_model.model
        core.freeze()
        mx.random.seed(seed)
        linear_to_lora_layers(core, lora_config["lastLayers"], {
            "rank": lora_config["rank"], "scale": lora_config["scale"],
            "dropout": lora_config["dropout"], "keys": lora_config["moduleKeys"],
        })
        mx.random.seed(seed)
        head = StructuredPointerHead(
            specification["hiddenSize"], config["sharedStructuredHead"]["width"],
            len(config["sharedStructuredHead"]["predicateClasses"]),
            len(config["sharedStructuredHead"]["truthClasses"]),
        )
        model = AdaptedStructuredGrounder(core, head, max_entities)
        parameter_path = PROJECT_ROOT / trained["lora_readout"]["seeds"][str(seed)]["parameters"]
        load_seed_parameters(model, parameter_path)
        collected = predict_rows(model, tokenizer, rows, config)
        lora_predictions[str(seed)] = collected
        lora_seed_results[str(seed)] = summarize_seed(rows, collected, config, apply_gates=True)
        write_jsonl(output_dir / f"lora-seed-{seed}-predictions.jsonl", collected)
        print(f"v31 sealed LoRA seed {seed}: {len(collected)}/{len(rows)}", file=sys.stderr, flush=True)
        del model, head, core, base
        mx.clear_cache()
    frozen_summary = system_summary(frozen_seed_results, config)
    lora_summary = system_summary(lora_seed_results, config)
    paired = family_bootstrap_delta(rows, frozen_predictions, lora_predictions, config)
    fact_delta = lora_summary["mean"]["exact_signed_fact_accuracy"] - frozen_summary["mean"]["exact_signed_fact_accuracy"]
    scene_delta = lora_summary["mean"]["exact_scene_accuracy"] - frozen_summary["mean"]["exact_scene_accuracy"]
    advantage_gates = config["gates"]["loraMaterialAdvantage"]
    advantage_checks = {
        "exact_signed_fact_delta": fact_delta >= advantage_gates["minimumMeanExactSignedFactDelta"],
        "exact_scene_delta": scene_delta >= advantage_gates["minimumMeanExactSceneDelta"],
        "paired_family_bootstrap_lower_bound": paired["bootstrap_95_interval"][0] > advantage_gates["minimumPairedFamilyBootstrapLowerBound"],
    }
    material_advantage = all(advantage_checks.values())
    if frozen_summary["passed"]:
        selected = "lora_readout" if lora_summary["passed"] and material_advantage else "frozen_readout"
    elif lora_summary["passed"]:
        selected = "lora_readout"
    else:
        selected = None
    decision = {
        "frozen_readout": "frozen_pass_selected_no_material_lora_advantage",
        "lora_readout": "lora_pass_selected_representation_adaptation_supported",
        None: "both_learned_systems_fail_stop_no_v28_replay",
    }[selected]
    prediction_artifacts = {
        path.name: file_sha256(path) for path in sorted(output_dir.glob("*predictions.jsonl"))
    }
    result = {
        "schema_version": 31, "experiment": config["experiment"],
        "protocol_lock_sha256": file_sha256(protocol_path),
        "trained_system_lock_sha256": file_sha256(trained_path), "evaluation_number": 1,
        "zero_shot_reference": zero_summary,
        "frozen_readout": frozen_summary, "lora_readout": lora_summary,
        "lora_minus_frozen": {
            "mean_exact_signed_fact_delta": fact_delta,
            "mean_exact_scene_delta": scene_delta, "paired_family": paired,
            "checks": advantage_checks, "material_advantage": material_advantage,
        },
        "selected_system": selected, "passed": selected is not None,
        "decision": decision, "v28_integration_authorized": selected is not None,
        "evaluation_features": str(feature_path.relative_to(PROJECT_ROOT)),
        "evaluation_features_sha256": feature_sha,
        "prediction_artifacts": prediction_artifacts,
        "minimum_prompt_tokens": min(prompt_lengths), "maximum_prompt_tokens": max(prompt_lengths),
        "prompt_payload_sha256": prompt_payload.hexdigest(), "truncated_prompts": 0,
        "data_access": {
            "evaluation_records_read": len(rows),
            "zero_shot_model_forward_passes": len(rows) * 4,
            "frozen_feature_model_forward_passes": len(rows),
            "lora_model_forward_passes": len(rows) * len(config["training"]["seeds"]),
            "sealed_evaluations": 1, "seed_selections": 0,
            "checkpoint_selections": 0, "hyperparameter_selections": 0,
            "v28_integration_replays": 0,
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({
        "status": "completed", "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
    })
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
