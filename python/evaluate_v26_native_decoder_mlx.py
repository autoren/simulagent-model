#!/usr/bin/env python3
"""Run the single locked V26 full-depth native decoder evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Sequence

import mlx.core as mx
import numpy as np
from mlx_lm import load

from audit_v22r2_grounding import read_jsonl_directory
from audit_v26_native_decoder import read_rows
from evaluate_v22r2_relational_grounding import (
    condition_modes,
    grounding_summary,
    integration_condition,
)
from evaluate_v25_truth_hypotheses import gate_checks
from extract_v10_features_mlx import chat_prompt
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT, predicted_epistemic_rows, validate_scene_prediction
from v26_native_decoder import decoder_prompt, select_label


def dequantized_label_rows(model: Any, token_ids: Sequence[int]) -> mx.array:
    embedding = model.language_model.model.embed_tokens
    indices = mx.array(list(token_ids))
    return mx.dequantize(
        embedding.weight[indices], embedding.scales[indices], embedding.biases[indices],
        group_size=embedding.group_size, bits=embedding.bits, mode=embedding.mode,
        dtype=mx.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v26-native-truth-decoder-lock.json")
    parser.add_argument("--output-dir", default="outputs/v26-native-truth-decoder/evaluation")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V26 native decoder evaluation was already attempted")
    lock = json.loads(lock_path.read_text())
    limits = lock["limits"]
    if not (
        limits["nativeDecoderEvaluationAttempts"] == 1
        and limits["modelForwardPasses"] == lock["pre_evaluation_audit"]["budget"]["planned_model_forwards"]
        and limits["headFits"] == 0 and limits["thresholdFits"] == 0
        and limits["integrationEvaluations"] == 1
    ):
        raise RuntimeError("V26 lock does not authorize the registered decoder evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V26 locked implementation changed: {path}")
    audit_path = PROJECT_ROOT / lock["source"]["pre_evaluation_audit"]
    if file_sha256(audit_path) != lock["source"]["pre_evaluation_audit_sha256"]:
        raise RuntimeError("V26 pre-evaluation audit changed after lock")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v26_protocol_lock":
        raise RuntimeError("V26 audit did not authorize model access")
    corpus_root = PROJECT_ROOT / lock["source"]["corpus"]
    rows = sorted(read_rows(corpus_root), key=lambda row: row["id"])
    for name, expected in lock["source"]["corpus_file_sha256"].items():
        if file_sha256(corpus_root / name) != expected:
            raise RuntimeError(f"V26 corpus file changed after lock: {name}")

    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 26,
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "attempt_number": 1,
        "status": "started_before_model_load",
    }, indent=2, sort_keys=True) + "\n")
    specification = lock["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if (
        text_config["num_hidden_layers"] != specification["totalLayers"]
        or text_config["hidden_size"] != specification["hiddenSize"]
    ):
        raise RuntimeError("V26 loaded model architecture differs from the lock")
    labels = lock["config_payload"]["labels"]
    encoded = {
        row["token"]: tokenizer.encode(row["token"], add_special_tokens=False) for row in labels
    }
    if any(len(tokens) != 1 for tokens in encoded.values()):
        raise RuntimeError(f"V26 labels are not single tokens: {encoded}")
    token_ids = [encoded[row["token"]][0] for row in labels]
    label_indices = mx.array(token_ids)
    label_rows = dequantized_label_rows(model, token_ids)
    mx.eval(label_rows)
    embedding = model.language_model.model.embed_tokens

    score_rows = []
    prompt_lengths = []
    observed_dtypes = None
    for index, row in enumerate(rows, start=1):
        content = decoder_prompt(row)
        prompt = chat_prompt(content, specification["systemPrompt"], tokenizer)
        tokens = tokenizer.encode(prompt)
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError(f"V26 prompt exceeds locked maximum: {row['id']} has {len(tokens)}")
        hidden = model.language_model.model(mx.array([tokens]))[0, -1]
        bf16_logits = embedding.as_linear(hidden)[label_indices]
        fp32_logits = hidden.astype(mx.float32) @ label_rows.T
        mx.eval(hidden, bf16_logits, fp32_logits)
        bf16_values = [float(value) for value in bf16_logits.tolist()]
        fp32_values = [float(value) for value in fp32_logits.tolist()]
        selected = select_label(fp32_values, labels)
        score_rows.append({
            "id": row["id"],
            "scene_id": row["scene_id"],
            "evidence_id": row["evidence_id"],
            "candidate_id": row["candidate_id"],
            "split": row["split"],
            "role": row["role"],
            "selected_token": selected["token"],
            "predicted_truth_label": selected["truthLabel"],
            "gold_truth_label": row["target"]["truth_label"],
            "fp32_direct_logits": {
                label["token"]: fp32_values[position] for position, label in enumerate(labels)
            },
            "bf16_native_logits": {
                label["token"]: bf16_values[position] for position, label in enumerate(labels)
            },
        })
        prompt_lengths.append(len(tokens))
        observed_dtypes = {
            "hidden": str(hidden.dtype),
            "bf16_native_logits": str(bf16_logits.dtype),
            "dequantized_label_rows": str(label_rows.dtype),
            "fp32_direct_logits": str(fp32_logits.dtype),
        }
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"v26 native decoder: scored {index}/{len(rows)} prompts", file=sys.stderr, flush=True)
        mx.clear_cache()

    scored = {(row["scene_id"], row["evidence_id"]): row for row in score_rows}
    v25_lock = json.loads((PROJECT_ROOT / lock["source"]["v25_lock"]).read_text())
    v24_lock = json.loads((PROJECT_ROOT / v25_lock["source"]["v24_lock"]).read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    predictions = []
    for scene in scenes:
        prediction_rows = []
        for evidence in scene["agent_input"]["evidence"]:
            value = scored[(scene["id"], evidence["id"])]
            prediction_rows.append({
                "evidence_id": evidence["id"],
                "candidate_id": value["candidate_id"],
                "truth_label": value["predicted_truth_label"],
                "selected_token": value["selected_token"],
                "fp32_direct_logits": value["fp32_direct_logits"],
            })
        validate_scene_prediction(scene, prediction_rows)
        predictions.append({
            "scene_id": scene["id"], "episode_id": scene["episode_id"],
            "split": scene["split"], "role": scene["role"], "rows": prediction_rows,
            "epistemic_state": predicted_epistemic_rows(scene, prediction_rows),
        })
    grounding = grounding_summary(scenes, predictions)
    prediction_lookup = {row["scene_id"]: row for row in predictions}
    records = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "records")
    evaluation_records = [row for row in records if row["split"] == "grounding_evaluation"]
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())
    integration = {}
    for condition in lock["integration_conditions"]:
        support_mode, query_mode = condition_modes(condition)
        integration[condition] = integration_condition(
            evaluation_records, support_mode, query_mode, prediction_lookup,
            v22_config, original_lock["config_payload"],
        )
    checks = gate_checks(grounding, integration, lock["gates"]["development"])
    passed = all(checks.values())
    if passed:
        decision = "authorize_fresh_relational_surface_benchmark_design"
        interpretation = (
            "The fixed V24 matcher plus full-depth native truth decoder clears every exposed-data gate."
        )
    elif not checks["evaluation_truth"]:
        decision = "native_truth_decoder_insufficient_pivot_grounder_family_no_lora"
        interpretation = (
            "The frozen model's native full-depth decoder does not reliably ground declared truth semantics. "
            "Stop frozen readout variants and compare a declared parser or separately justified learned grounder."
        )
    else:
        decision = "repair_exact_graph_or_symbolic_composition_no_lora"
        interpretation = (
            "Native truth semantics pass, but exact graph assembly or symbolic integration remains below gate."
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    scores_path = output_dir / "native-decoder-scores.jsonl"
    scores_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in score_rows
    ))
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions
    ))
    result = {
        "schema_version": 26,
        "experiment": lock["experiment"],
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "evaluation_number": 1,
        "model": specification,
        "label_token_ids": {row["token"]: token_ids[index] for index, row in enumerate(labels)},
        "observed_dtypes": observed_dtypes,
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "truncated_prompts": 0,
        "grounding": grounding,
        "integration": integration,
        "checks": checks,
        "passed": passed,
        "decision": decision,
        "interpretation": interpretation,
        "selected_token_counts_by_split": {
            split: dict(sorted(Counter(
                row["selected_token"] for row in score_rows if row["split"] == split
            ).items()))
            for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation")
        },
        "native_decoder_scores": str(scores_path.relative_to(PROJECT_ROOT)),
        "native_decoder_scores_sha256": file_sha256(scores_path),
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "lora_authorized": False,
        "final_suite_constructed": False,
        "data_access": {
            "model_forward_passes": len(rows),
            "head_fits": 0,
            "threshold_fits": 0,
            "integration_evaluations": 1,
            "hyperparameter_selections": 0,
            "adapter_training_runs": 0,
            "fresh_benchmark_records_read": 0,
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
