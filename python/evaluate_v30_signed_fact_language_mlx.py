#!/usr/bin/env python3
"""Run the single frozen V30 language evaluation and conditional NLI diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
from mlx_lm import load

from audit_v30_signed_fact_language import read_rows
from extract_v10_features_mlx import chat_prompt
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_evaluation import lora_eligibility, primary_summary, truth_summary
from v30_language import (
    LABEL_TOKENS, candidate_nli_prompt, field_options, primary_field_prompt,
    select_option, v26_baseline_prompt,
)


def dequantized_label_rows(model: Any, token_ids: Sequence[int]) -> mx.array:
    embedding = model.language_model.model.embed_tokens
    indices = mx.array(list(token_ids))
    return mx.dequantize(
        embedding.weight[indices], embedding.scales[indices], embedding.biases[indices],
        group_size=embedding.group_size, bits=embedding.bits, mode=embedding.mode,
        dtype=mx.float32,
    )


def score_prompt(
    content: str, system_prompt: str, option_count: int, model: Any,
    tokenizer: Any, label_rows: mx.array, max_length: int,
) -> tuple[list[float], int, str]:
    prompt = chat_prompt(content, system_prompt, tokenizer)
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_length:
        raise RuntimeError(f"V30 prompt exceeds locked maximum: {len(tokens)}")
    hidden = model.language_model.model(mx.array([tokens]))[0, -1]
    logits = hidden.astype(mx.float32) @ label_rows[:option_count].T
    mx.eval(hidden, logits)
    values = [float(value) for value in logits.tolist()]
    dtype = str(hidden.dtype)
    mx.clear_cache()
    return values, len(tokens), dtype


def truth_options(config: dict[str, Any], method: str) -> list[dict[str, str]]:
    key = "truthLabels"
    return [
        {"token": row["token"], "value": row["truthStatus"]}
        for row in config["methods"][method][key]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v30-signed-fact-language-lock.json")
    parser.add_argument("--output-dir", default="outputs/v30-signed-fact-language/evaluation")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V30 language evaluation was already attempted")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    if (
        lock["limits"]["primaryEvaluations"] != 1
        or lock["limits"]["v26BaselineEvaluations"] != 1
        or lock["limits"]["candidateNliDiagnosticEvaluations"] != 1
        or lock["limits"]["headFits"] != 0
        or lock["limits"]["thresholdFits"] != 0
    ):
        raise RuntimeError("V30 lock does not authorize the registered frozen comparison")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V30 locked implementation changed: {path}")
    for name, expected in lock["source"]["corpus_file_sha256"].items():
        if file_sha256(PROJECT_ROOT / lock["source"]["corpus"] / name) != expected:
            raise RuntimeError(f"V30 locked corpus changed: {name}")
    audit_path = PROJECT_ROOT / lock["source"]["pre_model_audit"]
    if file_sha256(audit_path) != lock["source"]["pre_model_audit_sha256"]:
        raise RuntimeError("V30 pre-model audit changed after lock")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v30_protocol_lock":
        raise RuntimeError("V30 audit does not authorize model access")
    rows = sorted(
        read_rows(PROJECT_ROOT / lock["source"]["corpus"], tuple(config["splits"])),
        key=lambda row: row["id"],
    )

    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 30, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_model_load",
    }, indent=2, sort_keys=True) + "\n")
    specification = config["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if (
        text_config["num_hidden_layers"] != specification["totalLayers"]
        or text_config["hidden_size"] != specification["hiddenSize"]
    ):
        raise RuntimeError("V30 loaded model architecture differs from the lock")
    encoded = {
        token: tokenizer.encode(token, add_special_tokens=False) for token in LABEL_TOKENS
    }
    if any(len(values) != 1 for values in encoded.values()):
        raise RuntimeError(f"V30 labels are not single tokens: {encoded}")
    token_ids = [encoded[token][0] for token in LABEL_TOKENS]
    label_rows = dequantized_label_rows(model, token_ids)
    mx.eval(label_rows)

    primary_predictions = []
    baseline_predictions = []
    prompt_lengths = []
    hidden_dtype = None
    baseline_options = truth_options(config, "v26Baseline")
    for index, row in enumerate(rows, start=1):
        selected_fields = {}
        field_logits = {}
        field_option_rows = {}
        for field in config["methods"]["primary"]["fields"]:
            content, options = primary_field_prompt(row, field, config)
            logits, length, hidden_dtype = score_prompt(
                content, specification["primarySystemPrompt"], len(options), model,
                tokenizer, label_rows, specification["maxSequenceLength"],
            )
            selected = select_option(logits, options)
            selected_fields[field] = selected["value"]
            field_logits[field] = {
                option["token"]: logits[position] for position, option in enumerate(options)
            }
            field_option_rows[field] = options
            prompt_lengths.append(length)
        primary_predictions.append({
            "id": row["id"], "split": row["split"], "scene_id": row["scene_id"],
            "selected_fields": selected_fields, "field_logits": field_logits,
            "field_options": field_option_rows,
        })
        baseline_logits, length, hidden_dtype = score_prompt(
            v26_baseline_prompt(row), specification["v26SystemPrompt"],
            len(baseline_options), model, tokenizer, label_rows,
            specification["maxSequenceLength"],
        )
        baseline_selected = select_option(baseline_logits, baseline_options)
        baseline_predictions.append({
            "id": row["id"], "split": row["split"], "scene_id": row["scene_id"],
            "predicted_truth_status": baseline_selected["value"],
            "logits": {
                option["token"]: baseline_logits[position]
                for position, option in enumerate(baseline_options)
            },
            "options": baseline_options,
        })
        prompt_lengths.append(length)
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"v30 primary+baseline: scored {index}/{len(rows)} clauses", file=sys.stderr, flush=True)

    primary = primary_summary(rows, primary_predictions, config)
    baseline = truth_summary(rows, baseline_predictions, config)
    diagnostic_predictions = None
    diagnostic = None
    if not primary["passed"]:
        diagnostic_predictions = []
        diagnostic_options = truth_options(config, "candidateNliDiagnostic")
        for index, row in enumerate(rows, start=1):
            logits, length, hidden_dtype = score_prompt(
                candidate_nli_prompt(row), specification["nliSystemPrompt"],
                len(diagnostic_options), model, tokenizer, label_rows,
                specification["maxSequenceLength"],
            )
            selected = select_option(logits, diagnostic_options)
            diagnostic_predictions.append({
                "id": row["id"], "split": row["split"], "scene_id": row["scene_id"],
                "predicted_truth_status": selected["value"],
                "logits": {
                    option["token"]: logits[position]
                    for position, option in enumerate(diagnostic_options)
                },
                "options": diagnostic_options,
            })
            prompt_lengths.append(length)
            if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
                print(f"v30 conditional NLI: scored {index}/{len(rows)} clauses", file=sys.stderr, flush=True)
        diagnostic = truth_summary(rows, diagnostic_predictions, config)

    eligibility = lora_eligibility(primary, diagnostic, audit["passed"], config)
    if primary["passed"]:
        decision = "signed_fact_pass_authorize_one_v28_reintegration"
    elif eligibility["eligible"]:
        decision = "frozen_signed_fact_methods_fail_lora_eligible_separate_protocol_required"
    else:
        decision = "frozen_signed_fact_insufficient_repair_interface_no_lora"

    output_dir.mkdir(parents=True, exist_ok=False)
    primary_path = output_dir / "primary-predictions.jsonl"
    primary_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in primary_predictions
    ))
    baseline_path = output_dir / "v26-baseline-predictions.jsonl"
    baseline_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in baseline_predictions
    ))
    diagnostic_path: Path | None = None
    if diagnostic_predictions is not None:
        diagnostic_path = output_dir / "candidate-nli-predictions.jsonl"
        diagnostic_path.write_text("".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in diagnostic_predictions
        ))
    forward_passes = len(rows) * 5 + (len(rows) if diagnostic_predictions is not None else 0)
    result = {
        "schema_version": 30, "experiment": config["experiment"],
        "protocol_lock_sha256": file_sha256(lock_path), "evaluation_number": 1,
        "model": specification,
        "label_token_ids": dict(zip(LABEL_TOKENS, token_ids, strict=True)),
        "observed_hidden_dtype": hidden_dtype,
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths), "truncated_prompts": 0,
        "primary": primary, "v26_baseline": baseline,
        "candidate_nli_diagnostic": diagnostic,
        "candidate_nli_triggered": diagnostic is not None,
        "lora_eligibility": eligibility,
        "passed": primary["passed"], "decision": decision,
        "primary_predictions": str(primary_path.relative_to(PROJECT_ROOT)),
        "primary_predictions_sha256": file_sha256(primary_path),
        "v26_baseline_predictions": str(baseline_path.relative_to(PROJECT_ROOT)),
        "v26_baseline_predictions_sha256": file_sha256(baseline_path),
        "candidate_nli_predictions": (
            None if diagnostic_path is None else str(diagnostic_path.relative_to(PROJECT_ROOT))
        ),
        "candidate_nli_predictions_sha256": (
            None if diagnostic_path is None else file_sha256(diagnostic_path)
        ),
        "v28_integration_authorized": primary["passed"],
        "adapter_training_authorized": False,
        "data_access": {
            "model_forward_passes": forward_passes,
            "primary_evaluations": 1, "v26_baseline_evaluations": 1,
            "candidate_nli_diagnostic_evaluations": int(diagnostic is not None),
            "head_fits": 0, "threshold_fits": 0, "hyperparameter_selections": 0,
            "adapter_training_runs": 0, "v28_integration_replays": 0,
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
