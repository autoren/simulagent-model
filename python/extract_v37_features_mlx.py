#!/usr/bin/env python3
"""Extract the one locked set of V37 candidate and direct semantic views."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load

from evaluate_v30_signed_fact_language_mlx import dequantized_label_rows
from extract_v10_features_mlx import chat_prompt
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_language import representation_prompt_layout
from v34_operation import operation_prompt
from v37_language import candidate_prompt


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", default="configs/v37-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v37-semantic-invariance/features")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.seal).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "feature-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V37 feature extraction was already attempted")
    seal = json.loads(seal_path.read_text())
    if seal["authorization"]["feature_extraction_attempts"] != 1:
        raise RuntimeError("V37 seal does not authorize exactly one extraction")
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    if seal["implementation_lock_sha256"] != file_sha256(implementation_path):
        raise RuntimeError("V37 seal does not bind the implementation")
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V37 locked implementation changed: {path}")
    rows = []
    for name in ("fit", "validation"):
        metadata = seal["corpora"][name]
        path = PROJECT_ROOT / metadata["path"]
        if file_sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"V37 sealed {name} corpus changed")
        rows.extend(read_jsonl(path))
    rows = sorted(rows, key=lambda row: row["id"])
    config = implementation["config_payload"]
    expected_forwards = len(rows) * config["interfaces"]["viewsPerRecord"]
    if expected_forwards != seal["authorization"]["backbone_forward_passes"]:
        raise RuntimeError("V37 forward budget differs from sealed population")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 37,
        "attempt_number": 1,
        "status": "started_before_model_load",
        "corpus_seal_sha256": file_sha256(seal_path),
        "planned_backbone_forward_passes": expected_forwards,
    }, indent=2, sort_keys=True) + "\n")

    spec = config["model"]
    model, tokenizer, model_config = load(spec["model"], revision=spec["revision"], return_config=True)
    model.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != spec["totalLayers"] or text_config["hidden_size"] != spec["hiddenSize"]:
        raise RuntimeError("V37 loaded model architecture differs from lock")
    answer_tokens = config["interfaces"]["candidateAnswerTokens"]
    encoded = [tokenizer.encode(token, add_special_tokens=False) for token in answer_tokens]
    if any(len(values) != 1 for values in encoded):
        raise RuntimeError(f"V37 candidate answer labels are not single tokens: {encoded}")
    token_ids = [values[0] for values in encoded]
    answer_rows = dequantized_label_rows(model, token_ids)
    mx.eval(answer_rows)
    yes_index = answer_tokens.index("Yes")
    no_index = answer_tokens.index("No")

    values = {
        "direct_sign_hidden": [],
        "direct_operation_hidden": [],
        "sign_candidate_hidden": [],
        "sign_candidate_margin": [],
        "operation_candidate_hidden": [],
        "operation_candidate_margin": [],
    }
    lengths: dict[str, list[int]] = {key: [] for key in (
        "direct_sign", "direct_operation", "sign_candidate", "operation_candidate"
    )}
    hashes: dict[str, list[str]] = {key: [] for key in lengths}
    started = time.perf_counter()

    def encode_content(content: str, prompt_kind: str, system_prompt: str) -> list[int]:
        prompt = chat_prompt(content, system_prompt, tokenizer)
        tokens = tokenizer.encode(prompt)
        if len(tokens) > spec["maxSequenceLength"]:
            raise RuntimeError(f"V37 {prompt_kind} prompt exceeds maximum ({len(tokens)})")
        lengths[prompt_kind].append(len(tokens))
        hashes[prompt_kind].append(hashlib.sha256(content.encode()).hexdigest())
        return tokens

    def hidden_for(tokens: list[int]) -> mx.array:
        hidden = model.language_model.model(mx.array([tokens]))[0, -1].astype(mx.float32)
        mx.eval(hidden)
        return hidden

    for index, row in enumerate(rows, start=1):
        generic_content, _ = representation_prompt_layout(row, implementation["v32_config_payload"])
        generic_hidden = hidden_for(encode_content(
            generic_content, "direct_sign", implementation["v32_config_payload"]["model"]["systemPrompt"]
        ))
        operation_content = operation_prompt(row, implementation["v34_config_payload"])
        direct_operation_hidden = hidden_for(encode_content(
            operation_content, "direct_operation", implementation["v34_config_payload"]["model"]["systemPrompt"]
        ))
        sign_hidden, sign_margin = [], []
        for candidate in config["interfaces"]["lexicalSignClasses"]:
            content = candidate_prompt(row, "lexical_sign", candidate)
            hidden = hidden_for(encode_content(content, "sign_candidate", spec["systemPrompt"]))
            logits = hidden @ answer_rows.T
            mx.eval(logits)
            sign_hidden.append(np.asarray(hidden, dtype=np.float32))
            sign_margin.append(float(logits[yes_index].item() - logits[no_index].item()))
        operation_hidden, operation_margin = [], []
        for candidate in config["interfaces"]["outerOperationClasses"]:
            content = candidate_prompt(row, "outer_operation", candidate)
            hidden = hidden_for(encode_content(content, "operation_candidate", spec["systemPrompt"]))
            logits = hidden @ answer_rows.T
            mx.eval(logits)
            operation_hidden.append(np.asarray(hidden, dtype=np.float32))
            operation_margin.append(float(logits[yes_index].item() - logits[no_index].item()))
        values["direct_sign_hidden"].append(np.asarray(generic_hidden, dtype=np.float32))
        values["direct_operation_hidden"].append(np.asarray(direct_operation_hidden, dtype=np.float32))
        values["sign_candidate_hidden"].append(np.stack(sign_hidden))
        values["sign_candidate_margin"].append(np.asarray(sign_margin, dtype=np.float32))
        values["operation_candidate_hidden"].append(np.stack(operation_hidden))
        values["operation_candidate_margin"].append(np.asarray(operation_margin, dtype=np.float32))
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(
                f"v37 semantic views: {index}/{len(rows)} clauses "
                f"({index * 9}/{expected_forwards} forwards)",
                file=sys.stderr, flush=True,
            )
        mx.clear_cache()

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "features.npz"
    np.savez_compressed(
        artifact,
        record_ids=np.asarray([row["id"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        **{key: np.stack(value) for key, value in values.items()},
    )
    metadata = {
        "schema_version": 37,
        "experiment": "v37_candidate_conditioned_features",
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "feature_extraction_attempt": 1,
        "records": len(rows),
        "fit_records": sum(row["split"] == "semantic_invariance_fit" for row in rows),
        "validation_records": sum(row["split"] == "semantic_invariance_validation" for row in rows),
        "views_per_record": 9,
        "backbone_forward_passes": expected_forwards,
        "answer_token_ids": dict(zip(answer_tokens, token_ids, strict=True)),
        "prompt_lengths": {
            name: {"minimum": min(items), "maximum": max(items)} for name, items in lengths.items()
        },
        "truncated_prompts": 0,
        "prompt_payload_sha256": {
            name: hashlib.sha256("".join(items).encode()).hexdigest() for name, items in hashes.items()
        },
        "feature_artifact": str(artifact.relative_to(PROJECT_ROOT)),
        "feature_artifact_sha256": file_sha256(artifact),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "target_fields_used_in_prompts": 0,
            "fit_runs": 0,
            "selection_runs": 0,
            "validation_evaluations": 0,
            "v32_evaluation_records_read": 0,
            "v28_runs": 0,
            "adapter_training_runs": 0,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "metadata_sha256": file_sha256(metadata_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
