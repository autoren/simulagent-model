#!/usr/bin/env python3
"""Perform the single locked V35 atom-focused frozen extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

from audit_v32_factorized_semantics import read_rows
from evaluate_v30_signed_fact_language_mlx import dequantized_label_rows
from extract_v10_features_mlx import chat_prompt
from extract_v22r2_relational_features_mlx import token_spans
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v35_binding import atom_prompt_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v35-binding-assembly-lock.json")
    parser.add_argument("--output-dir", default="outputs/v35-binding-assembly/features")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    lock_path, output_dir = (PROJECT_ROOT / args.lock).resolve(), (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "feature-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V35 feature extraction was already attempted")
    lock = json.loads(lock_path.read_text()); config = {**lock["config_payload"], "v32_config": lock["v32_config_payload"]}
    if lock["limits"]["featureExtractions"] != 1:
        raise RuntimeError("V35 lock does not authorize exactly one extraction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V35 locked implementation changed: {path}")
    for name, expected in lock["source"]["allowed_corpus_file_sha256"].items():
        if file_sha256(PROJECT_ROOT / config["sourceCorpus"] / name) != expected:
            raise RuntimeError(f"V35 allowed corpus changed: {name}")
    rows = sorted(read_rows(PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])), key=lambda row: row["id"])
    if len(rows) != lock["limits"]["backboneForwardPasses"]:
        raise RuntimeError("V35 population differs from locked forward-pass budget")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({"schema_version": 35, "attempt_number": 1, "status": "started_before_model_load", "protocol_lock_sha256": file_sha256(lock_path), "evaluation_records_read": 0}, indent=2, sort_keys=True) + "\n")
    spec = config["model"]
    model, tokenizer, model_config = load(spec["model"], revision=spec["revision"], return_config=True)
    model.eval(); text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != spec["totalLayers"] or text_config["hidden_size"] != spec["hiddenSize"]:
        raise RuntimeError("V35 loaded model architecture differs from lock")
    encoded = [tokenizer.encode(token, add_special_tokens=False) for token in config["atomInterface"]["predicateLabelTokens"]]
    if any(len(values) != 1 for values in encoded):
        raise RuntimeError(f"V35 predicate labels are not single tokens: {encoded}")
    token_ids = [values[0] for values in encoded]
    label_rows = dequantized_label_rows(model, token_ids); mx.eval(label_rows)
    maximum = max(config["v32_config"]["construction"]["entityCounts"])
    clauses, logits_values, entity_values, entity_masks, lengths, hashes = [], [], [], [], [], []
    for index, row in enumerate(rows, start=1):
        content, character_spans = atom_prompt_layout(row, config)
        prompt = chat_prompt(content, spec["systemPrompt"], tokenizer)
        flat = {f"{entity}|{number}": span for entity, spans in character_spans.items() for number, span in enumerate(spans)}
        if flat:
            tokens, mapped = token_spans(prompt, content, flat, tokenizer)
        else:
            tokens, mapped = tokenizer.encode(prompt), {}
        if len(tokens) > spec["maxSequenceLength"]:
            raise RuntimeError(f"V35 prompt exceeds maximum: {row['id']} ({len(tokens)})")
        hidden = model.language_model.model(mx.array([tokens]))[0].astype(mx.float32)
        clause = hidden[-1]; predicate_logits = clause @ label_rows.T
        evidence_entities, mask = [], []
        for entity in row["agent_input"]["entities"]:
            indices = sorted({token for number in range(len(character_spans[entity["id"]])) for token in mapped[f"{entity['id']}|{number}"]})
            evidence_entities.append(mx.mean(hidden[mx.array(indices)], axis=0) if indices else mx.zeros_like(clause))
            mask.append(bool(indices))
        while len(evidence_entities) < maximum:
            evidence_entities.append(mx.zeros_like(clause)); mask.append(False)
        entity_stack = mx.stack(evidence_entities)
        mx.eval(clause, predicate_logits, entity_stack)
        clauses.append(np.asarray(clause, dtype=np.float32)); logits_values.append(np.asarray(predicate_logits, dtype=np.float32))
        entity_values.append(np.asarray(entity_stack, dtype=np.float32)); entity_masks.append(mask)
        lengths.append(len(tokens)); hashes.append(hashlib.sha256(content.encode()).hexdigest())
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"v35 atom features: {index}/{len(rows)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "features.npz"
    np.savez_compressed(artifact, record_ids=np.asarray([row["id"] for row in rows]), splits=np.asarray([row["split"] for row in rows]), clause_features=np.stack(clauses), native_predicate_logits=np.stack(logits_values), evidence_entity_features=np.stack(entity_values), evidence_entity_mask=np.asarray(entity_masks, dtype=bool))
    metadata = {
        "schema_version": 35, "experiment": config["experiment"], "protocol_lock_sha256": file_sha256(lock_path),
        "feature_extraction_number": 1, "records": len(rows), "fit_records": sum(row["split"] == "factor_fit" for row in rows), "calibration_records": sum(row["split"] == "factor_calibration" for row in rows),
        "model": spec, "label_token_ids": dict(zip(config["atomInterface"]["predicateLabelTokens"], token_ids, strict=True)),
        "minimum_prompt_tokens": min(lengths), "maximum_prompt_tokens": max(lengths), "truncated_prompts": 0,
        "prompt_payload_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
        "feature_artifact": str(artifact.relative_to(PROJECT_ROOT)), "feature_artifact_sha256": file_sha256(artifact),
        "data_access": {"backbone_forward_passes": len(rows), "fit_records_read": sum(row["split"] == "factor_fit" for row in rows), "calibration_records_read": sum(row["split"] == "factor_calibration" for row in rows), "v32_evaluation_records_read": 0, "adapter_training_runs": 0},
    }
    metadata_path = output_dir / "metadata.json"; metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text()); attempt.update({"status": "completed", "metadata_sha256": file_sha256(metadata_path)}); attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
