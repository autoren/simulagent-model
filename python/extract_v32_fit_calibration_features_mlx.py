#!/usr/bin/env python3
"""Extract the one locked V32 frozen fit/calibration representation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_structured_model import features_from_hidden, prompt_tokens_and_entity_spans, target_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v32-factorized-semantics-lock.json")
    parser.add_argument("--output-dir", default="outputs/v32-factorized-semantics/fit-calibration-features")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    lock_path, output_dir = (PROJECT_ROOT / args.lock).resolve(), (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "fit-calibration-feature-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V32 fit/calibration feature extraction was already attempted")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["featureExtractions"] != 1:
        raise RuntimeError("V32 lock does not authorize one feature extraction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V32 locked implementation changed: {path}")
    for name, expected in lock["source"]["corpus_file_sha256"].items():
        if file_sha256(PROJECT_ROOT / lock["source"]["corpus"] / name) != expected:
            raise RuntimeError(f"V32 corpus changed after lock: {name}")
    config = lock["config_payload"]
    rows = sorted(read_rows(
        PROJECT_ROOT / lock["source"]["corpus"], ("factor_fit", "factor_calibration")
    ), key=lambda row: row["id"])
    if len(rows) != lock["planned_training"]["fit_calibration_records"]:
        raise RuntimeError("V32 fit/calibration population differs from lock")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 32, "attempt_number": 1, "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_model_load", "evaluation_records_read": 0,
    }, indent=2, sort_keys=True) + "\n")
    specification = config["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != specification["totalLayers"] or text_config["hidden_size"] != specification["hiddenSize"]:
        raise RuntimeError("V32 model architecture differs from lock")
    maximum = max(config["construction"]["entityCounts"])
    clauses, entities, masks, targets, lengths, prompt_hashes = [], [], [], [], [], []
    for index, row in enumerate(rows, start=1):
        tokens, spans, content = prompt_tokens_and_entity_spans(row, config, tokenizer)
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError(f"V32 prompt exceeds maximum: {row['id']} ({len(tokens)})")
        hidden = model.language_model.model(mx.array([tokens]))[0]
        clause, entity, mask = features_from_hidden(hidden, spans, maximum)
        mx.eval(clause, entity, mask)
        clauses.append(np.asarray(clause, dtype=np.float32))
        entities.append(np.asarray(entity, dtype=np.float32))
        masks.append(np.asarray(mask, dtype=bool))
        targets.append(target_arrays(row, config))
        lengths.append(len(tokens))
        prompt_hashes.append(hashlib.sha256(content.encode()).hexdigest())
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"v32 frozen features: {index}/{len(rows)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "features.npz"
    np.savez_compressed(
        artifact, record_ids=np.asarray([row["id"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]), clause_features=np.stack(clauses),
        entity_features=np.stack(entities), entity_mask=np.stack(masks),
        **{f"{key}_targets": np.asarray([row[key] for row in targets], dtype=np.int32) for key in targets[0]},
    )
    metadata = {
        "schema_version": 32, "experiment": "v32_fit_calibration_features",
        "protocol_lock_sha256": file_sha256(lock_path), "feature_extraction_number": 1,
        "records": len(rows), "fit_records": sum(row["split"] == "factor_fit" for row in rows),
        "calibration_records": sum(row["split"] == "factor_calibration" for row in rows),
        "model": specification, "minimum_prompt_tokens": min(lengths), "maximum_prompt_tokens": max(lengths),
        "truncated_prompts": 0, "prompt_payload_sha256": hashlib.sha256("".join(prompt_hashes).encode()).hexdigest(),
        "feature_artifact": str(artifact.relative_to(PROJECT_ROOT)), "feature_artifact_sha256": file_sha256(artifact),
        "data_access": {"model_forward_passes": len(rows), "evaluation_records_read": 0, "evaluation_features_read": 0, "training_runs": 0},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "metadata_sha256": file_sha256(metadata_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__": main()
