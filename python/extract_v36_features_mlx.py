#!/usr/bin/env python3
"""Run exactly three frozen representation forwards for each sealed V36 clause."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

from extract_v10_features_mlx import chat_prompt
from extract_v22r2_relational_features_mlx import token_spans
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_language import representation_prompt_layout
from v34_operation import operation_prompt
from v35_binding import atom_prompt_layout


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", default="configs/v36-confirmation-seal.json")
    parser.add_argument("--output-dir", default="outputs/v36-independent-confirmation/features")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    seal_path, output_dir = (PROJECT_ROOT / args.seal).resolve(), (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "feature-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V36 feature extraction was already attempted")
    seal = json.loads(seal_path.read_text())
    if not seal["authorization"]["feature_extraction"] or seal["authorization"]["feature_extractions"] != 1:
        raise RuntimeError("V36 seal does not authorize one extraction")
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V36 locked implementation changed: {path}")
    corpus_path = PROJECT_ROOT / seal["corpus_artifact"]
    if file_sha256(corpus_path) != seal["corpus_artifact_sha256"]:
        raise RuntimeError("V36 sealed corpus changed")
    rows = sorted(jsonl(corpus_path), key=lambda row: row["id"])
    if len(rows) * 3 != seal["authorization"]["backbone_forward_passes"]:
        raise RuntimeError("V36 forward budget differs from sealed population")
    v32_config = implementation["v32_config_payload"]
    v34_config = implementation["v34_config_payload"]
    v35_config = {**implementation["v35_config_payload"], "v32_config": v32_config}
    spec = v35_config["model"]
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({"schema_version": 36, "attempt_number": 1, "status": "started_before_model_load", "confirmation_seal_sha256": file_sha256(seal_path), "planned_backbone_forward_passes": len(rows) * 3}, indent=2, sort_keys=True) + "\n")
    model, tokenizer, model_config = load(spec["model"], revision=spec["revision"], return_config=True)
    model.eval(); text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != spec["totalLayers"] or text_config["hidden_size"] != spec["hiddenSize"]:
        raise RuntimeError("V36 loaded model architecture differs")
    maximum = max(v32_config["construction"]["entityCounts"])
    generic_values, operation_values, atom_values, entity_values, entity_masks = [], [], [], [], []
    lengths = {"generic": [], "operation": [], "atom": []}; hashes = {key: [] for key in lengths}
    for index, row in enumerate(rows, start=1):
        generic_content, _ = representation_prompt_layout(row, v32_config)
        generic_prompt = chat_prompt(generic_content, v32_config["model"]["systemPrompt"], tokenizer)
        generic_tokens = tokenizer.encode(generic_prompt)
        operation_content = operation_prompt(row, v34_config)
        operation_prompt_text = chat_prompt(operation_content, v34_config["model"]["systemPrompt"], tokenizer)
        operation_tokens = tokenizer.encode(operation_prompt_text)
        atom_content, character_spans = atom_prompt_layout(row, v35_config)
        atom_prompt_text = chat_prompt(atom_content, v35_config["model"]["systemPrompt"], tokenizer)
        flat = {f"{entity}|{number}": span for entity, spans in character_spans.items() for number, span in enumerate(spans)}
        atom_tokens, mapped = token_spans(atom_prompt_text, atom_content, flat, tokenizer) if flat else (tokenizer.encode(atom_prompt_text), {})
        for name, tokens, maximum_length in (
            ("generic", generic_tokens, v32_config["model"]["maxSequenceLength"]),
            ("operation", operation_tokens, v34_config["model"]["maxSequenceLength"]),
            ("atom", atom_tokens, v35_config["model"]["maxSequenceLength"]),
        ):
            if len(tokens) > maximum_length:
                raise RuntimeError(f"V36 {name} prompt exceeds maximum: {row['id']} ({len(tokens)})")
        generic_hidden = model.language_model.model(mx.array([generic_tokens]))[0, -1].astype(mx.float32)
        operation_hidden = model.language_model.model(mx.array([operation_tokens]))[0, -1].astype(mx.float32)
        atom_hidden_all = model.language_model.model(mx.array([atom_tokens]))[0].astype(mx.float32)
        atom_hidden = atom_hidden_all[-1]
        entities, mask = [], []
        for entity in row["agent_input"]["entities"]:
            indices = sorted({token for number in range(len(character_spans[entity["id"]])) for token in mapped[f"{entity['id']}|{number}"]})
            entities.append(mx.mean(atom_hidden_all[mx.array(indices)], axis=0) if indices else mx.zeros_like(atom_hidden))
            mask.append(bool(indices))
        while len(entities) < maximum:
            entities.append(mx.zeros_like(atom_hidden)); mask.append(False)
        entity_stack = mx.stack(entities)
        mx.eval(generic_hidden, operation_hidden, atom_hidden, entity_stack)
        generic_values.append(np.asarray(generic_hidden, dtype=np.float32)); operation_values.append(np.asarray(operation_hidden, dtype=np.float32)); atom_values.append(np.asarray(atom_hidden, dtype=np.float32)); entity_values.append(np.asarray(entity_stack, dtype=np.float32)); entity_masks.append(mask)
        for name, content, tokens in (("generic", generic_content, generic_tokens), ("operation", operation_content, operation_tokens), ("atom", atom_content, atom_tokens)):
            lengths[name].append(len(tokens)); hashes[name].append(hashlib.sha256(content.encode()).hexdigest())
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"v36 frozen representations: {index}/{len(rows)} clauses ({index * 3}/{len(rows) * 3} forwards)", file=sys.stderr, flush=True)
        mx.clear_cache()
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "features.npz"
    np.savez_compressed(artifact, record_ids=np.asarray([row["id"] for row in rows]), generic_hidden=np.stack(generic_values), operation_hidden=np.stack(operation_values), atom_hidden=np.stack(atom_values), atom_entity_features=np.stack(entity_values), atom_entity_mask=np.asarray(entity_masks, dtype=bool))
    metadata = {
        "schema_version": 36, "experiment": "v36_confirmation_features", "confirmation_seal": str(seal_path.relative_to(PROJECT_ROOT)), "confirmation_seal_sha256": file_sha256(seal_path),
        "records": len(rows), "representation_views": 3, "backbone_forward_passes": len(rows) * 3,
        "prompt_lengths": {name: {"minimum": min(values), "maximum": max(values)} for name, values in lengths.items()}, "truncated_prompts": 0,
        "prompt_payload_sha256": {name: hashlib.sha256("".join(values).encode()).hexdigest() for name, values in hashes.items()},
        "feature_artifact": str(artifact.relative_to(PROJECT_ROOT)), "feature_artifact_sha256": file_sha256(artifact),
        "data_access": {"confirmation_records_represented": len(rows), "confirmation_target_fields_used": 0, "backbone_forward_passes": len(rows) * 3, "interface_fit_runs": 0, "selection_runs": 0, "v32_evaluation_records_read": 0, "v28_integration_replays": 0},
    }
    metadata_path = output_dir / "metadata.json"; metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text()); attempt.update({"status": "completed", "metadata_sha256": file_sha256(metadata_path)}); attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
