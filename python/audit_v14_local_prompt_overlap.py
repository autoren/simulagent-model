#!/usr/bin/env python3
"""Audit exact local-NLI pair overlap before the V14 model protocol is frozen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from extract_v10_features_mlx import nli_text
from v10_protocol import file_sha256
from v14_protocol import load_records_from_manifest, primary_folds, zero_shot_operator_folds


def examples(records):
    result = []
    for record_index, record in enumerate(records):
        hypotheses = {value["determinant_id"]: value["statements"] for value in record["agent_input"]["state_hypotheses"]}
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            if target["temporal_status"] != "CURRENT":
                continue
            evidence_index = next(index for index, unit in enumerate(record["evidence_units"]) if (
                unit["start"] == target["evidence_span"]["start"]
                and unit["end"] == target["evidence_span"]["end"]
            ))
            pair = tuple(
                nli_text(record, determinant_index, evidence_index, hypothesis)
                for hypothesis in hypotheses[target["determinant_id"]]
            )
            result.append({
                "record_index": record_index,
                "pair": pair,
                "target": target["current_value"],
            })
    return result


def fold_audit(fold, values, record_indices):
    training = {value["pair"] for value in values if fold["train"][value["record_index"]]}
    evaluation = {value["pair"] for value in values if fold["evaluation"][value["record_index"]]}
    overlap = training & evaluation
    conflicts = 0
    for pair in overlap:
        train_targets = {value["target"] for value in values if value["pair"] == pair and fold["train"][value["record_index"]]}
        eval_targets = {value["target"] for value in values if value["pair"] == pair and fold["evaluation"][value["record_index"]]}
        conflicts += train_targets != eval_targets
    return {
        "training_unique_pairs": len(training),
        "evaluation_unique_pairs": len(evaluation),
        "exact_pair_overlap": len(overlap),
        "conflicting_overlap": conflicts,
        "transfer_clean": len(overlap) == 0,
    }


def main() -> None:
    manifest_path = Path("data/v14/manifest.json")
    shortcut_path = Path("outputs/v14-pre-model/shortcut-audit.json")
    output_path = Path("outputs/v14-pre-model/local-prompt-overlap-audit.json")
    records = load_records_from_manifest(manifest_path)
    values = examples(records)
    pair_targets = {}
    for value in values:
        previous = pair_targets.setdefault(value["pair"], value["target"])
        if previous != value["target"]:
            raise RuntimeError("V14 has a local NLI pair with conflicting targets")
    primary = {
        fold["name"]: fold_audit(fold, values, np.asarray([value["record_index"] for value in values]))
        for fold in primary_folds(records)
    }
    diagnostics = {
        fold["name"]: fold_audit(fold, values, np.asarray([value["record_index"] for value in values]))
        for fold in zero_shot_operator_folds(records)
    }
    clean_transfer = all(value["transfer_clean"] for name, value in primary.items() if name != "context")
    result = {
        "schema_version": 14,
        "experiment": "v14_local_nli_prompt_overlap_audit",
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "shortcut_audit_sha256": file_sha256(shortcut_path),
        "eligible_record_weighted_examples": len(values),
        "unique_local_nli_pairs": len(pair_targets),
        "unique_nli_prompts": len({prompt for pair in pair_targets for prompt in pair}),
        "targets_consistent_within_unique_pairs": True,
        "primary_folds": primary,
        "zero_shot_operator_diagnostics": diagnostics,
        "context_is_non_gating_repeated_prompt_control": primary["context"]["exact_pair_overlap"] > 0,
        "all_26_transfer_folds_have_zero_exact_overlap": clean_transfer,
        "passed": clean_transfer and all(value["conflicting_overlap"] == 0 for value in primary.values()),
        "decision": "gate_unique_pairs_on_26_clean_transfer_folds_context_diagnostic_only" if clean_transfer else "stop_before_model_access",
        "data_access": {
            "v3_test_records_read": 0, "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0, "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0, "final_v9_mechanic_records_read": 0,
        },
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
