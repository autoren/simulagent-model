#!/usr/bin/env python3
"""Run V14 shortcut gates and semantic-operator support checks before model access."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_v10_shortcuts import fit_dict_metric
from v10_protocol import file_sha256
from v14_protocol import load_records_from_manifest, primary_folds, zero_shot_operator_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v14-grounding-lock.json")
    parser.add_argument("--output", default="outputs/v14-pre-model/shortcut-audit.json")
    return parser.parse_args()


def mention_signature(record: dict[str, Any], target: dict[str, Any]) -> str:
    hypotheses = next(
        value["statements"] for value in record["agent_input"]["state_hypotheses"]
        if value["determinant_id"] == target["determinant_id"]
    )
    text = target["evidence_span"]["text"].lower()
    present = [value.lower() in text for value in hypotheses]
    gold = 0 if target["current_value"] == "active" else 1
    if present == [True, True]:
        return "both"
    if present[gold]:
        return "gold_only"
    if present[1 - gold]:
        return "opposite_only"
    return "neither"


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError(f"V14 shortcut audit already exists: {output_path}")
    lock = json.loads(lock_path.read_text())
    manifest_path = Path(lock["expected_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["grounding_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V14 corpus does not share its grounding lock")
    records = load_records_from_manifest(manifest_path)
    folds = primary_folds(records)
    diagnostics = zero_shot_operator_folds(records)

    pair_records = []
    pair_metadata = []
    pair_positions = []
    pair_match = []
    pair_current = []
    relation_pair_indices = []
    relation_positions = []
    relation_gold = []
    signature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record_index, record in enumerate(records):
        hypotheses = {value["determinant_id"]: value["statements"] for value in record["agent_input"]["state_hypotheses"]}
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            positive = next(index for index, unit in enumerate(record["evidence_units"]) if (
                unit["start"] == target["evidence_span"]["start"] and unit["end"] == target["evidence_span"]["end"]
            ))
            if target["temporal_status"] == "CURRENT":
                signature_counts[record["template_family"]][mention_signature(record, target)] += 1
            for evidence_index, _unit in enumerate(record["evidence_units"]):
                pair_index = len(pair_records)
                pair_records.append(record_index)
                base = {
                    "determinant_position": str(determinant_index),
                    "evidence_position": str(evidence_index),
                    "determinant_count": str(len(record["agent_input"]["transition_determinants"])),
                }
                pair_positions.append(base)
                pair_metadata.append({
                    **base,
                    "mechanic": record["mechanic"],
                    "transition_operator": record["operator_family"],
                    "semantic_operator": record["semantic_operator_family"],
                    "surface": record["template_family"],
                    "lexicon": record["state_lexicon_family"],
                })
                matched = evidence_index == positive
                pair_match.append(matched)
                current = target["current_value"]
                pair_current.append((1 if current == "active" else 0) if matched and current is not None else -1)
                if matched and current is not None:
                    for hypothesis_index, _statement in enumerate(hypotheses[target["determinant_id"]]):
                        relation_pair_indices.append(pair_index)
                        relation_positions.append({"hypothesis_position": str(hypothesis_index)})
                        relation_gold.append(target["hypothesis_relations"][hypothesis_index] == "ENTAILED")

    pair_records_np = np.asarray(pair_records, dtype=np.int32)
    match_np = np.asarray(pair_match, dtype=bool)
    current_np = np.asarray(pair_current, dtype=np.int8)
    relation_pairs_np = np.asarray(relation_pair_indices, dtype=np.int32)
    relation_gold_np = np.asarray(relation_gold, dtype=bool)
    audits: dict[str, dict[str, Any]] = {
        "metadata_match": {}, "position_match": {}, "metadata_polarity": {},
        "hypothesis_position_relation": {},
    }
    for fold in folds:
        train_pairs = fold["train"][pair_records_np]
        evaluation_pairs = fold["evaluation"][pair_records_np]
        audits["metadata_match"][fold["name"]] = fit_dict_metric(
            pair_metadata, match_np, train_pairs, evaluation_pairs,
        )
        audits["position_match"][fold["name"]] = fit_dict_metric(
            pair_positions, match_np, train_pairs, evaluation_pairs,
        )
        current_train = train_pairs & (current_np >= 0)
        current_evaluation = evaluation_pairs & (current_np >= 0)
        audits["metadata_polarity"][fold["name"]] = fit_dict_metric(
            pair_metadata, current_np == 1, current_train, current_evaluation,
        )
        relation_train = fold["train"][pair_records_np[relation_pairs_np]]
        relation_evaluation = fold["evaluation"][pair_records_np[relation_pairs_np]]
        audits["hypothesis_position_relation"][fold["name"]] = fit_dict_metric(
            relation_positions, relation_gold_np, relation_train, relation_evaluation,
        )

    support = {}
    surfaces = sorted(signature_counts)
    for heldout in surfaces:
        training = Counter()
        for surface in surfaces:
            if surface != heldout:
                training.update(signature_counts[surface])
        evaluation = [name for name, count in signature_counts[heldout].items() if count]
        unsupported = [name for name in evaluation if not training[name]]
        support[heldout] = {
            "semantic_operator": next(record["semantic_operator_family"] for record in records if record["template_family"] == heldout),
            "evaluation_signature_counts": dict(signature_counts[heldout]),
            "training_signature_counts": dict(training),
            "unsupported_evaluation_signatures": unsupported,
            "passed": not unsupported,
        }

    gate_config = lock["config"]["shortcutGates"]
    specifications = [
        ("metadata_match", "maximumMetadataMatchBalancedAccuracy"),
        ("position_match", "maximumPositionMatchBalancedAccuracy"),
        ("metadata_polarity", "maximumMetadataPolarityBalancedAccuracy"),
        ("hypothesis_position_relation", "maximumHypothesisPositionRelationBalancedAccuracy"),
    ]
    checks = []
    for audit_name, config_name in specifications:
        value = max(item["balanced_accuracy"] for item in audits[audit_name].values())
        maximum = gate_config[config_name]
        checks.append({
            "name": f"{audit_name}_maximum_primary_fold_balanced_accuracy",
            "value": value, "maximum": maximum, "passed": value <= maximum,
        })
    checks.append({
        "name": "all_surface_holdouts_retain_semantic_signature_support",
        "value": sum(value["passed"] for value in support.values()),
        "minimum": len(support),
        "passed": all(value["passed"] for value in support.values()),
    })
    validation = manifest["validation"]
    structural_passed = not validation["errors"] and all(validation[name] == 0 for name in (
        "malformed_spans", "malformed_hypotheses", "relation_mismatches",
        "allowed_value_derivation_mismatches", "symbolic_mismatches",
        "imbalanced_current_cells", "operator_signature_mismatches",
        "complement_cross_split_overlaps", "context_cross_split_overlaps",
        "duplicate_prompts", "cross_split_duplicate_prompts", "conflicting_duplicate_prompts",
    )) and not validation["unsupported_surface_holdouts"]
    passed = structural_passed and all(check["passed"] for check in checks)
    report = {
        "schema_version": 14,
        "experiment": "v14_pre_model_operator_support_and_shortcut_audit",
        "grounding_lock": str(lock_path),
        "grounding_lock_sha256": file_sha256(lock_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "records": len(records),
        "pair_examples": len(pair_records),
        "current_relation_examples": len(relation_gold),
        "primary_folds": [{"name": fold["name"], "kind": fold["kind"]} for fold in folds],
        "zero_shot_operator_diagnostics": [{"name": fold["name"], "kind": fold["kind"]} for fold in diagnostics],
        "surface_operator_support": support,
        "audits": audits,
        "structural_passed": structural_passed,
        "gates": {"passed": passed, "checks": checks},
        "decision": "authorize_separately_locked_v14_4b_token_mean_baseline" if passed else "stop_before_model_access",
        "data_access": lock["data_access"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
