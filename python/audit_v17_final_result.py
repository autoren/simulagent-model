#!/usr/bin/env python3
"""Independent, read-only post-result audit of the completed V17r2 run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_v10_frozen import gate_report
from v10_protocol import file_sha256, read_jsonl
from v14_protocol import load_records_from_manifest
from v17_protocol import load_v17_records


def main() -> None:
    lock_path = Path("configs/v17-final-evaluation-lock.json")
    feature_metadata_path = Path("outputs/v17-final/features/metadata.json")
    result_path = Path("outputs/v17-final/evaluation/result.json")
    output_path = Path("outputs/v17-final/post-result-audit.json")
    if output_path.exists():
        raise RuntimeError(f"V17 post-result audit already exists: {output_path}")
    lock = json.loads(lock_path.read_text())
    metadata = json.loads(feature_metadata_path.read_text())
    result = json.loads(result_path.read_text())
    final_records = load_v17_records(lock)
    dev_records = load_records_from_manifest(Path(lock["source"]["v14_manifest"]))

    final_feature_path = Path(metadata["feature_artifact"])
    dev_feature_path = Path(lock["source"]["v15_feature_artifact"])
    with np.load(final_feature_path, allow_pickle=False) as values:
        final_base = set(values["base_prompts"].tolist())
        final_nli = set(values["nli_prompts"].tolist())
    with np.load(dev_feature_path, allow_pickle=False) as values:
        dev_base = set(values["base_prompts"].tolist())
        dev_nli = set(values["nli_prompts"].tolist())

    final_actions = {record["agent_input"]["candidate_action"] for record in final_records}
    dev_actions = {record["agent_input"]["candidate_action"] for record in dev_records}
    final_evidence = {unit["text"] for record in final_records for unit in record["evidence_units"]}
    dev_evidence = {unit["text"] for record in dev_records for unit in record["evidence_units"]}
    final_hypotheses = {
        statement
        for record in final_records
        for pair in record["agent_input"]["state_hypotheses"]
        for statement in pair["statements"]
    }
    dev_hypotheses = {
        statement
        for record in dev_records
        for pair in record["agent_input"]["state_hypotheses"]
        for statement in pair["statements"]
    }
    recomputed_gates = gate_report(result["template_folds"], lock["gates"])
    evaluator_source = Path("python/evaluate_v17_final_mechanic.py").read_text()
    fit_patterns = [
        "match_model.fit(dev_base, dev_match)",
        "temporal_model.fit(dev_base[positive_train], dev_temporal[positive_train])",
        "polarity_model.fit(dev_polarity[current_train], dev_unique_current[current_train])",
    ]
    checks = {
        "evaluation_lock_hash_matches": result["evaluation_lock_sha256"] == file_sha256(lock_path),
        "sealed_dataset_hash_matches": result["dataset_sha256"] == lock["dataset_sha256"],
        "final_feature_hash_matches": metadata["feature_artifact_sha256"] == file_sha256(final_feature_path),
        "development_feature_hash_matches": file_sha256(dev_feature_path) == lock["source"]["v15_feature_artifact_sha256"],
        "head_hash_matches": result["head_artifact_sha256"] == file_sha256(Path(result["head_artifact"])),
        "result_is_first_and_only_final_evaluation": result["final_evaluation_number"] == 1,
        "exactly_three_development_fits": result["development_linear_fits"] == 3,
        "no_final_records_used_for_training": result["training_final_records"] == 0,
        "no_retry_authorized": result["final_retry_authorized"] is False,
        "no_lora_authorized": result["lora_authorized"] is False,
        "all_fit_calls_are_explicitly_development_only": all(pattern in evaluator_source for pattern in fit_patterns)
            and ".fit(final" not in evaluator_source,
        "zero_exact_base_prompt_overlap": not final_base.intersection(dev_base),
        "zero_exact_nli_prompt_overlap": not final_nli.intersection(dev_nli),
        "zero_candidate_action_overlap": not final_actions.intersection(dev_actions),
        "all_final_evidence_language_is_supported": final_evidence.issubset(dev_evidence),
        "all_final_hypothesis_language_is_supported": final_hypotheses.issubset(dev_hypotheses),
        "gates_recompute_exactly": recomputed_gates == result["final_gates"],
        "all_preregistered_gates_pass": recomputed_gates["passed"] is True,
        "final_records_match_sealed_cardinality": len(final_records) == lock["expected"]["records"],
        "protected_firewall_counts_are_zero": all(
            result["data_access"][key] == 0
            for key in (
                "v3_test_records_read", "prior_holdout_records_read", "v7_tone_drift_records_read",
                "v7_model_results_read", "untouched_v8_mechanic_records_read",
            )
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 17,
        "experiment": "v17r2_read_only_post_result_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "interpretation": {
            "final_candidate_actions": sorted(final_actions),
            "development_candidate_action_overlaps": sorted(final_actions.intersection(dev_actions)),
            "exact_base_prompt_overlaps": len(final_base.intersection(dev_base)),
            "exact_nli_prompt_overlaps": len(final_nli.intersection(dev_nli)),
            "supported_final_evidence_text_fraction": len(final_evidence.intersection(dev_evidence)) / len(final_evidence),
            "supported_final_hypothesis_text_fraction": len(final_hypotheses.intersection(dev_hypotheses)) / len(final_hypotheses),
            "claim_boundary": "unseen candidate action and transition table with fully supported evidence and state-hypothesis language",
        },
        "artifacts": {
            "evaluation_lock_sha256": file_sha256(lock_path),
            "dataset_sha256": lock["dataset_sha256"],
            "feature_artifact_sha256": file_sha256(final_feature_path),
            "head_artifact_sha256": file_sha256(Path(result["head_artifact"])),
            "result_sha256": file_sha256(result_path),
        },
        "recomputed_gates": recomputed_gates,
    }
    if not payload["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"V17 post-result audit failed: {failed}")
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
