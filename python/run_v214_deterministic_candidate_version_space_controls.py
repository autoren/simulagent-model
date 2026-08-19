#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v214_deterministic_candidate_version_space_controls import (
    audit_controls,
    reconstruct_development_subsplit,
    score_controls,
)
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v214-deterministic-candidate-version-space-controls-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V214 design lock or dependency hash mismatch")
    if not lock["authorization"]["run_one_development_only_deterministic_control_study"]:
        raise RuntimeError("V214 deterministic study is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if any(path.exists() for path in artifacts.values()):
        raise RuntimeError("V214 formal output already exists")
    v213_config = json.loads((PROJECT_ROOT / lock["parent_V213_config"]).read_text())
    semantics_path = PROJECT_ROOT / lock["parent_public_semantics"]
    semantics = json.loads(semantics_path.read_text())
    fit_public, fit_truth, evaluation_public, evaluation_truth, subsplit = reconstruct_development_subsplit(
        v213_config, config, semantics
    )
    write_json(artifacts["developmentSubsplit"], subsplit)
    write_jsonl(artifacts["fitPublic"], fit_public)
    write_jsonl(artifacts["fitTruth"], fit_truth)
    write_jsonl(artifacts["evaluationPublic"], evaluation_public)
    write_jsonl(artifacts["evaluationTruthSealed"], evaluation_truth)
    del evaluation_truth

    worker_path = PROJECT_ROOT / lock["control_worker"]
    subprocess.run(
        [
            sys.executable,
            str(worker_path),
            "--config",
            str(PROJECT_ROOT / lock["config"]),
            "--semantics",
            str(semantics_path),
            "--fit-public",
            str(artifacts["fitPublic"]),
            "--fit-labels",
            str(artifacts["fitTruth"]),
            "--evaluation-public",
            str(artifacts["evaluationPublic"]),
            "--predictions",
            str(artifacts["predictions"]),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    worker_source = worker_path.read_text()
    prediction_freeze = {
        "schema_version": "214-deterministic-control-prediction-freeze",
        "design_lock_sha256": file_sha256(lock_path),
        "parent_V213_outcome_sha256": file_sha256(PROJECT_ROOT / lock["parent_V213_outcome"]),
        "parent_V213_design_lock_sha256": file_sha256(PROJECT_ROOT / lock["parent_V213_design_lock"]),
        "parent_public_semantics_sha256": file_sha256(semantics_path),
        "development_subsplit_sha256": file_sha256(artifacts["developmentSubsplit"]),
        "fit_public_sha256": file_sha256(artifacts["fitPublic"]),
        "fit_truth_sha256": file_sha256(artifacts["fitTruth"]),
        "evaluation_public_sha256": file_sha256(artifacts["evaluationPublic"]),
        "evaluation_truth_sealed_sha256": file_sha256(artifacts["evaluationTruthSealed"]),
        "predictions_sha256": file_sha256(artifacts["predictions"]),
        "predictions_frozen_before_evaluation_truth_join": True,
        "evaluation_truth_joined_before_prediction_freeze": False,
        "control_worker_evaluation_truth_path_count": worker_source.count("evaluation-truth"),
        "control_worker_hidden_evaluation_field_count": sum(
            worker_source.count(token) for token in ("expected_candidate_ids", "concept_family")
        ),
    }
    write_json(artifacts["predictionFreeze"], prediction_freeze)

    predictions = json.loads(artifacts["predictions"].read_text())
    evaluation_truth = read_jsonl(artifacts["evaluationTruthSealed"])
    metrics = score_controls(
        predictions,
        evaluation_public,
        evaluation_truth,
        subsplit,
        semantics,
        prediction_freeze,
        config,
    )
    access = {
        "deterministic_control_run_count": 1,
        "v213_protected_public_access_count": 0,
        "v213_protected_truth_access_count": 0,
        "protected_group_construction_count": 0,
        "natural_language_surface_read_count": 0,
        "external_ontology_payload_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "api_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    audit = audit_controls(metrics, prediction_freeze, access, config)
    summary = {
        "schema_version": "214-deterministic-candidate-version-space-controls-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    result = {
        "schema_version": "214-deterministic-candidate-version-space-controls-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "branch": audit["branch"],
        "model_eligible": audit["model_eligible"],
        "decision": audit["decision"],
        "claim_boundary": config["claimBoundary"],
        "authorization": {
            "design_separate_bounded_local_candidate_generator": audit["model_eligible"],
            "design_next_non_model_research_stage": bool(audit["passed"] and not audit["model_eligible"]),
            "open_V213_protected_or_run_model": False,
            "read_external_payload_register_mutate_call_act_execute": False,
        },
    }
    write_json(artifacts["summary"], summary)
    write_json(artifacts["result"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    compact = {
        method: {
            key: metrics["methods"][method][key]
            for key in (
                "oracle_class_recall",
                "exact_version_space_accuracy",
                "false_class_proposal_rate",
                "average_candidate_set_size",
                "evidence_status_accuracy",
                "shadow_action_accuracy",
                "average_normalized_decision_regret",
                "residual_record_count",
                "residual_group_count",
            )
        }
        for method in metrics["methods"]
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in metrics.items() if key != "methods"}, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
