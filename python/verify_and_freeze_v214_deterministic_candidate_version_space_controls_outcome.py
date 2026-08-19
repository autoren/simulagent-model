#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v214_deterministic_candidate_version_space_controls import dependency_hashes_exact
from v10_protocol import file_sha256
from v214_deterministic_candidate_version_space_controls import (
    audit_controls,
    reconstruct_development_subsplit,
    run_controls,
    score_controls,
)
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v214-deterministic-candidate-version-space-controls-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v214-deterministic-controls/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v214-deterministic-candidate-version-space-controls-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v214-deterministic-candidate-version-space-controls-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V214 outcome is already audited or frozen")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    v213_config = json.loads((PROJECT_ROOT / lock["parent_V213_config"]).read_text())
    semantics = json.loads((PROJECT_ROOT / lock["parent_public_semantics"]).read_text())
    fit_public, fit_truth, evaluation_public, evaluation_truth, subsplit = reconstruct_development_subsplit(
        v213_config, config, semantics
    )
    rebuilt_predictions = run_controls(fit_public, fit_truth, evaluation_public, semantics, config)
    freeze = json.loads(artifacts["predictionFreeze"].read_text())
    stored_predictions = json.loads(artifacts["predictions"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    metrics = score_controls(
        rebuilt_predictions,
        evaluation_public,
        evaluation_truth,
        subsplit,
        semantics,
        freeze,
        config,
    )
    rebuilt_audit = audit_controls(metrics, freeze, summary["access"], config)
    checks = {
        "design_lock_and_dependencies_exact": dependency_hashes_exact(lock),
        "development_subsplit_and_inputs_reconstruct_exactly": bool(
            json.loads(artifacts["developmentSubsplit"].read_text()) == subsplit
            and read_jsonl(artifacts["fitPublic"]) == fit_public
            and read_jsonl(artifacts["fitTruth"]) == fit_truth
            and read_jsonl(artifacts["evaluationPublic"]) == evaluation_public
            and read_jsonl(artifacts["evaluationTruthSealed"]) == evaluation_truth
        ),
        "predictions_reconstruct_exactly": stored_predictions == rebuilt_predictions,
        "prediction_freeze_hashes_and_evaluation_truth_firewall_exact": bool(
            freeze["design_lock_sha256"] == file_sha256(lock_path)
            and freeze["parent_V213_outcome_sha256"] == file_sha256(PROJECT_ROOT / lock["parent_V213_outcome"])
            and freeze["parent_V213_design_lock_sha256"] == file_sha256(PROJECT_ROOT / lock["parent_V213_design_lock"])
            and freeze["parent_public_semantics_sha256"] == file_sha256(PROJECT_ROOT / lock["parent_public_semantics"])
            and freeze["development_subsplit_sha256"] == file_sha256(artifacts["developmentSubsplit"])
            and freeze["fit_public_sha256"] == file_sha256(artifacts["fitPublic"])
            and freeze["fit_truth_sha256"] == file_sha256(artifacts["fitTruth"])
            and freeze["evaluation_public_sha256"] == file_sha256(artifacts["evaluationPublic"])
            and freeze["evaluation_truth_sealed_sha256"] == file_sha256(artifacts["evaluationTruthSealed"])
            and freeze["predictions_sha256"] == file_sha256(artifacts["predictions"])
            and freeze["predictions_frozen_before_evaluation_truth_join"]
            and not freeze["evaluation_truth_joined_before_prediction_freeze"]
            and freeze["control_worker_evaluation_truth_path_count"] == 0
            and freeze["control_worker_hidden_evaluation_field_count"] == 0
        ),
        "summary_metrics_access_and_audit_reconstruct_exactly": bool(
            summary["metrics"] == metrics
            and summary["audit"] == rebuilt_audit
            and summary["claim_boundary"] == config["claimBoundary"]
        ),
        "result_reconstructs_and_control_audit_passes": bool(
            result["passed"] == rebuilt_audit["passed"]
            and result["branch"] == rebuilt_audit["branch"]
            and result["model_eligible"] == rebuilt_audit["model_eligible"]
            and result["decision"] == rebuilt_audit["decision"]
            and rebuilt_audit["passed"]
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "214-deterministic-candidate-version-space-controls-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "branch": rebuilt_audit["branch"],
        "model_eligible": rebuilt_audit["model_eligible"],
        "decision": "freeze_verified_V214" if passed else "freeze_failed_V214_verification",
        "checks": checks,
        "metrics": metrics,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "design_lock": lock_path,
        "audit": audit_path,
        "development_subsplit": artifacts["developmentSubsplit"],
        "fit_public": artifacts["fitPublic"],
        "fit_truth": artifacts["fitTruth"],
        "evaluation_public": artifacts["evaluationPublic"],
        "evaluation_truth_sealed": artifacts["evaluationTruthSealed"],
        "predictions": artifacts["predictions"],
        "prediction_freeze": artifacts["predictionFreeze"],
        "summary": artifacts["summary"],
        "result": artifacts["result"],
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "214-deterministic-candidate-version-space-controls-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "branch": rebuilt_audit["branch"],
            "model_eligible": rebuilt_audit["model_eligible"],
            "decision": rebuilt_audit["decision"],
            "metrics": metrics,
        },
        "authorization": {
            "design_bounded_local_candidate_generator": rebuilt_audit["model_eligible"],
            "design_next_non_model_stage": not rebuilt_audit["model_eligible"],
            "open_V213_protected_or_run_model_without_separate_lock": False,
            "read_external_payload_register_mutate_call_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
