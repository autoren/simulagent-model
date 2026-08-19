#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v211_deterministic_residual_baselines import audit_scores, fit_baselines, predict_evaluation, score_predictions
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def canonical(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v211-deterministic-residual-baselines-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v211-deterministic-residual-baselines/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v211-deterministic-residual-baselines-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v211-deterministic-residual-baselines-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V211 outcome already frozen")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    split = json.loads(artifacts["split"].read_text())
    calibration_surface = read_jsonl(artifacts["calibrationSurface"])
    calibration_truth = read_jsonl(artifacts["calibrationTruth"])
    evaluation_surface = read_jsonl(artifacts["evaluationSurface"])
    evaluation_truth = read_jsonl(artifacts["evaluationTruthSealed"])
    rebuilt_learned = fit_baselines(calibration_surface, calibration_truth, config)
    rebuilt_predictions = predict_evaluation(evaluation_surface, rebuilt_learned)
    learned_exact = json.loads(artifacts["learnedLexicon"].read_text()) == rebuilt_learned
    predictions_exact = artifacts["predictions"].read_text() == canonical(rebuilt_predictions)
    freeze = json.loads(artifacts["predictionFreeze"].read_text())
    freeze_exact = bool(
        freeze["predictions_sha256"] == file_sha256(artifacts["predictions"])
        and not freeze["evaluation_truth_opened_before_freeze"]
        and freeze["prediction_worker_evaluation_truth_path_count"] == 0
        and freeze["prediction_worker_group_id_read_count"] == 0
    )
    parent_outcome = json.loads((PROJECT_ROOT / lock["reference_V209r1_outcome"]).read_text())
    repair_lock = json.loads((PROJECT_ROOT / parent_outcome["repair_lock"]).read_text())
    v209_lock = json.loads((PROJECT_ROOT / repair_lock["parent_V209_design_lock"]).read_text())
    scores = score_predictions(rebuilt_predictions, evaluation_truth, v209_lock["config_payload"])
    rebuilt_audit = audit_scores(split, rebuilt_learned, rebuilt_predictions, scores, config)
    stored_summary = json.loads(artifacts["summary"].read_text())
    summary_exact = bool(
        stored_summary["split"] == split
        and stored_summary["learned_lexicon_sizes"] == {name: len(value["token_to_label"]) for name, value in rebuilt_learned.items()}
        and stored_summary["prediction_freeze"] == freeze
        and stored_summary["scores"] == scores
        and stored_summary["model_eligible"] == rebuilt_audit["model_eligible"]
        and stored_summary["branch"] == rebuilt_audit["branch"]
    )
    result = json.loads(artifacts["result"].read_text())
    result_exact = bool(
        result["passed"] == rebuilt_audit["passed"] and result["branch"] == rebuilt_audit["branch"]
        and result["model_eligible"] == rebuilt_audit["model_eligible"] and result["decision"] == rebuilt_audit["decision"]
        and result["checks"] == rebuilt_audit["checks"] and result["access_checks"] == rebuilt_audit["access_checks"]
        and result["summary"] == stored_summary
    )
    checks = {
        "design_lock_and_dependencies_exact": dependencies_exact,
        "learned_lexicon_reconstructs_exactly": learned_exact,
        "predictions_reconstruct_exactly": predictions_exact,
        "prediction_freeze_precedes_truth_join_and_is_exact": freeze_exact,
        "summary_reconstructs_exactly": summary_exact,
        "result_reconstructs_exactly": result_exact,
        "access_and_scientific_audit_pass": rebuilt_audit["passed"],
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "211-deterministic-residual-baselines-outcome-audit",
        "experiment": lock["experiment"], "passed": passed,
        "branch": rebuilt_audit["branch"], "model_eligible": rebuilt_audit["model_eligible"],
        "decision": "freeze_verified_V211" if passed else "freeze_failed_V211_verification",
        "checks": checks, "scores": scores,
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {
        "evaluation_lock": lock_path, "audit": audit_path, "split": artifacts["split"],
        "learned_lexicon": artifacts["learnedLexicon"], "predictions": artifacts["predictions"],
        "prediction_freeze": artifacts["predictionFreeze"], "summary": artifacts["summary"],
        "result": artifacts["result"], "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "211-deterministic-residual-baselines-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {"passed": True, "branch": rebuilt_audit["branch"], "model_eligible": rebuilt_audit["model_eligible"], "decision": rebuilt_audit["decision"], "scores": scores},
        "authorization": {
            "design_new_identifiable_open_class_population": rebuilt_audit["branch"] == "ZERO_MODEL_ELIGIBILITY",
            "preregister_separate_local_model_design_only": rebuilt_audit["branch"] == "NONTRIVIAL_MODEL_ELIGIBLE_RESIDUAL",
            "open_protected_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
