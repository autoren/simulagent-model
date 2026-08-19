#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "repair_config": PROJECT_ROOT / "configs/v211r1-compositional-baseline-name-repair.json",
        "plan": PROJECT_ROOT / "docs/v211r1-compositional-baseline-name-repair-plan.md",
        "protocol": PROJECT_ROOT / "python/v211r1_compositional_baseline_name_repair.py",
        "tests": PROJECT_ROOT / "python/test_v211r1_compositional_baseline_name_repair.py",
        "prediction_worker": PROJECT_ROOT / "python/v211r1_prediction_worker.py",
        "runner": PROJECT_ROOT / "python/run_v211r1_compositional_baseline_name_repair.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v211r1_compositional_baseline_name_repair_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v211r1_compositional_baseline_name_repair.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v211r1-compositional-baseline-name-repair/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v211r1-compositional-baseline-name-repair-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v211r1-compositional-baseline-name-repair-outcome-lock.json"
    output = PROJECT_ROOT / "outputs/v211r1-compositional-baseline-name-repair/evaluation"
    if any(path.exists() for path in (audit_path, lock_path, outcome_path, output)): raise RuntimeError("V211r1 exists")
    config = json.loads(paths["repair_config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV211DesignLock"]
    failure_path = PROJECT_ROOT / config["parentV211TechnicalFailure"]
    parent = json.loads(parent_path.read_text()); failure = json.loads(failure_path.read_text())
    repair = config["repair"]
    checks = {
        "parent_lock_valid_and_failed_result_preserved": bool(valid_lock(parent) and failure["design_lock_sha256"] == file_sha256(parent_path) and not json.loads((PROJECT_ROOT / parent["config_payload"]["artifacts"]["result"]).read_text())["passed"]),
        "failure_is_exact_obsolete_prediction_key": failure["cause"].endswith("COMPOSITIONAL_RESPONSE_SPAN"),
        "repair_changes_only_key_and_no_values_science_or_gates": bool(repair["oldEmittedKey"] == "CONTEXT_CONTRAST" and repair["newEmittedKey"] == "COMPOSITIONAL_RESPONSE_SPAN" and repair["maximumChangedPredictionValueCount"] == repair["maximumChangedSplitOrLearnedLexiconCount"] == repair["maximumChangedMetricGateOrDecisionCount"] == 0),
        "repair_reuses_frozen_inputs_and_prediction_worker_has_no_truth_or_group": bool(all((PROJECT_ROOT / config[key]).is_file() for key in ("parentPredictionFreeze", "parentLearnedLexicon", "parentEvaluationSurface", "parentEvaluationTruthSealed", "parentSplit")) and "evaluation-truth" not in paths["prediction_worker"].read_text() and "group_id" not in paths["prediction_worker"].read_text()),
        "authorization_is_one_prediction_rescore_without_refit_model_or_protected": bool(config["authorization"]["rerunCorrectedPredictionAndScoringOnce"] and not config["authorization"]["refitBaselineOrReadCalibrationAgain"] and not config["authorization"]["readProtectedOrRunModel"]),
        "required_files_exist_and_output_absent": all(path.is_file() for path in (*paths.values(), parent_path, failure_path)),
    }
    passed = all(checks.values())
    audit = {"schema_version": "211r1-compositional-baseline-name-repair-design-audit", "experiment": config["experiment"], "passed": passed, "decision": "freeze_and_authorize_one_V211r1_prediction_rescore" if passed else "reject_V211r1", "checks": checks}
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2)); raise SystemExit(1)
    dependencies = {**paths, "parent_V211_design_lock": parent_path, "parent_V211_technical_failure": failure_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "211r1-compositional-baseline-name-repair-lock", "experiment": config["experiment"], "repair_config_payload": config, "authorization": {"modify_parent_science_or_predictions": False, "run_one_corrected_prediction_rescore": True, "read_protected_or_run_model": False}}
    for key, path in dependencies.items(): lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
