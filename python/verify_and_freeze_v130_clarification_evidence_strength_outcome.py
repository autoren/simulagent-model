#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v130_clarification_evidence_strength import quality_pass, reliability_grid


def independent_checks(result, config):
    summary = result["summary"]; gates = config["qualityGates"]; feasibility = config["feasibilityRule"]
    grid = reliability_grid(config["singleAnswerReliabilityGrid"])
    thresholds = summary["single_answer_thresholds"]
    single_route = all(value is not None and value <= feasibility["maximumSingleAnswerReliability"] for value in thresholds.values())
    required_rhos = [rho for rho in config["multiAnswerGrid"]["commonShockCorrelations"] if rho <= feasibility["maximumRequiredCommonShockCorrelation"]]
    multi_route = any(
        all(summary["minimum_answer_counts"][f"{prior['id']}@{regime}@{rho:.2f}"] is not None and summary["minimum_answer_counts"][f"{prior['id']}@{regime}@{rho:.2f}"] <= count for prior in config["priorRegimes"] for regime in config["completeClarificationChannel"]["errorRegimes"] for rho in required_rhos)
        for count in config["multiAnswerGrid"]["answerCounts"] if count <= feasibility["maximumIndependentAnswerCount"]
    )
    threshold_exact = True
    for key, threshold in thresholds.items():
        passing = [float(value) for value, metrics in summary["single_answer_grid_conditions"][key].items() if quality_pass(metrics, gates)]
        expected = min(passing) if passing else None
        threshold_exact = threshold_exact and threshold == expected
    return {
        "reliability_grid_complete": len(grid) == config["outcomeGates"]["requiredReliabilityGridPointCount"] and all(len(rows) == len(grid) for rows in summary["single_answer_grid_conditions"].values()),
        "multi_answer_grid_complete": len(summary["multi_answer_conditions"]) * len(config["multiAnswerGrid"]["answerCounts"]) == config["outcomeGates"]["requiredMultiAnswerConditionCount"],
        "every_single_threshold_found": all(value is not None for value in thresholds.values()),
        "perfect_single_answer_known_exact": summary["audit_checks"]["perfect_single_answer_known_exact"],
        "perfect_single_answer_unsupported": summary["audit_checks"]["perfect_single_answer_unsupported"],
        "complete_hypothesis_retention": summary["true_hypothesis_retention"] == config["outcomeGates"]["requiredTrueHypothesisRetention"],
        "zero_individual_pair_emission": summary["individual_pair_emission_count"] == config["outcomeGates"]["maximumIndividualPairEmissionCount"],
        "zero_execution": summary["actual_execution_count"] == config["outcomeGates"]["maximumActualExecutionCount"],
        "reported_threshold_metrics_pass": threshold_exact,
        "single_route_classification_exact": summary["single_route_feasible"] == single_route,
        "multi_route_classification_exact": summary["multi_route_feasible"] == multi_route,
        "feasibility_classification_exact": summary["feasibility_pass"] == (all(summary["audit_checks"].values()) and (single_route or multi_route)),
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v130-clarification-evidence-strength-lock.json"; result_path = PROJECT_ROOT / "outputs/v130-clarification-evidence-strength/evaluation/result.json"; doc_path = PROJECT_ROOT / "docs/v130-clarification-evidence-strength-results.md"; audit_path = PROJECT_ROOT / "outputs/v130-clarification-evidence-strength/outcome-audit.json"; outcome_path = PROJECT_ROOT / "configs/v130-clarification-evidence-strength-outcome-lock.json"; verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v130_clarification_evidence_strength_outcome.py"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V130 outcome already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V130 result document first")
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text()); dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]; independent = independent_checks(result, lock["config_payload"])
    checks = {"lock_and_dependencies_exact": payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies), "independent_audit_and_feasibility_classification": all(independent.values()), "deterministic_recomputation_recorded": result["deterministic_recomputation_exact"], "zero_language_record_model_and_execution_access": all(value == 0 for value in result["access"].values()) and result["summary"]["actual_execution_count"] == 0}
    passed = all(checks.values()); audit = {"schema_version": "130-clarification-evidence-strength-outcome-audit", "experiment": lock["config_payload"]["experiment"], "passed": passed, "checks": checks, "independent_checks": independent}; write_json(audit_path, audit)
    if not passed: raise SystemExit(1)
    feasible = bool(result["passed"]); paths = {"analysis_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {"schema_version": "130-clarification-evidence-strength-outcome-lock", "experiment": "v130_clarification_evidence_strength_outcome_lock", "outcome": {"passed": True, "audit_pass": True, "evidence_feasibility_pass": feasible, "decision": result["decision"], "summary": result["summary"]}, "authorization": {"modify_rerun_retune_or_mine_V130": False, "preregister_evidence_realization_audit": feasible, "run_language_human_model_or_protected": False, "begin_induction_or_richer_planning": False, "run_API_training_action_or_execution": False}}
    for key, path in paths.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); write_json(outcome_path, outcome); print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
