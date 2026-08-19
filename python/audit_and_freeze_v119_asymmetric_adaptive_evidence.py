#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(value: dict[str, Any]) -> bool:
    return payload_hash({key: item for key, item in value.items() if key != "lock_payload_sha256"}) == value.get("lock_payload_sha256")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v119-asymmetric-adaptive-evidence.json"
    plan_path = PROJECT_ROOT / "docs/v119-asymmetric-adaptive-evidence-plan.md"
    protocol_path = PROJECT_ROOT / "python/v119_asymmetric_adaptive_evidence.py"
    tests_path = PROJECT_ROOT / "python/test_v119_asymmetric_adaptive_evidence.py"
    runner_path = PROJECT_ROOT / "python/run_v119_asymmetric_adaptive_evidence.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v119_asymmetric_adaptive_evidence_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v119_asymmetric_adaptive_evidence.py"
    audit_path = PROJECT_ROOT / "outputs/v119-asymmetric-adaptive-evidence/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v119-asymmetric-adaptive-evidence-lock.json"
    result_path = PROJECT_ROOT / "outputs/v119-asymmetric-adaptive-evidence/simulator/result.json"
    if any(path.exists() for path in (audit_path, lock_path, result_path)): raise RuntimeError("V119 is already frozen or evaluated")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV118OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    population_path = PROJECT_ROOT / config["historicalPopulation"]
    historical_path = PROJECT_ROOT / config["historicalModelResult"]
    catalog_path = PROJECT_ROOT / config["choiceCatalog"]
    baseline_lock_path = PROJECT_ROOT / parent_lock["baseline_lock"]
    checks = {
        "V118_is_valid_and_authorizes_only_language_free_adaptive_simulator": bool(valid_lock(parent) and valid_lock(parent_lock) and parent["outcome"]["passed"] and parent["outcome"]["audit_pass"] and parent["authorization"]["preregister_language_free_adaptive_causal_simulator"] and not parent["authorization"]["run_language_or_model"] and file_sha256(parent_lock_path) == parent["analysis_lock_sha256"]),
        "asymmetric_tree_and_fixed_cost_are_explicit": bool(config["adaptiveTree"]["totalCostEveryPath"] == 0.30 and config["adaptiveTree"]["rootCandidateConfirmation"]["cost"] == 0.10 and len(config["adaptiveTree"]["ifConfirm"]["mechanisms"]) == 2 and len(config["adaptiveTree"]["otherwise"]["mechanisms"]) == 2 and config["adaptiveTree"]["modelGeneratesOrInterpretsNoObservation"]),
        "frozen_V117_stress_gates_are_preserved": bool(config["outcomeGates"]["requiredReliability"] == 0.95 and config["outcomeGates"]["maximumRequiredCorrelation"] == 0.50 and config["outcomeGates"]["maximumAwareMeanRegretEveryPriorAndRequiredCorrelation"] == parent_lock["v117_config_payload"]["outcomeGates"]["maximumAwareMeanRegretEveryPriorAndRequiredCorrelation"] and config["channel"]["sharedFailureCorrelation"] == [0.00, 0.25, 0.50, 0.75, 1.00]),
        "historical_dependencies_are_exact_parent_bound_artifacts": bool(file_sha256(population_path) == parent_lock["historical_population_sha256"] and file_sha256(historical_path) == parent_lock["historical_model_result_sha256"] and file_sha256(catalog_path) == parent_lock["choice_catalog_sha256"]),
        "complete_hypotheses_no_novel_action_and_zero_execution": bool(config["frozenBayesPolicy"]["completeSafeHypothesisUniverseAlwaysRetained"] and config["frozenBayesPolicy"]["novelAndInsufficientCannotBecomeActions"] and config["frozenBayesPolicy"]["actualExecutionCount"] == 0),
        "no_language_model_protected_induction_API_training_or_execution_authorized": bool(all(value == 0 for value in config["accessGates"].values()) and config["decisionRule"]["passAuthorizesOnlyMechanismRealizationAudit"] and not config["decisionRule"]["passAuthorizesImmediateLanguageOrModelRun"] and not config["decisionRule"]["passAuthorizesProtectedTest"] and not config["decisionRule"]["passAuthorizesSchemaInductionOrRicherPlanning"] and not config["decisionRule"]["passAuthorizesAPITrainingActionOrExecution"]),
        "all_code_exists_and_outputs_are_absent": bool(all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)) and not result_path.exists()),
    }
    passed = all(checks.values())
    audit = {"schema_version": "119-asymmetric-adaptive-evidence-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "decision": "freeze_and_authorize_one_language_free_adaptive_simulator" if passed else "reject_V119_design"}
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {"config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path, "historical_population": population_path, "historical_model_result": historical_path, "choice_catalog": catalog_path, "baseline_lock": baseline_lock_path, "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "119-asymmetric-adaptive-evidence-lock", "experiment": config["experiment"], "config_payload": config, "baseline_config_payload": parent_lock["baseline_config_payload"], "authorization": {"run_one_language_free_adaptive_simulator": True, "modify_tree_channel_priors_costs_policy_metrics_gates_or_decision": False, "inspect_language_or_raw_responses": False, "load_or_generate_with_model": False, "begin_induction_or_richer_planning": False, "grant_capability_belief_action_or_execution_authority": False}}
    for key, path in dependencies.items(): lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
