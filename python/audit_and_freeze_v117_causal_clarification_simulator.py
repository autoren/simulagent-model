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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v117-causal-clarification-simulator.json"
    plan_path = PROJECT_ROOT / "docs/v117-causal-clarification-simulator-plan.md"
    protocol_path = PROJECT_ROOT / "python/v117_causal_clarification_simulator.py"
    tests_path = PROJECT_ROOT / "python/test_v117_causal_clarification_simulator.py"
    runner_path = PROJECT_ROOT / "python/run_v117_causal_clarification_simulator.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v117_causal_clarification_simulator_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v117_causal_clarification_simulator.py"
    audit_path = PROJECT_ROOT / "outputs/v117-causal-clarification-simulator/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v117-causal-clarification-simulator-lock.json"
    result_path = PROJECT_ROOT / "outputs/v117-causal-clarification-simulator/simulator/result.json"
    if any(path.exists() for path in (audit_path, lock_path, result_path)):
        raise RuntimeError("V117 is already frozen or evaluated")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV116OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    population_path = PROJECT_ROOT / config["historicalPopulation"]
    historical_path = PROJECT_ROOT / config["historicalModelResult"]
    catalog_path = PROJECT_ROOT / config["choiceCatalog"]
    baseline_lock_path = PROJECT_ROOT / parent_lock["baseline_lock"]
    checks = {
        "V116_is_valid_conditional_feasibility_and_authorizes_simulator_only": bool(
            valid_lock(parent) and valid_lock(parent_lock)
            and parent["outcome"]["passed"] and parent["outcome"]["summary"]["independent_pass"]
            and not parent["outcome"]["summary"]["correlated_pass"]
            and parent["authorization"]["preregister_unprotected_simulator_benchmark"]
            and not parent["authorization"]["run_model_or_API_or_training"]
            and file_sha256(parent_lock_path) == parent["analysis_lock_sha256"]
        ),
        "two_mechanisms_and_shared_failure_channel_are_explicit": bool(
            set(config["mechanisms"]) == {"candidateConfirmation", "catalogStatus", "modelGeneratesOrInterpretsNeitherObservation"}
            and config["mechanisms"]["modelGeneratesOrInterpretsNeitherObservation"]
            and config["channel"]["marginalCorrectness"] == [0.90, 0.95, 1.00]
            and config["channel"]["sharedFailureCorrelation"] == [0.00, 0.25, 0.50, 0.75, 1.00]
            and config["planners"] == [
                {"id": "correlation_aware", "assumedCorrelation": "actual"},
                {"id": "independence_assumed", "assumedCorrelation": 0.0},
            ]
        ),
        "historical_dependencies_are_exact_parent_bound_artifacts": bool(
            file_sha256(population_path) == parent_lock["historical_population_sha256"]
            and file_sha256(historical_path) == parent_lock["historical_model_result_sha256"]
            and file_sha256(catalog_path) == parent_lock["choice_catalog_sha256"]
        ),
        "policy_retains_hypotheses_and_never_emits_novel_or_executes": bool(
            config["frozenBayesPolicy"]["completeSafeHypothesisUniverseAlwaysRetained"]
            and config["frozenBayesPolicy"]["novelAndInsufficientCannotBecomeActions"]
            and config["frozenBayesPolicy"]["actualExecutionCount"] == 0
        ),
        "no_language_model_protected_induction_API_training_or_execution_authorized": bool(
            all(value == 0 for value in config["accessGates"].values())
            and not config["decisionRule"]["passAuthorizesImmediateLanguageOrModelRun"]
            and not config["decisionRule"]["passAuthorizesProtectedTest"]
            and not config["decisionRule"]["passAuthorizesSchemaInductionOrRicherPlanning"]
            and not config["decisionRule"]["passAuthorizesAPITrainingActionOrExecution"]
        ),
        "all_code_exists_and_outputs_are_absent": bool(
            all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path))
            and not result_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {"schema_version": "117-causal-clarification-simulator-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "decision": "freeze_and_authorize_one_causal_simulator_run" if passed else "reject_V117_design"}
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path,
        "historical_population": population_path, "historical_model_result": historical_path,
        "choice_catalog": catalog_path, "baseline_lock": baseline_lock_path, "plan": plan_path,
        "protocol": protocol_path, "tests": tests_path, "runner": runner_path,
        "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {"schema_version": "117-causal-clarification-simulator-lock", "experiment": config["experiment"], "config_payload": config, "baseline_config_payload": parent_lock["baseline_config_payload"], "authorization": {"run_one_language_free_causal_simulator": True, "modify_mechanisms_channel_priors_costs_policy_metrics_gates_or_decision": False, "inspect_language_or_raw_responses": False, "load_or_generate_with_model": False, "begin_induction_or_richer_planning": False, "grant_capability_belief_action_or_execution_authority": False}}
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
