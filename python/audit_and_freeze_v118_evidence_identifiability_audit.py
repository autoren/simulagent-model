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
    config_path = PROJECT_ROOT / "configs/v118-evidence-identifiability-audit.json"
    plan_path = PROJECT_ROOT / "docs/v118-evidence-identifiability-audit-plan.md"
    protocol_path = PROJECT_ROOT / "python/v118_evidence_identifiability_audit.py"
    tests_path = PROJECT_ROOT / "python/test_v118_evidence_identifiability_audit.py"
    runner_path = PROJECT_ROOT / "python/run_v118_evidence_identifiability_audit.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v118_evidence_identifiability_audit_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v118_evidence_identifiability_audit.py"
    audit_path = PROJECT_ROOT / "outputs/v118-evidence-identifiability-audit/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v118-evidence-identifiability-audit-lock.json"
    result_path = PROJECT_ROOT / "outputs/v118-evidence-identifiability-audit/audit/result.json"
    if any(path.exists() for path in (audit_path, lock_path, result_path)):
        raise RuntimeError("V118 is already frozen or evaluated")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV117OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / config["parentV117AnalysisLock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    population_path = PROJECT_ROOT / config["historicalPopulation"]
    historical_path = PROJECT_ROOT / config["historicalModelResult"]
    catalog_path = PROJECT_ROOT / config["choiceCatalog"]
    v117_result_path = PROJECT_ROOT / parent["result"]
    baseline_lock_path = PROJECT_ROOT / parent_lock["baseline_lock"]
    checks = {
        "V117_is_valid_negative_and_authorizes_no_language_or_model": bool(
            valid_lock(parent) and valid_lock(parent_lock)
            and parent["outcome"]["passed"] and not parent["outcome"]["simulator_pass"]
            and parent["outcome"]["decision"] == "causal_simulator_fails_keep_language_and_model_branch_closed"
            and not any(parent["authorization"].values())
            and file_sha256(parent_lock_path) == parent["analysis_lock_sha256"]
            and file_sha256(v117_result_path) == parent["result_sha256"]
        ),
        "audit_is_algebraic_not_retuning": bool(
            config["auditPoints"]["reliability"] == 0.95
            and config["auditPoints"]["requiredSharedFailureCorrelations"] == [0.00, 0.25, 0.50]
            and set(config["derivations"]) == {
                "exactCandidatePosteriorThresholdEnvelope", "unsupportedPosteriorThresholdEnvelope",
                "priorToPosteriorBayesFactorRequirements", "perfectPartitionCeiling",
                "frozenV117EffectiveLikelihoodRatios", "strongPriorUnsupportedDecisivePosterior",
                "minimumIndependentExactConfirmationUnits", "minimumAdditionalUnsupportedSpecificBayesFactor",
            }
        ),
        "historical_dependencies_are_exact_parent_bound_artifacts": bool(
            file_sha256(population_path) == parent_lock["historical_population_sha256"]
            and file_sha256(historical_path) == parent_lock["historical_model_result_sha256"]
            and file_sha256(catalog_path) == parent_lock["choice_catalog_sha256"]
        ),
        "no_language_model_protected_induction_API_training_or_execution_authorized": bool(
            all(value == 0 for value in config["accessGates"].values())
            and config["decisionRule"]["passAuthorizesOnlyPreregisterLanguageFreeAdaptiveCausalSimulator"]
            and not config["decisionRule"]["passAuthorizesLanguageOrModelRun"]
            and not config["decisionRule"]["passAuthorizesProtectedTest"]
            and not config["decisionRule"]["passAuthorizesSchemaInductionOrRicherPlanning"]
            and not config["decisionRule"]["passAuthorizesAPITrainingActionOrExecution"]
        ),
        "aggregate_retention_and_zero_execution_are_locked": bool(
            config["outcomeGates"]["requiredTrueHypothesisRetention"] == 1.0
            and config["outcomeGates"]["maximumIndividualRecordEmissionCount"] == 0
            and config["outcomeGates"]["maximumActualExecutionCount"] == 0
        ),
        "all_code_exists_and_outputs_are_absent": bool(
            all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path))
            and not result_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "118-evidence-identifiability-design-audit",
        "experiment": config["experiment"], "passed": passed, "checks": checks,
        "decision": "freeze_and_authorize_one_aggregate_algebraic_audit" if passed else "reject_V118_design",
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path,
        "parent_result": v117_result_path, "historical_population": population_path,
        "historical_model_result": historical_path, "choice_catalog": catalog_path,
        "baseline_lock": baseline_lock_path, "plan": plan_path, "protocol": protocol_path,
        "tests": tests_path, "runner": runner_path, "verifier": verifier_path,
        "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "118-evidence-identifiability-audit-lock", "experiment": config["experiment"],
        "config_payload": config, "v117_config_payload": parent_lock["config_payload"],
        "baseline_config_payload": parent_lock["baseline_config_payload"],
        "authorization": {
            "run_one_aggregate_algebraic_audit": True,
            "modify_thresholds_channel_reliability_priors_costs_gates_or_decision": False,
            "inspect_language_or_raw_responses": False, "load_or_generate_with_model": False,
            "begin_induction_or_richer_planning": False,
            "grant_capability_belief_action_or_execution_authority": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
