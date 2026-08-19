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
    config_path = PROJECT_ROOT / "configs/v116-typed-clarification-voi.json"
    plan_path = PROJECT_ROOT / "docs/v116-typed-clarification-voi-plan.md"
    protocol_path = PROJECT_ROOT / "python/v116_typed_clarification_voi.py"
    tests_path = PROJECT_ROOT / "python/test_v116_typed_clarification_voi.py"
    runner_path = PROJECT_ROOT / "python/run_v116_typed_clarification_voi.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v116_typed_clarification_voi_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v116_typed_clarification_voi.py"
    audit_path = PROJECT_ROOT / "outputs/v116-typed-clarification-voi/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v116-typed-clarification-voi-lock.json"
    result_path = PROJECT_ROOT / "outputs/v116-typed-clarification-voi/audit/result.json"
    if any(path.exists() for path in (audit_path, lock_path, result_path)):
        raise RuntimeError("V116 is already frozen or evaluated")

    config = json.loads(config_path.read_text())
    parent_outcome_path = PROJECT_ROOT / config["parentV115OutcomeLock"]
    parent_outcome = json.loads(parent_outcome_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent_outcome["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    population_path = PROJECT_ROOT / config["historicalPopulation"]
    model_result_path = PROJECT_ROOT / config["historicalModelResult"]
    choice_catalog_path = PROJECT_ROOT / config["choiceCatalog"]
    baseline_lock_path = PROJECT_ROOT / parent_lock["baseline_lock"]
    population = json.loads(population_path.read_text())
    result = json.loads(model_result_path.read_text())
    choices = json.loads(choice_catalog_path.read_text())
    checks = {
        "V115_is_valid_frozen_negative_with_no_induction_authority": bool(
            valid_lock(parent_outcome) and valid_lock(parent_lock)
            and parent_outcome["outcome"]["passed"]
            and not parent_outcome["outcome"]["contrastive_evidence_pass"]
            and parent_outcome["outcome"]["decision"] == "contrastive_evidence_negative_close_two_pass_single_model_branch"
            and not parent_outcome["authorization"]["begin_schema_or_capability_induction"]
            and file_sha256(parent_lock_path) == parent_outcome["analysis_lock_sha256"]
        ),
        "historical_inputs_are_exact_frozen_V115_artifacts": bool(
            file_sha256(population_path) == parent_lock["fresh_population_sha256"]
            and file_sha256(model_result_path) == parent_outcome["result_sha256"]
            and file_sha256(choice_catalog_path) == parent_lock["choice_catalog_sha256"]
            and population["selected_record_count"] == 192
            and len(result["fixtures"]) == 240
            and choices["choice_count"] == 17
        ),
        "answer_channel_priors_costs_and_reliability_are_explicit": bool(
            config["clarificationQuery"]["truthResponseUsesOnlyFrozenClassScenarioAndIntentMetadata"]
            and config["clarificationQuery"]["modelDoesNotGenerateQuestionOrAnswer"]
            and config["clarificationQuery"]["modelDoesNotInterpretAnswer"]
            and config["simulatedAnswerChannel"]["correctResponseProbabilities"] == [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
            and config["simulatedAnswerChannel"]["twoAnswerConditions"] == ["conditionally_independent", "fully_correlated"]
            and [row["candidateProbability"] for row in config["priorRegimes"]] == [1 / 17, 0.5, 0.75]
            and config["feasibilityGates"]["requiredReliability"] == 0.95
        ),
        "Bayes_policy_retains_hypotheses_and_cannot_emit_novel_or_execute": bool(
            config["frozenBayesPolicy"]["novelResponseAlwaysAbstains"]
            and config["frozenBayesPolicy"]["insufficientResponseAlwaysAbstains"]
            and config["frozenBayesPolicy"]["completeSafeHypothesisUniverseAlwaysRetained"]
            and config["frozenBayesPolicy"]["actualExecutionCount"] == 0
        ),
        "no_language_model_API_training_induction_action_or_execution_authorized": bool(
            all(value == 0 for value in config["accessGates"].values())
            and not config["decisionRule"]["passAuthorizesFreshLanguageOrModelGeneration"]
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
    audit = {
        "schema_version": "116-typed-clarification-voi-design-audit",
        "experiment": config["experiment"], "passed": passed, "checks": checks,
        "decision": "freeze_and_authorize_one_language_free_audit" if passed else "reject_V116_design",
        "prelock_access": {
            "fresh_language_read_count": 0, "protected_test_language_read_count": 0,
            "manual_language_or_raw_response_inspection_count": 0, "model_load_count": 0,
            "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_outcome": parent_outcome_path,
        "parent_analysis_lock": parent_lock_path, "historical_population": population_path,
        "historical_model_result": model_result_path, "choice_catalog": choice_catalog_path,
        "baseline_lock": baseline_lock_path, "plan": plan_path, "protocol": protocol_path,
        "tests": tests_path, "runner": runner_path, "verifier": verifier_path,
        "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "116-typed-clarification-voi-lock",
        "experiment": config["experiment"], "config_payload": config,
        "baseline_config_payload": parent_lock["baseline_config_payload"],
        "authorization": {
            "run_one_language_free_aggregate_audit": True,
            "modify_channel_priors_costs_policy_metrics_gates_or_decision": False,
            "inspect_language_or_raw_model_responses": False,
            "load_or_generate_with_model": False, "call_API_or_train_adapter": False,
            "begin_induction_or_richer_planning": False,
            "grant_capability_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
