#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v187-clean-typed-clarification-planner.json"
    plan_path = PROJECT_ROOT / "docs/v187-clean-typed-clarification-planner-plan.md"
    protocol_path = PROJECT_ROOT / "python/v187_clean_typed_clarification_planner.py"
    tests_path = PROJECT_ROOT / "python/test_v187_clean_typed_clarification_planner.py"
    runner_path = PROJECT_ROOT / "python/run_v187_clean_typed_clarification_planner.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v187_clean_typed_clarification_planner_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v187_clean_typed_clarification_planner.py"
    audit_path = PROJECT_ROOT / "outputs/v187-clean-typed-clarification-planner/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v187-clean-typed-clarification-planner-lock.json"
    output_root = PROJECT_ROOT / "outputs/v187-clean-typed-clarification-planner/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v187-clean-typed-clarification-planner-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V187 is already preregistered, evaluated, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV186OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    source_paths = {
        "question_codebook": PROJECT_ROOT / config["questionCodebook"],
        "contract_answer_vectors": PROJECT_ROOT / config["contractAnswerVectors"],
        "development_bindings": PROJECT_ROOT / config["developmentBindings"],
        "protected_bindings": PROJECT_ROOT / config["protectedBindings"],
    }
    problem = config["problem"]
    gates = config["developmentGates"]
    decision = config["decisionRule"]
    exposure = config["preLockExposure"]
    required_policies = {
        "exact_adaptive_expected_cost", "best_fixed_open_loop_sequence_with_singleton_early_stopping",
        "adaptive_greedy_weighted_information_gain", "frozen_source_order_with_generic_fallback",
        "always_generic_trusted_clarification", "immediate_safe_deferral",
        "target_informed_minimum_question_oracle",
    }
    checks = {
        "V186_is_valid_identifying_and_authorizes_planner_preregistration": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_feasibility_gates_passed"]
            and parent["authorization"]["preregister_clean_exact_planner_comparison"]
            and not parent["authorization"]["score_planner_without_separate_lock"]
            and not parent["authorization"]["read_utterance_or_protected_language_run_model_API_or_training"]
        ),
        "V186_source_artifacts_are_exact": bool(
            file_sha256(source_paths["question_codebook"]) == parent["question_codebook_sha256"]
            and file_sha256(source_paths["contract_answer_vectors"]) == parent["contract_answer_vectors_sha256"]
            and file_sha256(source_paths["development_bindings"]) == parent["development_bindings_sha256"]
            and file_sha256(source_paths["protected_bindings"]) == parent["protected_bindings_sha256"]
        ),
        "clean_problem_costs_horizon_and_terminal_authority_are_fixed": bool(
            problem["initialVersionSpaceIsAll14Contracts"]
            and problem["developmentPriorIsObservedDevelopmentTargetFrequency"]
            and problem["all14ContractsMustHavePositivePrior"]
            and problem["protectedSuccessorMustReuseFrozenDevelopmentPrior"]
            and problem["maximumTypedQuestionCount"] == 4
            and problem["typedQuestionCost"] == 0.10
            and problem["genericTrustedClarificationCost"] == 0.40
            and problem["safeDeferralCost"] == 0.50
            and problem["onlySingletonOrGenericMayProduceExactTerminalState"]
            and problem["questionPartitionsWithIdentical14ContractColumnsDeduplicatedByFirstFrozenQuestion"]
        ),
        "all_required_operational_and_oracle_controls_are_frozen": set(config["policies"]) == required_policies,
        "scientific_and_safety_gates_are_noncompensatory": bool(
            gates["requiredDevelopmentBindingCount"] == 132
            and gates["requiredObservedDevelopmentCount"] == 120
            and gates["requiredMissingDevelopmentCount"] == 12
            and gates["requiredObservedFinalExactnessRate"] == 1.0
            and gates["requiredAuthoritativeTargetRetentionRate"] == 1.0
            and gates["minimumImprovementOverAlwaysGeneric"] > 0
            and gates["minimumImprovementOverBestOpenLoop"] > 0
            and gates["maximumProtectedUtteranceLanguageReadCount"] == 0
        ),
        "prelock_and_successor_authority_are_closed": bool(
            all(value == 0 for value in exposure.values())
            and not decision["passAuthorizesImmediateErrorStressRun"]
            and not decision["passAuthorizesProtectedLanguageOrModelAccess"]
            and not decision["passAuthorizesRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, parent_path, plan_path, protocol_path, tests_path,
                runner_path, verifier_path, auditor_path, *source_paths.values(),
            )) and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "187-clean-typed-clarification-planner-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V187_clean_development_evaluation" if passed else "reject_V187_design",
        "checks": checks,
        "prelock_exposure": exposure,
        "policy_score_count": 0,
        "protected_utterance_language_read_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_V186_outcome": parent_path,
        **source_paths,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "187-clean-typed-clarification-planner-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_prior_costs_horizon_policies_gates_or_decision": False,
            "run_clean_development_evaluation_once": True,
            "read_protected_or_utterance_language": False,
            "run_error_stress_model_API_or_training": False,
            "register_mutate_call_service_act_or_execute": False,
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
