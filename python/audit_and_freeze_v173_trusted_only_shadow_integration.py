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
    config_path = PROJECT_ROOT / "configs/v173-trusted-only-shadow-integration.json"
    parent_path = PROJECT_ROOT / "configs/v172-trusted-shadow-integration-population-outcome-lock.json"
    v170_path = PROJECT_ROOT / "configs/v170-unchanged-planner-fresh-confirmation-outcome-lock.json"
    v171_path = PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-outcome-lock.json"
    planner_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    repair_path = PROJECT_ROOT / "configs/v167r1-history-action-metric-repair-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v173-trusted-only-shadow-integration-plan.md"
    protocol_path = PROJECT_ROOT / "python/v173_trusted_only_shadow_integration.py"
    tests_path = PROJECT_ROOT / "python/test_v173_trusted_only_shadow_integration.py"
    runner_path = PROJECT_ROOT / "python/run_v173_trusted_only_shadow_integration.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v173_trusted_only_shadow_integration_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v173_trusted_only_shadow_integration.py"
    audit_path = PROJECT_ROOT / "outputs/v173-trusted-only-shadow-integration/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v173-trusted-only-shadow-integration-lock.json"
    output_root = PROJECT_ROOT / "outputs/v173-trusted-only-shadow-integration/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v173-trusted-only-shadow-integration-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V173 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    v170 = json.loads(v170_path.read_text())
    v171 = json.loads(v171_path.read_text())
    planner = json.loads(planner_path.read_text())
    repair = json.loads(repair_path.read_text())
    v171_lock_path = PROJECT_ROOT / v171["confirmation_lock"]
    v171_lock = json.loads(v171_lock_path.read_text())
    states_path = PROJECT_ROOT / parent["constraint_states"]
    eligible_path = PROJECT_ROOT / parent["eligible_state_ids"]
    targets_path = PROJECT_ROOT / parent["target_cases"]
    contract = config["frozenPolicyContract"]
    original = planner["config_payload"]
    consensus = config["deterministicConsensusGate"]
    routes = config["trustedTypedRoutes"]
    population = config["populationUse"]
    gates = config["integrityAndSafetyGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V172_population_is_frozen_unscored_and_authorizes_separate_integration": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_population_gates_passed"]
            and parent["outcome"]["summary"]["integration_eligible_state_count"] == 132
            and parent["outcome"]["summary"]["target_case_count"] == 4224
            and parent["authorization"]["preregister_trusted_only_shadow_integration_on_all_eligible_states_and_targets"]
            and not parent["authorization"]["score_or_run_integration_without_separate_lock"]
            and not parent["authorization"]["select_subsample_or_tune_using_V172_outcomes"]
        ),
        "V170_planner_and_V171_sandbox_confirmations_are_exact": bool(
            valid_lock(v170)
            and v170["outcome"]["passed"]
            and v170["outcome"]["strong_confirmation"]
            and valid_lock(v171)
            and v171["outcome"]["passed"]
            and v171["outcome"]["scientific_confirmation_gates_passed"]
            and valid_lock(v171_lock)
            and file_sha256(v171_lock_path) == v171["confirmation_lock_sha256"]
        ),
        "V167_policy_contract_and_metric_repair_are_unchanged": bool(
            valid_lock(planner)
            and valid_lock(repair)
            and repair["outcome"]["passed"]
            and contract["priorKind"] == original["prior"]["kind"]
            and contract["classMass"] == original["prior"]["classMass"]
            and contract["queryCost"] == original["queryModel"]["queryCost"]
            and contract["maximumQueries"] == original["queryModel"]["maximumQueries"]
            and contract["outcomeNoise"] == original["queryModel"]["outcomeNoise"]
            and contract["terminalDecisions"] == original["terminalDecisions"]
            and contract["terminalLoss"] == original["terminalLoss"]
            and set(contract["policies"]) == set(original["policies"])
        ),
        "consensus_gate_excludes_planner_hidden_truth_and_provisional_authority": bool(
            consensus["routeAliasOnlyIfEveryCandidateClassIsAlias"]
            and consensus["routeCompositionOnlyIfEveryCandidateClassIsComposition"]
            and consensus["deferIfCandidateClassesAreMixed"]
            and consensus["deferIfEveryCandidateClassIsProvisionalPrimitive"]
            and not consensus["plannerTerminalDecisionCanAuthorizeCommit"]
            and not consensus["hiddenTargetCanAuthorizeOperationalCommit"]
            and not consensus["oracleComparatorIsOperational"]
            and not consensus["provisionalPrimitiveMayEnterSandbox"]
        ),
        "trusted_routes_use_only_V171_simulation_and_independent_verification": bool(
            routes["postCommitRestartVerification"]
            and routes["independentVerifierRequired"]
            and not routes["realServiceOrToolTargetExists"]
            and routes["alias"].startswith("single-entity registered")
            and routes["composition"].startswith("atomic registered multi-entity")
        ),
        "all_frozen_states_targets_and_weights_are_used_without_selection": bool(
            population["useAllEligibleStates"]
            and population["requiredEligibleStateCount"] == 132
            and population["useAllTargetCases"]
            and population["requiredTargetCaseCount"] == 4224
            and population["useFrozenClassBalancedTargetWeights"]
            and not population["selectionSubsamplingOrExclusionAfterScoringAllowed"]
            and all(
                path.is_file() and file_sha256(path) == parent[f"{key}_sha256"]
                for key, path in {
                    "constraint_states": states_path,
                    "eligible_state_ids": eligible_path,
                    "target_cases": targets_path,
                }.items()
            )
        ),
        "safety_benefit_and_strong_outcomes_are_separated_without_tuning": bool(
            gates["requiredFalseTrustedRouteProbability"] == 0.0
            and gates["requiredProvisionalSandboxEntryProbability"] == 0.0
            and gates["requiredPlannerCommitAuthorizationCount"] == 0
            and gates["requiredSandboxExactFinalState"] == 1.0
            and gates["requiredSandboxInvariantPreservation"] == 1.0
            and gates["requiredSandboxProvenanceValidity"] == 1.0
            and gates["requiredSandboxRestartVerification"] == 1.0
            and not config["benefitThresholds"]["failureInvalidatesSafety"]
            and not config["strongIntegrationThresholds"]["failureInvalidatesSafetyOrBenefit"]
            and config["decisionRule"]["retainNegativeOrMixedOutcomeWithoutTuning"]
        ),
        "prelock_and_real_effect_model_boundaries_hold": bool(
            exposure["implementationUnitTargetCount"] == 1
            and exposure["implementationUnitPolicyCount"] == 2
            and exposure["formalTargetPolicyScoreCount"] == 0
            and exposure["aggregateFormalMetricInspectionCount"] == 0
            and all(
                value == 0
                for key, value in exposure.items()
                if key not in {
                    "implementationUnitTargetCount",
                    "implementationUnitPolicyCount",
                    "formalTargetPolicyScoreCount",
                    "aggregateFormalMetricInspectionCount",
                }
            )
            and not config["decisionRule"]["passAuthorizesImmediateConfirmationRun"]
            and not config["decisionRule"]["passAuthorizesModelProvisionalRegistrationRealServiceOrExecution"]
        ),
        "required_locked_files_exist": all(
            path.is_file()
            for path in (
                config_path,
                parent_path,
                v170_path,
                v171_path,
                planner_path,
                repair_path,
                v171_lock_path,
                states_path,
                eligible_path,
                targets_path,
                plan_path,
                protocol_path,
                tests_path,
                runner_path,
                verifier_path,
                auditor_path,
            )
        ),
        "formal_output_absent_before_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "173-trusted-only-shadow-integration-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_trusted_shadow_integration" if passed else "reject_V173_design",
        "checks": checks,
        "prelock_exposure": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V172_outcome": parent_path,
        "source_V170_outcome": v170_path,
        "source_V171_outcome": v171_path,
        "source_V171_lock": v171_lock_path,
        "source_V167_planner_lock": planner_path,
        "source_V167r1_metric_repair": repair_path,
        "constraint_states": states_path,
        "eligible_state_ids": eligible_path,
        "target_cases": targets_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "173-trusted-only-shadow-integration-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "V167_config_payload": planner["config_payload"],
        "V171_config_payload": v171_lock["composed_config_payload"],
        "authorization": {
            "modify_policies_gate_routes_population_targets_weights_estimands_thresholds_or_decision": False,
            "run_formal_integration_once": True,
            "select_exclude_subsample_or_tune_after_scores": False,
            "allow_planner_hidden_target_model_or_provisional_commit_authority": False,
            "call_real_service_mutate_real_state_or_execute": False,
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
