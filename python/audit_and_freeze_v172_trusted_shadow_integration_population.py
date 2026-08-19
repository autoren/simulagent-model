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
    config_path = PROJECT_ROOT / "configs/v172-trusted-shadow-integration-population.json"
    parent_path = PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-outcome-lock.json"
    planner_path = PROJECT_ROOT / "configs/v170-unchanged-planner-fresh-confirmation-outcome-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v171.md"
    plan_path = PROJECT_ROOT / "docs/v172-trusted-shadow-integration-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v172_trusted_shadow_integration_population.py"
    tests_path = PROJECT_ROOT / "python/test_v172_trusted_shadow_integration_population.py"
    runner_path = PROJECT_ROOT / "python/run_v172_trusted_shadow_integration_population.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v172_trusted_shadow_integration_population_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v172_trusted_shadow_integration_population.py"
    audit_path = PROJECT_ROOT / "outputs/v172-trusted-shadow-integration-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v172-trusted-shadow-integration-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v172-trusted-shadow-integration-population/population"
    outcome_path = PROJECT_ROOT / "configs/v172-trusted-shadow-integration-population-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V172 is already preregistered, built, or frozen")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    planner = json.loads(planner_path.read_text())
    generator = config["generator"]
    eligibility = config["integrationEligibility"]
    targets = config["targetCases"]
    gates = config["populationGates"]
    exposure = config["preLockExposure"]
    authority = config["authorityBoundary"]
    checks = {
        "V171_is_positive_frozen_and_authorizes_only_separate_integration_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_confirmation_gates_passed"]
            and parent["authorization"]["design_trusted_only_shadow_integration"]
            and not parent["authorization"]["run_integration_without_separate_lock"]
            and not parent["authorization"]["allow_provisional_candidate_to_authorize_commit"]
        ),
        "V170_is_strong_frozen_planner_confirmation": bool(
            valid_lock(planner)
            and planner["outcome"]["passed"]
            and planner["outcome"]["scientific_integrity_passed"]
            and planner["outcome"]["strong_confirmation"]
            and not planner["authorization"]["modify_rerun_or_tune_V170"]
        ),
        "complete_three_constraint_generator_is_frozen_without_score_selection": bool(
            generator["candidateTruthTableCount"] == 256
            and generator["valuationCount"] == 8
            and generator["constraintCountPerState"] == 3
            and generator["outcomesPerConstraint"] == [0, 1]
            and generator["completeValuationTripleAndOutcomeEnumeration"]
            and generator["requiredSourceStateCount"] == 448
            and generator["retainEverySourceState"]
            and not generator["selectionUsesPlannerScoresPoliciesSandboxOutcomesOrTargetClass"]
        ),
        "eligibility_and_targets_use_only_frozen_structure": bool(
            eligibility["usesOnlyFrozenCandidateClassMetadata"]
            and eligibility["ineligibleStatesRetainedInPopulation"]
            and not eligibility["policyOrSandboxScoringDuringEligibility"]
            and targets["includeEveryCandidateInEveryEligibleVersionSpace"]
            and not targets["targetSelectionOrSubsampling"]
            and not targets["targetWeightsUsePolicyOutcomes"]
            and targets["targetIdentitiesFrozenBeforeIntegrationScoring"]
        ),
        "population_gates_freeze_exposed_structure_and_zero_integration_access": bool(
            gates["requiredSourceStateCount"] == 448
            and gates["requiredSelectedStateCount"] == 448
            and gates["requiredCandidatesPerState"] == 32
            and gates["requiredEligibleStateCount"] == 132
            and gates["requiredIneligibleStateCount"] == 316
            and gates["requiredTargetCaseCount"] == 4224
            and gates["requiredConstraintSignatureUniqueness"] == 1.0
            and gates["requiredVersionSpaceExactness"] == 1.0
            and gates["requiredTargetMembershipExactness"] == 1.0
            and gates["requiredPriorWeightNormalization"] == 1.0
            and gates["maximumPlannerPolicyScoreCount"] == 0
            and gates["maximumSandboxTransactionCount"] == 0
        ),
        "prelock_and_authority_boundaries_are_closed": bool(
            exposure["implementationPopulationBuildCount"] == 1
            and exposure["formalPopulationBuildCount"] == 0
            and all(
                value == 0
                for key, value in exposure.items()
                if key not in {"implementationPopulationBuildCount", "formalPopulationBuildCount"}
            )
            and authority["populationContainsOnlyShadowVersionSpacesAndTargetIdentities"]
            and authority["authoritativeOntologyAndStateImmutable"]
            and authority["plannerNotRun"]
            and authority["sandboxNotRun"]
            and authority["modelNotRun"]
            and not authority["provisionalRegistrationAllowed"]
            and not authority["actionOrExecutionAllowed"]
            and authority["realExecutionCount"] == 0
            and not config["decisionRule"]["passAuthorizesImmediateIntegrationRun"]
            and not config["decisionRule"]["passAuthorizesPolicyTuning"]
            and not config["decisionRule"]["passAuthorizesModelRegistrationAuthorityActionOrExecution"]
        ),
        "required_locked_files_exist": all(
            path.is_file()
            for path in (
                config_path,
                parent_path,
                planner_path,
                roadmap_path,
                plan_path,
                protocol_path,
                tests_path,
                runner_path,
                verifier_path,
                auditor_path,
            )
        ),
        "formal_population_absent_before_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "172-trusted-shadow-integration-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_population_build" if passed else "reject_V172_design",
        "checks": checks,
        "prelock_exposure": exposure,
        "planner_policy_score_count": 0,
        "sandbox_transaction_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V171_outcome": parent_path,
        "source_V170_outcome": planner_path,
        "roadmap": roadmap_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "172-trusted-shadow-integration-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_generator_membership_eligibility_targets_weights_gates_or_decision": False,
            "build_formal_population_once": True,
            "score_policy_or_run_sandbox_transaction": False,
            "run_model_register_mutate_state_act_or_execute": False,
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
