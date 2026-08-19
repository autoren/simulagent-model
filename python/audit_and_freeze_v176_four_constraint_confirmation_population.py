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
    config_path = PROJECT_ROOT / "configs/v176-four-constraint-confirmation-population.json"
    parent_path = PROJECT_ROOT / "configs/v175-certification-aware-planner-development-outcome-lock.json"
    source_v172_path = PROJECT_ROOT / "configs/v172-trusted-shadow-integration-population-outcome-lock.json"
    planner_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v175.md"
    plan_path = PROJECT_ROOT / "docs/v176-four-constraint-confirmation-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v176_four_constraint_confirmation_population.py"
    tests_path = PROJECT_ROOT / "python/test_v176_four_constraint_confirmation_population.py"
    runner_path = PROJECT_ROOT / "python/run_v176_four_constraint_confirmation_population.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v176_four_constraint_confirmation_population_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v176_four_constraint_confirmation_population.py"
    audit_path = PROJECT_ROOT / "outputs/v176-four-constraint-confirmation-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v176-four-constraint-confirmation-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v176-four-constraint-confirmation-population/population"
    outcome_path = PROJECT_ROOT / "configs/v176-four-constraint-confirmation-population-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V176 is already preregistered, built, or frozen")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    v172 = json.loads(source_v172_path.read_text())
    planner = json.loads(planner_path.read_text())
    V172_states_path = PROJECT_ROOT / v172["constraint_states"]
    V172_targets_path = PROJECT_ROOT / v172["target_cases"]
    generator = config["generator"]
    eligibility = config["confirmationEligibility"]
    targets = config["targetCases"]
    freshness = config["freshnessDefinition"]
    gates = config["populationGates"]
    exposure = config["preLockExposure"]
    authority = config["authorityBoundary"]
    checks = {
        "V175_is_frozen_strong_positive_and_authorizes_only_separate_confirmation_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_safety_gates_passed"]
            and parent["outcome"]["beneficial"]
            and parent["outcome"]["strong_development"]
            and parent["authorization"]["design_fresh_exact_signature_disjoint_confirmation"]
            and not parent["authorization"]["run_confirmation_without_separate_population_and_lock"]
            and not parent["authorization"]["modify_rerun_select_subsample_or_tune_V175"]
        ),
        "V172_source_artifacts_and_V167_prior_are_exact": bool(
            valid_lock(v172)
            and v172["outcome"]["passed"]
            and valid_lock(planner)
            and planner["config_payload"]["prior"]["classMass"]
            == {
                "alias": "1/3",
                "composition": "1/3",
                "provisional_primitive": "1/3",
            }
            and V172_states_path.is_file()
            and file_sha256(V172_states_path) == v172["constraint_states_sha256"]
            and V172_targets_path.is_file()
            and file_sha256(V172_targets_path) == v172["target_cases_sha256"]
        ),
        "complete_four_constraint_generator_is_frozen_without_score_selection": bool(
            generator["candidateTruthTableCount"] == 256
            and generator["valuationCount"] == 8
            and generator["constraintCountPerState"] == 4
            and generator["outcomesPerConstraint"] == [0, 1]
            and generator["completeValuationQuadrupleAndOutcomeEnumeration"]
            and generator["requiredSourceStateCount"] == 1120
            and generator["retainEverySourceState"]
            and not generator["selectionUsesPlannerScoresPoliciesSandboxOutcomesOrTargetClass"]
        ),
        "eligibility_targets_and_weights_use_only_frozen_structure": bool(
            eligibility["usesOnlyFrozenCandidateClassMetadata"]
            and eligibility["ineligibleStatesRetainedInPopulation"]
            and not eligibility["policyOrSandboxScoringDuringEligibility"]
            and targets["includeEveryCandidateInEveryEligibleVersionSpace"]
            and not targets["targetSelectionOrSubsampling"]
            and not targets["targetWeightsUsePolicyOutcomes"]
            and targets["targetIdentitiesFrozenBeforeConfirmationScoring"]
        ),
        "freshness_is_exact_context_disjointness_with_candidate_reuse_disclosed": bool(
            freshness["compareAgainstEveryV172TargetContextSignature"]
            and freshness["requiredExactSignatureOverlapCount"] == 0
            and not freshness["candidateIdentityDisjointnessRequired"]
            and freshness["candidateOntologyIsIntentionallyReused"]
            and freshness["freshnessClaim"].startswith("new exact evidence contexts")
        ),
        "population_gates_match_the_prelock_structural_census_and_zero_scores": bool(
            gates["requiredSourceStateCount"] == 1120
            and gates["requiredSelectedStateCount"] == 1120
            and gates["requiredCandidatesPerState"] == 16
            and gates["requiredEligibleStateCount"] == 135
            and gates["requiredIneligibleStateCount"] == 985
            and gates["requiredTargetCaseCount"] == 2160
            and gates["requiredTargetClassCounts"]
            == {"alias": 138, "composition": 180, "provisional_primitive": 1842}
            and gates["requiredClassCoverageCounts"]
            == {"1": 717, "2": 268, "3": 135}
            and gates["requiredExactTargetContextSignatureOverlapWithV172"] == 0
            and gates["maximumPlannerPolicyScoreCount"] == 0
            and gates["maximumSandboxTransactionCount"] == 0
        ),
        "prelock_and_authority_boundaries_are_closed": bool(
            exposure["implementationStructuralCensusCount"] == 1
            and exposure["implementationPopulationTestBuildCount"] == 1
            and exposure["formalPopulationBuildCount"] == 0
            and all(
                value == 0
                for key, value in exposure.items()
                if key
                not in {
                    "implementationStructuralCensusCount",
                    "implementationPopulationTestBuildCount",
                    "formalPopulationBuildCount",
                }
            )
            and authority["populationContainsOnlyShadowVersionSpacesAndTargetIdentities"]
            and authority["authoritativeOntologyAndStateImmutable"]
            and authority["plannerNotRun"]
            and authority["sandboxNotRun"]
            and authority["modelNotRun"]
            and not authority["provisionalRegistrationAllowed"]
            and not authority["actionOrExecutionAllowed"]
            and authority["realExecutionCount"] == 0
            and not config["decisionRule"]["passAuthorizesImmediateConfirmationRun"]
            and not config["decisionRule"]["passAuthorizesPolicyTuning"]
            and not config["decisionRule"]["passAuthorizesModelRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_formal_population_is_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    parent_path,
                    source_v172_path,
                    planner_path,
                    V172_states_path,
                    V172_targets_path,
                    roadmap_path,
                    plan_path,
                    protocol_path,
                    tests_path,
                    runner_path,
                    verifier_path,
                    auditor_path,
                )
            )
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "176-four-constraint-confirmation-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_V176_population_build" if passed else "reject_V176_design",
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
        "parent_V175_outcome": parent_path,
        "source_V172_outcome": source_v172_path,
        "source_V167_planner_lock": planner_path,
        "V172_constraint_states": V172_states_path,
        "V172_target_cases": V172_targets_path,
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
        "schema_version": "176-four-constraint-confirmation-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_generator_membership_eligibility_targets_weights_freshness_gates_or_decision": False,
            "build_formal_population_once": True,
            "score_policy_or_run_sandbox_transaction": False,
            "run_confirmation_without_separate_lock": False,
            "run_model_register_mutate_state_call_service_or_execute": False,
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
