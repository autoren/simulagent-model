#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/v56-symbolic-probabilistic-policy-verification.json",
    )
    parser.add_argument(
        "--plan",
        default="docs/v56-symbolic-probabilistic-policy-verification-plan.md",
    )
    parser.add_argument(
        "--output",
        default="outputs/v56-symbolic-probabilistic-policy-verification/design-audit.json",
    )
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.output)
    )
    config = json.loads(config_path.read_text())
    source_path = (PROJECT_ROOT / config["sourceV55r1OutcomeLock"]).resolve()
    source = json.loads(source_path.read_text())
    errors = []

    source_bound = (
        source["combined_planning_layer_qualification_passed"]
        and not source["v55_standalone_qualification_passed"]
        and source["v55r1_qualification_passed"]
        and source["authorization"][
            "preregister_symbolic_and_probabilistic_policy_verification"
        ]
        and not source["authorization"]["run_formal_verification"]
        and file_sha256(PROJECT_ROOT / source["result"])
        == source["result_sha256"]
        and file_sha256(PROJECT_ROOT / source["post_result_audit"])
        == source["post_result_audit_sha256"]
        and file_sha256(PROJECT_ROOT / source["summary"])
        == source["summary_sha256"]
        and file_sha256(PROJECT_ROOT / source["v55_outcome_lock"])
        == source["v55_outcome_lock_sha256"]
    )
    if not source_bound:
        errors.append("V55r1 does not authorize or bind the V56 preregistration")

    boundary = config["claimBoundary"]
    required_true = (
        "verifyEveryFrozenV55AndV55r1Policy",
        "boundedHorizonPolicyExecution",
        "symbolicReachableTransitionVerification",
        "probabilisticTerminationReachability",
        "probabilisticTerminalGoalReachability",
        "posteriorExpectedAccumulatedReward",
        "externalStormProcess",
        "independentZ3Encoding",
        "policyVerificationNotPlannerVerification",
        "posteriorExpectedNotWorstCase",
    )
    required_false = (
        "formalSafetyContract",
        "catastrophicOutcomeClaim",
        "worstCaseSafetyClaim",
        "parameterUniformGuarantee",
        "unboundedTemporalClaim",
        "longHorizonClaim",
        "plannerOptimalityClaimBeyondFrozenSources",
        "newPlanningEvaluation",
        "approximateOrLearnedPlanning",
        "languageGrounding",
        "modelAccess",
        "adapterTraining",
        "finalEvaluation",
    )
    boundary_ok = (
        boundary["frozenHorizon"] == 3
        and all(boundary[key] for key in required_true)
        and not any(boundary[key] for key in required_false)
    )
    if not boundary_ok:
        errors.append("V56 claim boundary is too broad or omits bounded verification")

    policies = config["sourcePolicies"]
    cohort_counts = {row["id"]: row["tasks"] for row in policies["cohorts"]}
    sources_ok = (
        cohort_counts == {"v55": 32, "v55r1": 16}
        and policies["totalPolicies"] == 48
        and not policies["truthFieldsAvailableToCompiler"]
        and policies["sourceResultMutation"] == "forbidden"
        and policies["sourceEvaluationRerun"] == "forbidden"
        and all(
            (PROJECT_ROOT / row[key]).exists()
            for row in policies["cohorts"]
            for key in ("populationSeal", "outcomeLock", "result")
        )
    )
    if not sources_ok:
        errors.append("V56 frozen source census or truth boundary is invalid")

    executor = config["formalExecutor"]
    executor_ok = (
        executor["horizonActions"] == 3
        and executor["actionCosts"] == {
            "pulse": 0.01,
            "route": 0.01,
            "wait": 0.0,
        }
        and "formal executor" not in executor["plannerRole"].lower()
        and "defines" in executor["executorRole"]
        and "checks" in executor["verifierRole"]
    )
    if not executor_ok:
        errors.append("V56 planner, executor, verifier, or reward roles are invalid")

    symbolic = config["symbolicVerification"]
    symbolic_ok = (
        symbolic["backend"] == "Z3Py"
        and symbolic["package"] == "z3-solver"
        and symbolic["packageVersion"] == "4.16.0.0"
        and symbolic["solverVersion"] == "4.16.0"
        and len(symbolic["reachableChecks"]) == 8
        and symbolic["exhaustiveSyntheticDomain"]["worldsPerTemplate"] == 256
        and symbolic["exhaustiveSyntheticDomain"]["actionsPerWorld"] == 5
        and len(symbolic["exhaustiveSyntheticDomain"]["queueFixtures"]) == 3
    )
    if not symbolic_ok:
        errors.append("V56 independent symbolic verification is underspecified")

    probabilistic = config["probabilisticVerification"]
    probabilistic_ok = (
        probabilistic["backend"] == "Storm standalone CLI"
        and probabilistic["version"] == "1.13.0"
        and not probabilistic["pythonBindings"]
        and set(probabilistic["properties"]) == {"termination", "success", "return"}
        and probabilistic["independentReferences"]["termination"] == 1.0
    )
    if not probabilistic_ok:
        errors.append("V56 probabilistic model checker protocol is invalid")

    implementation = config["implementationAudit"]
    controls_ok = (
        not implementation["sealedSourcePolicyRecordsAccessible"]
        and len(implementation["analyticStormFixtures"]) == 5
        and len(implementation["symbolicMutants"]) == 5
        and len(implementation["probabilisticMutants"]) == 5
        and implementation["requiredMutantKillRate"] == 1.0
        and implementation["requiredAnalyticFixturePassRate"] == 1.0
    )
    if not controls_ok:
        errors.append("V56 implementation controls are incomplete")

    bundle = config["verificationBundle"]
    bundle_ok = (
        bundle["constructionBeforeImplementationLock"] == "forbidden"
        and bundle["modelsPerPolicy"] == 1
        and bundle["expectedModels"] == 48
        and len(bundle["requiredFilesPerPolicy"]) == 5
        and bundle["sealBeforeVerification"]
        and bundle["postSealMutation"] == "forbidden"
    )
    if not bundle_ok:
        errors.append("V56 model bundle quotas or sealing rule are invalid")

    gates = config["gates"]
    gates_ok = (
        gates["minimumCompletedPolicyFraction"] == 1.0
        and gates["minimumPolicyCount"] == 48
        and gates["minimumV55PolicyCount"] == 32
        and gates["minimumV55r1PolicyCount"] == 16
        and gates["minimumReconstructedRootActionMatchRate"] == 1.0
        and gates["maximumReconstructedRootValueError"] == 1e-10
        and gates["minimumReachableStateInvariantProofRate"] == 1.0
        and gates["minimumReachableTransitionSupportEquivalenceProofRate"] == 1.0
        and gates["minimumPolicyObservationTotalityRate"] == 1.0
        and gates["maximumNonterminalDeadlockCount"] == 0
        and gates["maximumZ3UnknownCount"] == 0
        and gates["minimumStormCompletedModelFraction"] == 1.0
        and gates["maximumTerminationProbabilityError"] == 1e-10
        and gates["maximumSuccessProbabilityError"] == 1e-9
        and gates["maximumExpectedReturnErrorAgainstFrozenValue"] == 1e-9
        and gates["maximumExpectedReturnErrorAgainstIndependentPolicyEvaluator"] == 1e-9
        and gates["minimumImplementationMutantKillRate"] == 1.0
        and gates["minimumAnalyticFixturePassRate"] == 1.0
        and all(
            gates[key] == 0
            for key in (
                "maximumTruthFieldAccessCount",
                "maximumSourceResultMutationCount",
                "maximumVerificationBundleHashMismatchCount",
                "maximumToolVersionMismatchCount",
                "maximumUnexpectedVerificationAttemptCount",
            )
        )
    )
    if not gates_ok:
        errors.append("V56 exact, quantitative, control, or integrity gates are invalid")

    stage = config["stageAuthorization"]
    firewall_ok = (
        set(config["firewall"].values()) == {"forbidden"}
        and stage == {
            "installPinnedVerificationDependencies": True,
            "writeAndAuditIndependentVerifiers": True,
            "constructVerificationBundle": False,
            "runCandidateFormalVerification": False,
            "formalSafetyClaim": False,
            "languageGrounding": False,
            "modelAccess": False,
        }
    )
    if not firewall_ok:
        errors.append("V56 firewall or stage authorization is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v56-design-lock.json",
            "configs/v56-implementation-lock.json",
            "configs/v56-verification-bundle-seal.json",
            "configs/v56-evaluation-implementation-lock.json",
            "configs/v56-outcome-lock.json",
            "data/v56-symbolic-probabilistic-policy-verification",
            "outputs/v56-symbolic-probabilistic-policy-verification/implementation-audit.json",
            "outputs/v56-symbolic-probabilistic-policy-verification/evaluation-attempt.json",
            "outputs/v56-symbolic-probabilistic-policy-verification/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V56 downstream artifact exists before design lock")

    checks = {
        "source_v55r1_authorization_and_binding": source_bound,
        "bounded_claim_boundary": boundary_ok,
        "all_frozen_policy_sources_and_truth_firewall": sources_ok,
        "planner_executor_verifier_separation": executor_ok,
        "independent_symbolic_protocol": symbolic_ok,
        "external_probabilistic_protocol": probabilistic_ok,
        "analytic_and_mutation_controls": controls_ok,
        "verification_bundle_sealing": bundle_ok,
        "noncompensatory_gates": gates_ok,
        "firewall_and_stage_authorization": firewall_ok,
        "downstream_absent": downstream_absent,
    }
    audit = {
        "schema_version": 56,
        "experiment": "v56_design_audit",
        "passed": not errors,
        "decision": "authorize_v56_design_lock" if not errors else "repair_v56_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "checks": checks,
        "data_access": {
            "v55_candidate_policy_records_accessed": 0,
            "v55r1_candidate_policy_records_accessed": 0,
            "formal_verification_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
