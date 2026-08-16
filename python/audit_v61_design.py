#!/usr/bin/env python3
"""Audit the V61 bounded policy-verification preregistration."""
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v61-long-horizon-policy-verification.json"
    )
    parser.add_argument(
        "--plan", default="docs/v61-long-horizon-policy-verification-plan.md"
    )
    parser.add_argument(
        "--output", default="outputs/v61-long-horizon-policy-verification/design-audit.json"
    )
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.output)
    )
    config = json.loads(config_path.read_text())
    errors: list[str] = []

    source_paths = (
        config["sourceV56OutcomeLock"], config["sourceV60OutcomeLock"]
    )
    source_ok = all(
        (PROJECT_ROOT / path).exists()
        and json.loads((PROJECT_ROOT / path).read_text())["qualification_passed"]
        for path in source_paths
    )
    if not source_ok:
        errors.append("V61 requires passing frozen V56 and V60 outcomes")

    source = config["sourcePolicies"]
    seal_path = PROJECT_ROOT / source["populationSeal"]
    seal = json.loads(seal_path.read_text())
    census_ok = (
        source["tasks"] == 24
        and source["replicatesPerTask"] == 3
        and source["policiesPerHorizon"] == 24
        and source["totalPolicies"] == 72
        and source["inferenceBudget"] == 509
        and source["inferenceRepeats"] == 3
        and source["searchBudget"] == 1024
        and seal["experiment"] == "v59_population_seal"
        and not seal["authorization"]["candidate_access_v59_audit_truth"]
        and source["selection"].startswith("all_primary")
        and not source["truthFieldsAvailableToReconstructorOrCompiler"]
    )
    if not census_ok:
        errors.append("V61 must exhaustively bind all 72 public-only primary policies")

    independent = config["independentVerification"]
    verification_ok = (
        "does_not_call_continuous_unit_transition" in independent["transitionInterpreter"]
        and "does_not_call_v59_deployment_action" in independent["deploymentInterpreter"]
        and len(independent["reachableChecks"]) == 9
        and independent["z3SolverVersion"] == "4.16.0"
        and config["probabilisticVerification"]["version"] == "1.13.0"
        and len(config["implementationAudit"]["analyticFixtures"]) == 6
        and len(config["implementationAudit"]["mutants"]) == 10
    )
    if not verification_ok:
        errors.append("V61 independent executor, checker, or controls are incomplete")

    mc = config["probabilisticVerification"]["monteCarloBound"]
    mc_ok = (
        mc["familywiseAlpha"] == 0.01
        and mc["comparisons"] == 72
        and mc["episodesPerPolicy"] == 2048
        and mc["rewardRangeByHorizon"] == {"3": 1.03, "5": 1.05, "7": 1.07}
    )
    if not mc_ok:
        errors.append("V61 Monte Carlo source cross-check is not prospectively fixed")

    gates = config["gates"]
    gates_ok = (
        len(gates) == 27
        and gates["minimumCompletedPolicyFraction"] == 1.0
        and gates["minimumPolicyCount"] == 72
        and gates["minimumPolicyCountPerHorizon"] == 24
        and gates["minimumReconstructedTreeHashMatchRate"] == 1.0
        and gates["minimumReachableTransitionSupportEquivalenceProofRate"] == 1.0
        and gates["minimumStormCompletedModelFraction"] == 1.0
        and gates["maximumExpectedReturnErrorAgainstIndependentExecutor"] <= 1e-9
        and gates["minimumV60MonteCarloReturnWithinSimultaneousBoundRate"] == 1.0
        and gates["minimumImplementationMutantKillRate"] == 1.0
        and gates["maximumTruthFieldAccessCount"] == 0
        and gates["maximumUnexpectedVerificationAttemptCount"] == 0
    )
    if not gates_ok:
        errors.append("V61 gates are incomplete or compensatory")

    boundary = config["claimBoundary"]
    firewall = config["firewall"]
    boundary_ok = (
        boundary["verifyEveryPrimaryV60Policy"]
        and boundary["exactPosteriorExecutionDistribution"]
        and boundary["policyVerificationNotSearchAlgorithmVerification"]
        and not boundary["plannerOptimality"]
        and not boundary["formalSafetyContract"]
        and not boundary["parameterUniformGuarantee"]
        and not boundary["humanAuthoredLanguage"]
        and all(value == "forbidden" for value in firewall.values())
    )
    if not boundary_ok:
        errors.append("V61 claim boundary or firewall is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v61-design-lock.json",
            "configs/v61-implementation-lock.json",
            "configs/v61-verification-bundle-seal.json",
            "configs/v61-evaluation-implementation-lock.json",
            "configs/v61-outcome-lock.json",
            "outputs/v61-long-horizon-policy-verification/verification-attempt.json",
            "outputs/v61-long-horizon-policy-verification/verification",
            "docs/v61-results.md",
        )
    )
    if not downstream_absent:
        errors.append("V61 downstream artifacts already exist")

    audit = {
        "schema_version": 61,
        "experiment": "v61_design_audit",
        "passed": not errors,
        "decision": "freeze_v61_design" if not errors else "repair_v61_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_locks_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in source_paths
        },
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "checks": {
            "passing_frozen_v56_and_v60_sources": source_ok,
            "exhaustive_public_only_seventy_two_policy_census": census_ok,
            "independent_executor_external_checker_and_controls": verification_ok,
            "prospective_familywise_monte_carlo_cross_check": mc_ok,
            "twenty_seven_noncompensatory_gates": gates_ok,
            "claim_boundary_and_firewall": boundary_ok,
            "downstream_absence": downstream_absent,
        },
        "data_access": {
            "v59_candidate_public_records_accessed": 0,
            "v59_audit_truth_records_accessed": 0,
            "v60_source_policy_cells_accessed": 0,
            "v61_verification_runs": 0,
            "human_authored_v58_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
