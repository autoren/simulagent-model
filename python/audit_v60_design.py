#!/usr/bin/env python3
"""Audit the V60 approximate-belief decision-calibration preregistration."""
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v60-approximate-belief-decision-calibration.json"
    )
    parser.add_argument(
        "--plan", default="docs/v60-approximate-belief-decision-calibration-plan.md"
    )
    parser.add_argument(
        "--output", default="outputs/v60-approximate-belief-decision-calibration/design-audit.json"
    )
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.output)
    )
    config = json.loads(config_path.read_text())
    errors: list[str] = []

    source_locks = {
        config["sourceV53r2OutcomeLock"]: True,
        config["sourceV59OutcomeLock"]: True,
    }
    source_ok = all(
        (PROJECT_ROOT / path).exists()
        and json.loads((PROJECT_ROOT / path).read_text())["qualification_passed"]
        for path in source_locks
    )
    if not source_ok:
        errors.append("V60 requires passing frozen V53r2 and V59 outcomes")

    seal_path = PROJECT_ROOT / config["population"]["populationSeal"]
    seal = json.loads(seal_path.read_text())
    population_ok = (
        seal["experiment"] == "v59_population_seal"
        and not seal["authorization"]["modify_v59_population"]
        and not seal["authorization"]["candidate_access_v59_audit_truth"]
        and config["population"]["publicTasks"] == 24
        and config["population"]["tasksPerHorizon"] == 8
        and not config["population"]["auditTruthAccess"]
        and not config["population"]["newPopulationConstruction"]
    )
    if not population_ok:
        errors.append("V60 must reuse only the immutable V59 public population")

    inference_ok = (
        config["inference"]["outerThetaParticleBudgets"] == [31, 127, 509]
        and config["inference"]["independentRepeatsPerBudget"] == 3
        and config["inference"]["primaryBudget"] == 509
        and config["inference"]["innerStateParticleBudget"] == 127
        and len(config["inference"]["comparisonMetrics"]) == 4
    )
    if not inference_ok:
        errors.append("V60 inference budgets or exact-agreement endpoints are invalid")

    planning_ok = (
        config["planning"]["searchBudget"] == 1024
        and config["planning"]["replicatesPerTask"] == 3
        and config["planning"]["policyEvaluationEpisodes"] == 2048
        and config["planning"]["exactDynamicProgrammingReferenceHorizon"] == 3
        and config["planning"]["observationBlindControlAtPrimaryInferenceBudget"]
        and config["planning"]["commonPlanningSeedWithinApproximateExactAndBlindTriple"]
    )
    if not planning_ok:
        errors.append("V60 paired search, exact reference, or return evaluation is invalid")

    gates = config["gates"]
    gates_ok = (
        len(gates) == 23
        and gates["minimumCompletedTaskInferenceBudgetPlanningReplicateFraction"] == 1.0
        and gates["minimumSmcPosteriorNormalizationRate"] == 1.0
        and gates["maximumPrimaryMeanConfigurationTv"] <= 0.08
        and gates["minimumPrimaryHorizonThreeExactOptimalSetMembershipRate"] >= 0.75
        and gates["maximumPrimaryExactBeliefMinusApproximateBeliefPolicyReturn"] <= 0.03
        and gates["minimumSimulationBudgetAccountingRate"] == 1.0
        and gates["maximumTruthFieldAccessCount"] == 0
        and gates["maximumUnexpectedEvaluationAttemptCount"] == 0
    )
    if not gates_ok:
        errors.append("V60 gates are incomplete or compensatory")

    boundary = config["claimBoundary"]
    firewall = config["firewall"]
    boundary_ok = (
        boundary["frozenV53r2SmcSquared"]
        and boundary["exactPosteriorReference"]
        and not boundary["exactLongHorizonOptimality"]
        and not boundary["formalPolicyVerification"]
        and not boundary["humanAuthoredLanguage"]
        and not boundary["modelAccess"]
        and all(value == "forbidden" for value in firewall.values())
    )
    if not boundary_ok:
        errors.append("V60 claim boundary or firewall is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v60-design-lock.json",
            "configs/v60-implementation-lock.json",
            "configs/v60-evaluation-implementation-lock.json",
            "configs/v60-outcome-lock.json",
            "outputs/v60-approximate-belief-decision-calibration/evaluation-attempt.json",
            "outputs/v60-approximate-belief-decision-calibration/evaluation",
            "docs/v60-results.md",
        )
    )
    if not downstream_absent:
        errors.append("V60 downstream artifacts already exist")

    audit = {
        "schema_version": 60,
        "experiment": "v60_design_audit",
        "passed": not errors,
        "decision": "freeze_v60_design" if not errors else "repair_v60_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_locks_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in source_locks
        },
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "checks": {
            "passing_frozen_v53r2_and_v59_sources": source_ok,
            "immutable_public_only_v59_population_reuse": population_ok,
            "frozen_smc2_budgets_and_exact_agreement": inference_ok,
            "paired_search_exact_reference_and_return_evaluation": planning_ok,
            "twenty_three_noncompensatory_gates": gates_ok,
            "claim_boundary_and_firewall": boundary_ok,
            "downstream_absence": downstream_absent,
        },
        "data_access": {
            "v59_candidate_public_records_accessed": 0,
            "v59_audit_truth_records_accessed": 0,
            "v60_evaluation_runs": 0,
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
