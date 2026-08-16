#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", default="configs/v53r1-design-repair.json")
    parser.add_argument("--plan", default="docs/v53r1-design-repair-plan.md")
    parser.add_argument(
        "--output", default="outputs/v53r1-design-repair/design-audit.json"
    )
    args = parser.parse_args()
    repair_path = (PROJECT_ROOT / args.repair).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    repair = json.loads(repair_path.read_text())
    source_path = (PROJECT_ROOT / repair["sourceDesignLock"]).resolve()
    source = json.loads(source_path.read_text())
    config = source["config_payload"]
    errors = []

    source_bound = (
        source["authorization"]["write_and_audit_smc_squared_implementation"]
        and not source["authorization"]["construct_v53_populations"]
        and not source["authorization"]["run_v53_evaluation"]
        and file_sha256(PROJECT_ROOT / source["config"]) == source["config_sha256"]
        and file_sha256(PROJECT_ROOT / source["preregistration"])
        == source["preregistration_sha256"]
    )
    if not source_bound:
        errors.append("source V53 design lock is not intact or pre-population")

    changes = repair["authorizedDesignChanges"]
    repair_narrow = (
        changes == {
            "pmcmcProposalStandardDeviation": 1.3,
            "jointThetaBins": 10,
            "exactBenchmarkRepeatAggregation": (
                "equal_weight_posterior_mixture_across_three_independent_repeats"
            ),
        }
        and config["pmcmcReference"]["proposalStandardDeviation"] == 0.35
        and config["smcSquared"]["independentRepeatsOnExactBenchmark"] == 3
        and config["gates"]["maximumPrimaryMeanBinnedProgramThetaTv"] == 0.06
    )
    if not repair_narrow:
        errors.append("V53r1 changes exceed the diagnosed proposal and resolution repair")

    fixture = repair["alteredSeedChecks"]
    fixture_ok = (
        fixture["proposalStandardDeviation"] == 1.3
        and 0.1 <= fixture["acceptanceRange"][0]
        <= fixture["acceptanceMean"] <= fixture["acceptanceRange"][1] <= 0.7
        and fixture["seeds"] == "all_v53_root_seeds_plus_1000000"
        and not fixture["sealedCandidateOverlap"]
    )
    if not fixture_ok:
        errors.append("altered-seed proposal fixture does not satisfy the frozen gate")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v53-implementation-lock.json",
            "configs/v53r1-design-lock.json",
            "configs/v53r1-implementation-lock.json",
            "configs/v53-population-seal.json",
            "configs/v53r1-population-seal.json",
            "data/v53-continuous-parameter-smc2",
            "data/v53r1-continuous-parameter-smc2",
            "outputs/v53-continuous-parameter-smc2/evaluation-attempt.json",
            "outputs/v53r1-continuous-parameter-smc2/evaluation-attempt.json",
        )
    )
    if not downstream_absent:
        errors.append("V53 candidate population or downstream lock already exists")

    audit = {
        "schema_version": 53,
        "revision": "r1",
        "experiment": "v53r1_design_repair_audit",
        "passed": not errors,
        "decision": (
            "authorize_v53r1_design_lock" if not errors else "repair_v53r1_design"
        ),
        "errors": errors,
        "repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_sha256": file_sha256(repair_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_design_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_design_lock_sha256": file_sha256(source_path),
        "checks": {
            "source_design_bound_and_prepopulation": source_bound,
            "narrow_design_repair": repair_narrow,
            "altered_seed_fixture_within_gate": fixture_ok,
            "downstream_absent": downstream_absent,
        },
        "data_access": {
            "candidate_population_records_accessed": 0,
            "smc_squared_evaluation_runs": 0,
            "pmcmc_reference_evaluation_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
