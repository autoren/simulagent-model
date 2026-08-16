#!/usr/bin/env python3
"""Audit and freeze the pre-implementation V65 SMC²-EIG design."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _bound_file(lock: dict, path_key: str, hash_key: str) -> bool:
    path = PROJECT_ROOT / lock[path_key]
    return path.is_file() and file_sha256(path) == lock[hash_key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v65-smc2-eig-portability.json")
    parser.add_argument("--plan", default="docs/v65-smc2-eig-portability-plan.md")
    parser.add_argument(
        "--audit", default="outputs/v65-smc2-eig-portability/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v65-design-lock.json")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V65 design already frozen")
    config = _load(config_path)
    errors: list[str] = []

    source_paths = {
        "v64_outcome": (PROJECT_ROOT / config["sourceV64OutcomeLock"]).resolve(),
        "v64_design": (PROJECT_ROOT / config["sourceV64DesignLock"]).resolve(),
        "v64_implementation": (
            PROJECT_ROOT / config["sourceV64ImplementationLock"]
        ).resolve(),
        "v64_population": (
            PROJECT_ROOT / config["sourceV64PopulationSeal"]
        ).resolve(),
        "v63r1_outcome": (
            PROJECT_ROOT / config["sourceV63r1OutcomeLock"]
        ).resolve(),
        "v63_design": (PROJECT_ROOT / config["sourceV63DesignLock"]).resolve(),
    }
    sources = {name: _load(path) for name, path in source_paths.items()}

    v64_outcome = sources["v64_outcome"]
    outcome_ok = bool(
        v64_outcome["qualification_passed"]
        and v64_outcome["decision"]
        == "authorize_preregistration_of_pooled_three_repeat_SMC2_EIG_stage"
        and v64_outcome["authorization"][
            "preregister_pooled_three_repeat_SMC2_EIG_stage"
        ]
        and not v64_outcome["authorization"]["construct_or_run_SMC2_EIG_population"]
        and not v64_outcome["authorization"]["reward_planning"]
        and not v64_outcome["authorization"]["modify_or_rerun_v64"]
        and _bound_file(v64_outcome, "result", "result_sha256")
        and _bound_file(v64_outcome, "summary", "summary_sha256")
        and _bound_file(
            v64_outcome, "post_result_audit", "post_result_audit_sha256"
        )
    )
    if not outcome_ok:
        errors.append("V64 outcome does not authorize or bind V65 preregistration")

    v64_design = sources["v64_design"]
    v64_implementation = sources["v64_implementation"]
    v64_population = sources["v64_population"]
    v64_source_ok = bool(
        file_sha256(source_paths["v64_design"])
        == v64_implementation["design_lock_sha256"]
        and file_sha256(PROJECT_ROOT / "python/v64_external_eig.py")
        == v64_implementation["source_sha256"]["python/v64_external_eig.py"]
        and file_sha256(PROJECT_ROOT / v64_population["files"]["selection_public"]["path"])
        == v64_population["files"]["selection_public"]["sha256"]
        == config["pairedReuseBoundary"]["sourcePopulationSha256"]
        and v64_population["counts"]["selection_public"]
        == config["pairedReuseBoundary"]["sourceRecords"]
        and not v64_population["authorization"]["modify_or_rebuild_populations"]
        and v64_design["config_payload"]["selectionPopulation"]["records"] == 192
        and v64_design["config_payload"]["exactOracle"]["quadratureNodes"] == 257
        and v64_design["external_model_sha256"]
        == config["externalFamily"]["modelSha256"]
    )
    if not v64_source_ok:
        errors.append("V64 design, implementation, population, or exact source binding failed")

    reuse = config["pairedReuseBoundary"]
    subset = config["subset"]
    subset_ok = bool(
        reuse["sourceRecords"] == 192
        and reuse["sourcePrefixLengths"] == [0, 1, 2, 3, 4, 5]
        and reuse["sourceRecordsPerPrefixLength"] == 32
        and not reuse["independentExactReplication"]
        and not reuse["modifyOrRerunV64"]
        and not reuse["loadV64SelectionAudit"]
        and not reuse["loadV64EvaluationResultDuringSubsetSelection"]
        and not reuse["selectByExactEIGOrAction"]
        and subset["records"] == 48
        and subset["recordsPerPrefixLength"] == 8
        and subset["prefixLengths"] == reuse["sourcePrefixLengths"]
        and subset["recordsPerPrefixLength"] * len(subset["prefixLengths"])
        == subset["records"]
        and subset["retainEverySelectedRecord"]
        and not subset["recordRejectionAfterSelection"]
        and not subset["truthFields"]
        and not subset["freshPopulation"]
        and subset["selectionSeed"] == config["seeds"]["subsetSelectionSeed"] == 6511
        and "SHA256" in subset["selectionRule"]
    )
    if not subset_ok:
        errors.append("prospective V64-public subset rule or paired-reuse boundary is invalid")

    v63_smc = sources["v63_design"]["config_payload"]["smcSquared"]
    v65_smc = config["smcSquared"]
    inherited_ok = bool(
        v65_smc["outerThetaParticleBudgets"]
        == v63_smc["outerThetaParticleBudgets"]
        == [31, 127, 509]
        and v65_smc["primaryOuterThetaParticleBudget"]
        == v63_smc["primaryOuterThetaParticleBudget"]
        == 509
        and v65_smc["independentRepeatsPerBudget"]
        == v63_smc["independentRepeatsOnExactBenchmark"]
        == 3
        and v65_smc["innerStateParticleBudget"]
        == v63_smc["innerStateParticleBudget"]
        == 127
        and v65_smc["outerEssThresholdFraction"]
        == v63_smc["outerEssThresholdFraction"]
        == 0.5
        and v65_smc["innerEssThresholdFraction"]
        == v63_smc["innerEssThresholdFraction"]
        == 0.5
        and v65_smc["rejuvenationStepsPerOuterResampling"]
        == v63_smc["rejuvenationStepsPerOuterResampling"]
        == 2
        and v65_smc["proposalStandardDeviation"]
        == v63_smc["proposalStandardDeviation"]
        == 0.4
        and not v65_smc["adaptiveProposal"]
        and "equal_weight_posterior_mixture" in v65_smc["repeatAggregation"]
        and v65_smc["singleRepeatRole"]
        == "mandatory_nonqualifying_diagnostic_only"
    )
    if not inherited_ok:
        errors.append("V65 does not preserve the qualifying V53r2/V63 SMC2 architecture")

    acquisition = config["approximateAcquisition"]
    targets = config["comparisonTargets"]
    acquisition_ok = bool(
        acquisition["poolBeforeScore"]
        and acquisition["integrateInnerStateBeforeTreatingAnOuterParticleAsAStaticLatentAtom"]
        and acquisition["doNotTreatInnerStateOrRepeatIdentityAsTargetLatent"]
        and acquisition["scoreEveryCandidate"]
        and acquisition["candidateOrder"] == ["n", "e", "s", "w"]
        and acquisition["selectionBeforeOutcome"]
        and len(targets["posterior"]) == 4
        and len(targets["prediction"]) == 1
        and len(targets["acquisition"]) == 4
    )
    if not acquisition_ok:
        errors.append("pooled posterior, static-latent EIG, or comparison targets are invalid")

    gates = config["gates"]
    controls = config["controls"]
    zero_gate_keys = (
        "maximumUnexpectedEvaluationAttemptCount",
        "maximumV64SelectionAuditRecordsLoaded",
        "maximumTruthFieldAccessCount",
        "maximumRealizedOutcomeAccessBeforeSelectionCount",
        "maximumCandidateOmissionCount",
        "maximumTieBreakViolationCount",
        "maximumUnintendedRandomStreamCollisions",
        "maximumHumanRecordAccessCount",
        "maximumModelForwardPassCount",
        "maximumAdapterTrainingRunCount",
    )
    controls_and_gates_ok = bool(
        len([key for key in controls if key.endswith("Control")]) == 8
        and controls["minimumDetectedOrDominated"]
        == gates["minimumControlsDetectedOrDominated"]
        == 6
        and gates["minimumPrimaryStrictOptimalSetMembershipRate"] == 0.80
        and gates["minimumPrimaryEpsilonOptimalMembershipRate"] == 0.95
        and gates["epsilonOptimalRegretNats"] == 0.001
        and gates["maximumPrimaryMeanSelectedEigRegretNats"] == 0.0015
        and gates["maximumPrimaryQ95SelectedEigRegretNats"] == 0.006
        and gates["maximumPrimarySelectedEigRegretNats"] == 0.02
        and gates["maximumPrimaryMeanAbsoluteEigVectorErrorNats"] == 0.004
        and gates["maximumPrimaryQ95AbsoluteEigVectorErrorNats"] == 0.015
        and gates["minimumImplementationMutantKillRate"] == 1.0
        and gates["minimumAnalyticFixturePassRate"] == 1.0
        and all(gates[key] == 0 for key in zero_gate_keys)
    )
    if not controls_and_gates_ok:
        errors.append("V65 controls or noncompensatory gates are inconsistent")

    boundary = config["claimBoundary"]
    boundary_ok = bool(
        boundary["externalModelArraysFromPOBAX"]
        and boundary["unknownActuatorFamilyProjectAuthored"]
        and boundary["pairedReuseOfFrozenV64PublicHistories"]
        and not boundary["independentExactBenchmarkReplication"]
        and boundary["pooledThreeRepeatSMC2Posterior"]
        and boundary["approximateOneStepStaticLatentEIG"]
        and boundary["oneStepAcquisitionDecisionPortability"]
        and not any(
            boundary[key]
            for key in (
                "sequentialApproximateAdaptiveRollout",
                "rewardPlanning",
                "formalVerification",
                "languageGrounding",
                "humanData",
                "modelAccess",
                "adapterTraining",
            )
        )
    )
    if not boundary_ok:
        errors.append("V65 paired-portability claim boundary is too broad or incomplete")

    stage = config["stageAuthorization"]
    firewall_ok = bool(
        set(config["firewall"].values()) == {"forbidden"}
        and stage["writeAndAuditSMC2EIGImplementation"]
        and not any(
            value
            for key, value in stage.items()
            if key != "writeAndAuditSMC2EIGImplementation"
        )
    )
    if not firewall_ok:
        errors.append("V65 design-only firewall or stage authorization is invalid")

    seed_values = [
        value for key, value in config["seeds"].items() if key.endswith("Seed")
    ]
    seeds_ok = (
        all(isinstance(value, int) for value in seed_values)
        and len(seed_values) == len(set(seed_values))
        and len(seed_values) == 8
    )
    if not seeds_ok:
        errors.append("V65 root seeds are missing, duplicated, or non-integral")

    downstream = (
        "configs/v65-implementation-lock.json",
        "configs/v65-subset-seal.json",
        "configs/v65-evaluation-implementation-lock.json",
        "configs/v65-outcome-lock.json",
        "data/v65-smc2-eig-portability",
        "outputs/v65-smc2-eig-portability/implementation-audit.json",
        "outputs/v65-smc2-eig-portability/evaluation",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream)
    if not downstream_absent:
        errors.append("V65 downstream artifacts exist before the design lock")

    source_v63r1 = sources["v63r1_outcome"]
    v63r1_ok = bool(
        source_v63r1["qualification_passed"]
        and source_v63r1["measurement_repair_not_independent_replication"]
        and not source_v63r1["original_v63_qualification_passed"]
        and not source_v63r1["authorization"]["modify_or_rerun_v63_or_v63r1"]
        and _bound_file(source_v63r1, "result", "result_sha256")
        and _bound_file(source_v63r1, "summary", "summary_sha256")
    )
    if not v63r1_ok:
        errors.append("V63r1 qualifying pooled-repeat boundary is not intact")

    checks = {
        "v64_outcome_authorization_and_binding": outcome_ok,
        "v64_design_implementation_population_and_exact_source_binding": v64_source_ok,
        "prospective_public_only_prefix_stratified_subset": subset_ok,
        "frozen_three_repeat_smc2_architecture_inherited": inherited_ok,
        "pool_before_score_static_latent_eig": acquisition_ok,
        "controls_and_noncompensatory_gates": controls_and_gates_ok,
        "paired_portability_claim_boundary": boundary_ok,
        "design_only_firewall": firewall_ok,
        "distinct_fresh_root_seeds": seeds_ok,
        "downstream_absent": downstream_absent,
        "v63r1_pooled_repeat_qualification_boundary": v63r1_ok,
    }
    audit = {
        "schema_version": 65,
        "experiment": "v65_preimplementation_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "authorize_v65_design_lock_and_implementation_only"
            if not errors and all(checks.values())
            else "repair_v65_design"
        ),
        "errors": errors,
        "checks": checks,
        "frozen_design_summary": {
            "source_public_records": reuse["sourceRecords"],
            "selected_records": subset["records"],
            "records_per_prefix_length": subset["recordsPerPrefixLength"],
            "prefix_lengths": subset["prefixLengths"],
            "outer_budgets": v65_smc["outerThetaParticleBudgets"],
            "primary_outer_budget": v65_smc["primaryOuterThetaParticleBudget"],
            "inner_state_particles": v65_smc["innerStateParticleBudget"],
            "independent_repeats": v65_smc["independentRepeatsPerBudget"],
            "primary_cells": subset["records"] * v65_smc["independentRepeatsPerBudget"],
            "all_inference_cells": (
                subset["records"]
                * len(v65_smc["outerThetaParticleBudgets"])
                * v65_smc["independentRepeatsPerBudget"]
            ),
            "pooling_rule": v65_smc["repeatAggregation"],
        },
        "data_access": {
            "v64_selection_public_file_hashed_but_records_loaded": 0,
            "v64_selection_audit_records_loaded": 0,
            "v64_evaluation_records_loaded_for_subset_selection": 0,
            "v65_subset_records_materialized": 0,
            "v65_candidate_evaluations": 0,
            "human_records": 0,
            "simulated_human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": 65,
        "experiment": "v65_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_bindings": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(path),
            }
            for name, path in source_paths.items()
        },
        "v64_selection_public": {
            "path": v64_population["files"]["selection_public"]["path"],
            "sha256": v64_population["files"]["selection_public"]["sha256"],
            "records": v64_population["counts"]["selection_public"],
        },
        "frozen_design_summary": audit["frozen_design_summary"],
        "authorization": {
            "modify_design": False,
            "write_and_audit_smc2_eig_implementation": True,
            "materialize_subset": False,
            "run_evaluation": False,
            "reward_planning": False,
            "formal_verification": False,
            "access_human_data": False,
            "simulate_human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
