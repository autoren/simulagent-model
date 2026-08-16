#!/usr/bin/env python3
"""Audit and freeze the V66 external Bayes-adaptive reward design."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v66-external-bayes-adaptive-reward.json"
    plan_path = PROJECT_ROOT / "docs/v66-external-bayes-adaptive-reward-plan.md"
    audit_path = PROJECT_ROOT / "outputs/v66-external-bayes-adaptive-reward/design-audit.json"
    output_path = PROJECT_ROOT / "configs/v66-design-lock.json"
    if output_path.exists():
        raise RuntimeError("V66 design already frozen")

    config = json.loads(config_path.read_text())
    outcome_path = PROJECT_ROOT / config["sourceV65r3OutcomeLock"]
    outcome = json.loads(outcome_path.read_text())
    evaluator_path = PROJECT_ROOT / outcome["evaluation_implementation_lock"]
    evaluator = json.loads(evaluator_path.read_text())
    subset_seal_path = PROJECT_ROOT / evaluator["subset_seal"]
    subset_seal = json.loads(subset_seal_path.read_text())
    errors: list[str] = []

    outcome_payload = {
        key: value for key, value in outcome.items() if key != "lock_payload_sha256"
    }
    outcome_ok = bool(
        payload_hash(outcome_payload) == outcome["lock_payload_sha256"]
        and outcome["decision"]
        == "authorize_preregistration_of_external_Bayes_adaptive_reward_decisions"
        and outcome["authorization"][
            "preregister_external_Bayes_adaptive_reward_decisions"
        ]
        and not outcome["authorization"]["modify_or_rerun_v65r3"]
        and not outcome["authorization"][
            "run_reward_decision_evaluation_before_preregistration_and_locks"
        ]
        and file_sha256(evaluator_path)
        == outcome["evaluation_implementation_lock_sha256"]
        and file_sha256(PROJECT_ROOT / outcome["result"])
        == outcome["result_sha256"]
        and file_sha256(PROJECT_ROOT / outcome["outcome_audit"])
        == outcome["outcome_audit_sha256"]
        and file_sha256(PROJECT_ROOT / outcome["outcome_auditor"])
        == outcome["outcome_auditor_sha256"]
    )
    if not outcome_ok:
        errors.append("V65r3 outcome binding or reward-planning preregistration authorization failed")

    subset_ok = bool(
        file_sha256(subset_seal_path) == evaluator["subset_seal_sha256"]
        and subset_seal["experiment"] == "v65r1_subset_seal"
        and subset_seal["counts"] == {
            "subset_provenance": 48,
            "subset_public": 48,
        }
        and subset_seal["files"]["subset_public"]["path"]
        == "data/v65-smc2-eig-portability/subset-public.jsonl"
    )
    population = config["population"]
    population_ok = bool(
        subset_ok
        and population["records"] == 48
        and population["prefixLengths"] == [0, 1, 2, 3, 4, 5]
        and population["recordsPerPrefixLength"] == 8
        and not population["reselectionRejectionOrReordering"]
        and not population["truthFields"]
    )
    if not population_ok:
        errors.append("V66 does not bind the unchanged 48-record public subset")

    planning = config["planning"]
    approximate = config["approximatePosterior"]
    planning_ok = bool(
        planning["horizonActions"] == 3
        and planning["discount"] == 0.95
        and planning["candidateOrder"] == ["n", "e", "s", "w"]
        and planning["observations"] == 6
        and planning["physicalStates"] == 11
        and planning["exactQuadratureNodes"] == 257
        and planning["staticModelPersistence"].startswith("one_identity_theta_pair_is_fixed")
        and approximate["outerThetaParticlesPerIdentity"] == 509
        and approximate["innerStateParticles"] == 127
        and approximate["independentRepeats"] == 3
        and approximate["pooling"] == "equal_weight_posterior_mixture_before_planning"
        and approximate["dynamicStateForPlanning"].startswith("Rao_Blackwellized")
        and approximate["freshV66Inference"]
        and not approximate["V65r3EvaluationRerun"]
    )
    if not planning_ok:
        errors.append("V66 planning horizon, exact reference, or pooled SMC2 budget is invalid")

    strategies = config["strategies"]
    mixture = config["persistentMixtureQuadrature"]
    strategy_ok = bool(
        set(strategies)
        == {
            "exactBayesAdaptive",
            "pooledSMC2BayesAdaptive",
            "posteriorWeightedModelOracle",
            "jointMAPCertaintyEquivalent",
            "persistentPosteriorSamplingMixture",
            "myopicExpectedReward",
            "informationOnlyEIG",
            "invalidMeanTransition",
        }
        and "static_model_updating" in strategies["exactBayesAdaptive"]
        and "static_weight_updates" in strategies["pooledSMC2BayesAdaptive"]
        and "fixed_for_the_entire_policy_tree" in mixture["sampledModelPersistence"]
        and mixture["primaryPoints"] == 32
        and mixture["primarySystematicOffset"] == 0.5 / 32
        and mixture["sensitivityPoints"] == 64
        and mixture["sensitivitySystematicOffset"] == 0.5 / 64
        and not mixture["randomMonteCarlo"]
        and "full_exact_root_joint_posterior_predictive" in mixture[
            "environmentEvaluation"
        ]
        and "not_an_exact_514_atom_enumeration" in mixture["role"]
        and "never_be_described_as_a_valid" in strategies["invalidMeanTransition"]
    )
    if not strategy_ok:
        errors.append("V66 strategy set or persistent-mixture semantics is invalid")

    semantics = config["evaluationSemantics"]
    gates = config["gates"]
    evaluation_ok = bool(
        semantics["commonEnvironment"].startswith("exact_joint_posterior_predictive")
        and "exact_joint_belief" in semantics["policyEvaluation"]
        and not semantics["realizedTrajectorySimulation"]
        and not semantics["truthAuditAccess"]
        and gates["maximumMeanSMC2PolicyValueRegret"] == 0.005
        and gates["maximumQ95SMC2PolicyValueRegret"] == 0.02
        and gates["maximumSMC2PolicyValueRegret"] == 0.08
        and gates["minimumStrictExactRootOptimalSetMembershipRate"] == 0.85
        and gates["minimumEpsilonOptimalRootMembershipRate"] == 0.95
        and gates["epsilonOptimalRootReward"] == 0.005
        and gates["maximumOracleDominanceResidual"] == 1e-10
        and gates["minimumImplementationMutantKillRate"] == 1.0
        and gates["minimumAnalyticFixturePassRate"] == 1.0
        and gates["maximumUnexpectedEvaluationAttemptCount"] == 0
        and gates["maximumTruthFieldAccessCount"] == 0
        and gates["maximumHumanRecordAccessCount"] == 0
        and gates["maximumModelForwardPassCount"] == 0
        and gates["maximumAdapterTrainingRunCount"] == 0
    )
    if not evaluation_ok:
        errors.append("V66 common evaluation semantics or noncompensatory gates are incomplete")

    exact_audit = config["exactImplementationAudit"]
    exact_audit_ok = bool(
        all(exact_audit.values())
        and exact_audit["independentScalarReference"]
        and exact_audit["pointMassStaticBeliefReducesToKnownModelPOMDP"]
        and exact_audit["persistentMixtureSystematicQuantilesMatchIndependentInverseCDFReference"]
        and exact_audit["persistentMixture64PointSensitivityReported"]
    )
    if not exact_audit_ok:
        errors.append("V66 exact implementation audit requirements are incomplete")

    boundary = config["claimBoundary"]
    stage = config["stageAuthorization"]
    boundary_ok = bool(
        boundary["externalArraysPinned"]
        and boundary["unknownDynamicsProjectAuthored"]
        and boundary["boundedHorizon"] == 3
        and boundary["exactQuadratureReference"]
        and boundary["approximateStaticPosterior"] == "pooled_SMC2"
        and not any(
            boundary[key]
            for key in (
                "infiniteHorizonOptimality",
                "independentBenchmarkReplication",
                "formalVerification",
                "safetyProperty",
                "humanData",
                "modelAccess",
                "adapterTraining",
            )
        )
        and set(config["firewall"].values()) == {"forbidden"}
        and stage["writeAndAuditPlannerImplementation"]
        and not any(
            value
            for key, value in stage.items()
            if key != "writeAndAuditPlannerImplementation"
        )
    )
    if not boundary_ok:
        errors.append("V66 boundary, firewall, or implementation-only authorization is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v66-implementation-lock.json",
            "configs/v66-evaluation-implementation-lock.json",
            "configs/v66-outcome-lock.json",
            "configs/v67-design-lock.json",
            "python/v66_bayes_adaptive_reward.py",
            "python/evaluate_v66_reward.py",
            "outputs/v66-external-bayes-adaptive-reward/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V66 implementation, evaluation, or verification exists before design lock")

    checks = {
        "V65r3_success_and_reward_preregistration_authorization": outcome_ok,
        "unchanged_48_record_public_subset_bound": population_ok,
        "horizon_reference_and_pooled_SMC2_budget_frozen": planning_ok,
        "strategy_set_and_persistent_mixture_semantics_frozen": strategy_ok,
        "common_exact_evaluation_and_noncompensatory_gates": evaluation_ok,
        "independent_exact_implementation_audits_required": exact_audit_ok,
        "claim_boundary_firewall_and_implementation_only_authorization": boundary_ok,
        "V66_downstream_absent": downstream_absent,
    }
    audit = {
        "schema_version": "66",
        "experiment": "v66_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_v66_design_and_authorize_planner_implementation_only"
            if not errors and all(checks.values())
            else "reject_v66_design"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "sealed_public_records_loaded": 0,
            "sealed_reward_policies_evaluated": 0,
            "V64_or_V65_evaluation_result_records_loaded": 0,
            "truth_fields_accessed": 0,
            "V65r3_evaluation_reruns": 0,
            "V66_evaluation_attempts": 0,
            "human_records": 0,
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
        "schema_version": "66",
        "experiment": "v66_design_lock",
        "source_v65r3_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_v65r3_outcome_lock_sha256": file_sha256(outcome_path),
        "source_v65r3_evaluation_implementation_lock": str(
            evaluator_path.relative_to(PROJECT_ROOT)
        ),
        "source_v65r3_evaluation_implementation_lock_sha256": file_sha256(evaluator_path),
        "subset_seal": str(subset_seal_path.relative_to(PROJECT_ROOT)),
        "subset_seal_sha256": file_sha256(subset_seal_path),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_or_rerun_v65r3": False,
            "modify_v66_design": False,
            "write_and_audit_planner_implementation": True,
            "write_and_audit_durable_evaluator": False,
            "run_evaluation": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "checks": checks,
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
