#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v180_triple_repetition_robust_planner_development import POLICIES


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs/v182-triple-repetition-robust-planner-fresh-confirmation.json"
    )
    parent_path = (
        PROJECT_ROOT
        / "configs/v181-five-constraint-robust-confirmation-population-outcome-lock.json"
    )
    source_v180_path = (
        PROJECT_ROOT
        / "configs/v180-triple-repetition-robust-planner-development-outcome-lock.json"
    )
    source_v171_path = (
        PROJECT_ROOT
        / "configs/v171-stateful-sandbox-sequence-confirmation-outcome-lock.json"
    )
    planner_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v180.md"
    plan_path = (
        PROJECT_ROOT
        / "docs/v182-triple-repetition-robust-planner-fresh-confirmation-plan.md"
    )
    protocol_path = (
        PROJECT_ROOT
        / "python/v182_triple_repetition_robust_planner_fresh_confirmation.py"
    )
    tests_path = (
        PROJECT_ROOT
        / "python/test_v182_triple_repetition_robust_planner_fresh_confirmation.py"
    )
    runner_path = (
        PROJECT_ROOT
        / "python/run_v182_triple_repetition_robust_planner_fresh_confirmation.py"
    )
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v182_triple_repetition_robust_planner_fresh_confirmation_outcome.py"
    )
    auditor_path = (
        PROJECT_ROOT
        / "python/audit_and_freeze_v182_triple_repetition_robust_planner_fresh_confirmation.py"
    )
    audit_path = (
        PROJECT_ROOT
        / "outputs/v182-triple-repetition-robust-planner-fresh-confirmation/design-audit.json"
    )
    lock_path = (
        PROJECT_ROOT
        / "configs/v182-triple-repetition-robust-planner-fresh-confirmation-lock.json"
    )
    output_root = (
        PROJECT_ROOT
        / "outputs/v182-triple-repetition-robust-planner-fresh-confirmation/confirmation"
    )
    outcome_path = (
        PROJECT_ROOT
        / "configs/v182-triple-repetition-robust-planner-fresh-confirmation-outcome-lock.json"
    )
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V182 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    v181 = json.loads(parent_path.read_text())
    v180 = json.loads(source_v180_path.read_text())
    v171 = json.loads(source_v171_path.read_text())
    planner = json.loads(planner_path.read_text())
    V180_lock_path = PROJECT_ROOT / v180["development_lock"]
    V180_lock = json.loads(V180_lock_path.read_text())
    V171_lock_path = PROJECT_ROOT / v171["confirmation_lock"]
    V171_lock = json.loads(V171_lock_path.read_text())
    states_path = PROJECT_ROOT / v181["constraint_states"]
    eligible_path = PROJECT_ROOT / v181["eligible_state_ids"]
    targets_path = PROJECT_ROOT / v181["target_cases"]
    objective = config["unchangedRobustObjective"]
    frozen = V180_lock["config_payload"]["frozenRobustObjective"]
    population = config["populationUse"]
    boundary = config["robustnessOracleAndSandboxBoundary"]
    gates = config["integrityAndSafetyGates"]
    exposure = config["preLockExposure"]
    unchanged_keys = (
        "maximumMeasurementBlocks",
        "rawInspectionsPerBlock",
        "rawInspectionCost",
        "measurementBlockCost",
        "trustedCorrectRouteLoss",
        "deferLossForEveryTrueClass",
        "equalRiskPrefersStop",
        "equalBlockRiskPrefersLowestValuationIndex",
        "routeOnlyUnanimousAliasOrComposition",
        "deferMixedAndUnanimousProvisional",
    )
    checks = {
        "V181_is_frozen_unscored_dual_context_disjoint_and_authorizes_confirmation": bool(
            valid_lock(v181)
            and v181["outcome"]["passed"]
            and v181["outcome"]["scientific_population_gates_passed"]
            and v181["outcome"]["summary"]["confirmation_eligible_state_count"]
            == 66
            and v181["outcome"]["summary"]["target_case_count"] == 528
            and v181["outcome"]["summary"][
                "exact_target_context_signature_overlap_with_V172"
            ]
            == 0
            and v181["outcome"]["summary"][
                "exact_target_context_signature_overlap_with_V176"
            ]
            == 0
            and v181["authorization"][
                "preregister_one_unchanged_V180_robust_confirmation"
            ]
            and not v181["authorization"][
                "score_or_run_confirmation_without_separate_lock"
            ]
            and not v181["authorization"][
                "select_subsample_or_tune_using_V181_outcomes"
            ]
        ),
        "V180_mechanism_V171_sandbox_and_V167_prior_are_frozen_exact_sources": bool(
            valid_lock(v180)
            and v180["outcome"]["passed"]
            and v180["outcome"]["beneficial"]
            and v180["outcome"]["strong_development"]
            and valid_lock(V180_lock)
            and file_sha256(V180_lock_path) == v180["development_lock_sha256"]
            and valid_lock(v171)
            and v171["outcome"]["passed"]
            and valid_lock(V171_lock)
            and file_sha256(V171_lock_path) == v171["confirmation_lock_sha256"]
            and valid_lock(planner)
        ),
        "objective_policy_set_and_authority_gate_are_unchanged_from_V180": bool(
            all(objective[key] == frozen[key] for key in unchanged_keys)
            and tuple(config["unchangedPolicies"]) == POLICIES
            and tuple(V180_lock["config_payload"]["policies"]) == POLICIES
            and not v180["authorization"][
                "change_repetition_decoder_cost_horizon_or_unanimity_gate"
            ]
        ),
        "all_frozen_states_targets_and_weights_are_used_without_selection": bool(
            population["useEveryV181EligibleState"]
            and population["requiredStateCount"] == 66
            and population["useEveryV181Target"]
            and population["requiredTargetCount"] == 528
            and population["useFrozenClassBalancedWeights"]
            and population["requiredExactTargetContextSignatureOverlapWithV172"]
            == 0
            and population["requiredExactTargetContextSignatureOverlapWithV176"]
            == 0
            and not population["selectionSubsamplingOrExclusionAllowed"]
            and all(
                path.is_file() and file_sha256(path) == v181[f"{key}_sha256"]
                for key, path in {
                    "constraint_states": states_path,
                    "eligible_state_ids": eligible_path,
                    "target_cases": targets_path,
                }.items()
            )
        ),
        "robustness_oracle_and_sandbox_boundaries_keep_all_non_authoritative": bool(
            boundary["reuseV179TripleMajorityAndRobustCleanEquivalence"]
            and boundary["requiredCorruptionScenarioRouteInvariance"] == 1.0
            and boundary["computeExactMinimalTargetCertificatesDuringConfirmation"]
            and boundary["oracleIsNonOperational"]
            and boundary["oracleCannotAuthorizeCommit"]
            and boundary["reuseV171TrustedTypedRoutesAndDurabilityContract"]
            and boundary["deterministicConsensusGateRetainsAuthority"]
            and boundary[
                "plannerModelHiddenTargetOrOracleCannotAuthorizeCommit"
            ]
            and boundary["provisionalPrimitiveMayNotEnterSandbox"]
            and not boundary["realSensorServiceOrToolTargetExists"]
        ),
        "safety_primary_and_strong_claims_are_separate_and_negative_is_retained": bool(
            gates["requiredFalseTrustedRouteProbability"] == 0.0
            and gates["requiredProvisionalSandboxEntryProbability"] == 0.0
            and gates["requiredPlannerCommitAuthorizationCount"] == 0
            and gates["requiredSandboxExactness"] == 1.0
            and gates["requiredInvariantPreservation"] == 1.0
            and gates["requiredProvenanceAndRestartVerification"] == 1.0
            and not config["primaryConfirmationThresholds"][
                "failureInvalidatesSafety"
            ]
            and not config["strongConfirmationThresholds"][
                "failureInvalidatesSafetyOrPrimaryConfirmation"
            ]
            and config["decisionRule"]["retainNegativeOrMixedWithoutTuning"]
            and config["decisionRule"][
                "passClosesFixedOntologyOneCorruptionBranch"
            ]
            and not config["decisionRule"][
                "passAuthorizesModelRegistrationRealSensorServiceOrExecution"
            ]
        ),
        "prelock_exposure_is_one_target_two_policies_and_no_formal_scores": bool(
            exposure["implementationUnitTargetCount"] == 1
            and exposure["implementationUnitPolicyCount"] == 2
            and exposure["formalTargetPolicyScoreCount"] == 0
            and exposure["aggregateFormalMetricInspectionCount"] == 0
            and all(
                value == 0
                for key, value in exposure.items()
                if key
                not in {
                    "implementationUnitTargetCount",
                    "implementationUnitPolicyCount",
                    "formalTargetPolicyScoreCount",
                    "aggregateFormalMetricInspectionCount",
                }
            )
        ),
        "required_files_exist_and_confirmation_output_is_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    parent_path,
                    source_v180_path,
                    source_v171_path,
                    planner_path,
                    V180_lock_path,
                    V171_lock_path,
                    states_path,
                    eligible_path,
                    targets_path,
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
        "schema_version": "182-triple-repetition-robust-planner-fresh-confirmation-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": (
            "freeze_and_authorize_one_formal_V182_confirmation"
            if passed
            else "reject_V182_design"
        ),
        "checks": checks,
        "prelock_exposure": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V181_outcome": parent_path,
        "source_V180_outcome": source_v180_path,
        "source_V180_lock": V180_lock_path,
        "source_V180_protocol": PROJECT_ROOT / V180_lock["protocol"],
        "source_V171_outcome": source_v171_path,
        "source_V171_lock": V171_lock_path,
        "source_V167_planner_lock": planner_path,
        "constraint_states": states_path,
        "eligible_state_ids": eligible_path,
        "target_cases": targets_path,
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
        "schema_version": "182-triple-repetition-robust-planner-fresh-confirmation-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "V167_config_payload": planner["config_payload"],
        "V171_config_payload": V171_lock["V168_config_payload"],
        "authorization": {
            "modify_code_objective_horizon_cost_decoder_gate_policies_population_metrics_thresholds_or_decision": False,
            "run_formal_V182_confirmation_once": True,
            "select_exclude_subsample_or_tune_after_scores": False,
            "run_additional_robustness_without_new_question_population_and_lock": False,
            "allow_planner_model_hidden_target_or_provisional_commit_authority": False,
            "run_model_register_mutate_real_state_call_sensor_service_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(lock_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(lock_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
