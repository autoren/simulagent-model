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
    config_path = PROJECT_ROOT / "configs/v179-triple-repetition-robust-feasibility.json"
    parent_path = PROJECT_ROOT / "configs/v178-one-corruption-robust-certificate-feasibility-outcome-lock.json"
    source_v176_path = PROJECT_ROOT / "configs/v176-four-constraint-confirmation-population-outcome-lock.json"
    planner_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v178.md"
    plan_path = PROJECT_ROOT / "docs/v179-triple-repetition-robust-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v179_triple_repetition_robust_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v179_triple_repetition_robust_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v179_triple_repetition_robust_feasibility.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v179_triple_repetition_robust_feasibility_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v179_triple_repetition_robust_feasibility.py"
    audit_path = PROJECT_ROOT / "outputs/v179-triple-repetition-robust-feasibility/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v179-triple-repetition-robust-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v179-triple-repetition-robust-feasibility/census"
    outcome_path = PROJECT_ROOT / "configs/v179-triple-repetition-robust-feasibility-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V179 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    v178 = json.loads(parent_path.read_text())
    v176 = json.loads(source_v176_path.read_text())
    planner = json.loads(planner_path.read_text())
    states_path = PROJECT_ROOT / v176["constraint_states"]
    eligible_path = PROJECT_ROOT / v176["eligible_state_ids"]
    targets_path = PROJECT_ROOT / v176["target_cases"]
    population = config["populationUse"]
    code = config["fixedMeasurementCode"]
    census = config["structuralCensus"]
    authority = config["robustAuthorityGate"]
    gates = config["feasibilityGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V178_is_frozen_valid_single_pass_infeasibility_and_authorizes_only_fixed_repetition_design": bool(
            valid_lock(v178)
            and v178["outcome"]["passed"]
            and v178["outcome"]["scientific_feasibility_gates_passed"]
            and not v178["outcome"]["single_pass_target_blind_robust_feasibility_positive"]
            and v178["authorization"]["design_fixed_repeated_measurement_feasibility"]
            and not v178["authorization"]["design_separate_robust_planner_development"]
            and not v178["authorization"]["weaken_unanimity_or_add_posterior_threshold"]
            and not v178["authorization"]["modify_rerun_select_subsample_or_tune_V178"]
        ),
        "V176_population_and_V167_prior_are_exact_complete_declared_development_sources": bool(
            valid_lock(v176)
            and v176["outcome"]["passed"]
            and valid_lock(planner)
            and population["V176IsDevelopmentForV179"]
            and population["useEveryV176EligibleState"]
            and population["requiredStateCount"] == 135
            and population["useEveryV176Target"]
            and population["requiredTargetCount"] == 2160
            and population["useFrozenClassBalancedWeights"]
            and not population["selectionSubsamplingOrExclusionAllowed"]
            and all(
                path.is_file() and file_sha256(path) == v176[f"{key}_sha256"]
                for key, path in {
                    "constraint_states": states_path,
                    "eligible_state_ids": eligible_path,
                    "target_cases": targets_path,
                }.items()
            )
        ),
        "fixed_triple_code_is_selected_from_one_error_distance_and_hides_raw_bits_until_majority": bool(
            code["repetitionsPerSelectedValuation"] == 3
            and code["blockIsIndivisibleBeforePlannerObservation"]
            and code["maximumGlobalCorruptedRawOutcomes"] == 1
            and code["decoder"] == "strict binary majority"
            and code["decoderSelectedFromCodingDistanceNotV178TargetOutcomes"]
            and code["plannerObservesOnlyDecodedBlockBit"]
            and not code["singleAndDoubleRepetitionComparatorsScored"]
        ),
        "structural_census_is_exhaustive_cost_free_and_proves_decode_and_version_space_equivalence": bool(
            census["blockHorizons"] == [0, 1, 2, 3, 4]
            and census["correspondingMaximumRawInspectionCounts"] == [0, 3, 6, 9, 12]
            and census["enumerateNoCorruptionAndEverySingleRawFlip"]
            and census["proveMajorityDecodedBitEqualsTargetTruth"]
            and census["proveRawRobustVersionSpaceEqualsCleanDecodedVersionSpace"]
            and census["computeEveryTargetMinimalCertificateInBlocksAndRawInspections"]
            and census["computeTargetBlindWorstCaseTrustedCompletionByBlockHorizon"]
            and not census["queryCostOrRoutedLossUsed"]
        ),
        "authority_gate_remains_unanimous_and_excludes_provisional_planner_oracle_and_hidden_target": bool(
            authority["routeAliasOnlyIfEveryDecodedVersionSpaceCandidateIsAlias"]
            and authority["routeCompositionOnlyIfEveryDecodedVersionSpaceCandidateIsComposition"]
            and authority["deferMixedVersionSpaces"]
            and authority["deferUnanimousProvisionalVersionSpaces"]
            and not authority["plannerPosteriorOracleOrHiddenTargetCanAuthorizeCommit"]
            and not authority["provisionalPrimitiveMayEnterSandbox"]
        ),
        "gates_are_exact_structural_and_score_no_cost_or_sandbox": bool(
            gates["requiredStateCount"] == 135
            and gates["requiredTargetCount"] == 2160
            and gates["requiredFullMeasurementAdversarialScenarioCount"] == 28080
            and gates["requiredMajorityDecodeExactness"] == 1.0
            and gates["requiredRobustCleanVersionSpaceEquivalence"] == 1.0
            and gates["requiredFalseTrustedRouteProbability"] == 0.0
            and gates["maximumPlannerRiskOrCostScoreCount"] == 0
            and gates["maximumSandboxTransactionCount"] == 0
            and not config["decisionRule"]["passAuthorizesImmediatePlannerScoring"]
            and not config["decisionRule"]["passAuthorizesChangingRepetitionDecoderOrGate"]
            and not config["decisionRule"]["passAuthorizesModelRegistrationRealServiceOrExecution"]
        ),
        "prelock_exposure_is_one_state_and_no_formal_or_disallowed_access": bool(
            exposure["implementationUnitStateCount"] == 1
            and exposure["formalStateCount"] == 0
            and exposure["aggregateFormalMetricInspectionCount"] == 0
            and all(
                value == 0
                for key, value in exposure.items()
                if key
                not in {
                    "implementationUnitStateCount",
                    "formalStateCount",
                    "aggregateFormalMetricInspectionCount",
                }
            )
        ),
        "required_files_exist_and_formal_census_is_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    parent_path,
                    source_v176_path,
                    planner_path,
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
        "schema_version": "179-triple-repetition-robust-feasibility-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_V179_structural_census" if passed else "reject_V179_design",
        "checks": checks,
        "prelock_exposure": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V178_outcome": parent_path,
        "source_V176_outcome": source_v176_path,
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
        "schema_version": "179-triple-repetition-robust-feasibility-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "V167_config_payload": planner["config_payload"],
        "authorization": {
            "modify_population_code_decoder_certificate_policy_space_gate_metrics_or_decision": False,
            "run_formal_V179_census_once": True,
            "score_planner_risk_or_cost_or_run_sandbox": False,
            "change_repetition_decoder_or_unanimity_gate": False,
            "run_model_register_mutate_real_state_call_service_or_execute": False,
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
