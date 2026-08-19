#!/usr/bin/env python3
from __future__ import annotations

from fractions import Fraction
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


def as_fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v180-triple-repetition-robust-planner-development.json"
    parent_path = PROJECT_ROOT / "configs/v179-triple-repetition-robust-feasibility-outcome-lock.json"
    source_v177_path = PROJECT_ROOT / "configs/v177-certification-aware-planner-fresh-confirmation-outcome-lock.json"
    source_v171_path = PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-outcome-lock.json"
    source_v176_path = PROJECT_ROOT / "configs/v176-four-constraint-confirmation-population-outcome-lock.json"
    planner_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    plan_path = PROJECT_ROOT / "docs/v180-triple-repetition-robust-planner-development-plan.md"
    protocol_path = PROJECT_ROOT / "python/v180_triple_repetition_robust_planner_development.py"
    tests_path = PROJECT_ROOT / "python/test_v180_triple_repetition_robust_planner_development.py"
    runner_path = PROJECT_ROOT / "python/run_v180_triple_repetition_robust_planner_development.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v180_triple_repetition_robust_planner_development_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v180_triple_repetition_robust_planner_development.py"
    audit_path = PROJECT_ROOT / "outputs/v180-triple-repetition-robust-planner-development/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v180-triple-repetition-robust-planner-development-lock.json"
    output_root = PROJECT_ROOT / "outputs/v180-triple-repetition-robust-planner-development/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v180-triple-repetition-robust-planner-development-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V180 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    v179 = json.loads(parent_path.read_text())
    v177 = json.loads(source_v177_path.read_text())
    v171 = json.loads(source_v171_path.read_text())
    v176 = json.loads(source_v176_path.read_text())
    planner = json.loads(planner_path.read_text())
    V171_lock_path = PROJECT_ROOT / v171["confirmation_lock"]
    V171_lock = json.loads(V171_lock_path.read_text())
    states_path = PROJECT_ROOT / v176["constraint_states"]
    eligible_path = PROJECT_ROOT / v176["eligible_state_ids"]
    targets_path = PROJECT_ROOT / v176["target_cases"]
    certificates_path = PROJECT_ROOT / v179["target_certificate_results"]
    objective = config["frozenRobustObjective"]
    curve = v179["outcome"]["summary"]["adaptive_worst_case_trusted_completion_by_block_horizon"]
    first_positive = next(int(key) for key in sorted(curve, key=int) if as_fraction(curve[key]) > 0)
    population = config["populationUse"]
    boundary = config["robustnessAndSandboxBoundary"]
    gates = config["integrityAndSafetyGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V179_is_frozen_positive_exact_feasibility_and_authorizes_only_separate_planner_design": bool(
            valid_lock(v179)
            and v179["outcome"]["passed"]
            and v179["outcome"]["scientific_feasibility_gates_passed"]
            and v179["outcome"]["triple_repetition_target_blind_robust_feasibility_positive"]
            and v179["authorization"]["design_separate_triple_repetition_robust_planner_development"]
            and not v179["authorization"]["run_planner_without_separate_design_lock"]
            and not v179["authorization"]["modify_rerun_select_subsample_or_tune_V179"]
        ),
        "horizon_and_block_cost_are_prospectively_derived_without_target_risk_scores": bool(
            first_positive == 4
            and objective["maximumMeasurementBlocks"] == 4
            and objective["rawInspectionsPerBlock"] == 3
            and objective["rawInspectionCost"] == planner["config_payload"]["queryModel"]["queryCost"] == "1/10"
            and objective["measurementBlockCost"] == "3/10"
        ),
        "V177_clean_mechanism_and_V171_sandbox_are_frozen_positive_sources": bool(
            valid_lock(v177)
            and v177["outcome"]["passed"]
            and v177["outcome"]["confirmed"]
            and valid_lock(v171)
            and v171["outcome"]["passed"]
            and valid_lock(V171_lock)
            and file_sha256(V171_lock_path) == v171["confirmation_lock_sha256"]
            and valid_lock(planner)
        ),
        "all_V176_development_states_targets_weights_and_V179_certificates_are_exact": bool(
            valid_lock(v176)
            and population["V176IsDevelopmentForV180"]
            and population["useEveryEligibleState"]
            and population["requiredStateCount"] == 135
            and population["useEveryTarget"]
            and population["requiredTargetCount"] == 2160
            and population["useFrozenClassBalancedWeights"]
            and not population["selectionSubsamplingOrExclusionAllowed"]
            and all(
                path.is_file() and file_sha256(path) == v176[f"{key}_sha256"]
                for key, path in {"constraint_states": states_path, "eligible_state_ids": eligible_path, "target_cases": targets_path}.items()
            )
            and certificates_path.is_file()
            and file_sha256(certificates_path) == v179["target_certificate_results_sha256"]
        ),
        "policy_set_cost_gate_robustness_and_sandbox_boundaries_are_frozen": bool(
            tuple(config["policies"]) == POLICIES
            and objective["trustedCorrectRouteLoss"] == 0
            and objective["deferLossForEveryTrueClass"] == 2
            and objective["equalRiskPrefersStop"]
            and objective["equalBlockRiskPrefersLowestValuationIndex"]
            and objective["routeOnlyUnanimousAliasOrComposition"]
            and objective["deferMixedAndUnanimousProvisional"]
            and boundary["reuseV179DecodeAndRobustCleanEquivalence"]
            and boundary["requiredCorruptionScenarioRouteInvariance"] == 1.0
            and boundary["reuseV171TrustedTypedRoutesAndDurabilityContract"]
            and boundary["deterministicConsensusGateRetainsAuthority"]
            and boundary["plannerCleanPolicyOracleHiddenTargetOrModelMayNotAuthorizeCommit"]
            and boundary["provisionalPrimitiveMayNotEnterSandbox"]
            and not boundary["realSensorServiceOrToolTargetExists"]
        ),
        "safety_benefit_and_strong_outcomes_are_separate_and_negative_is_retained": bool(
            gates["requiredFalseTrustedRouteProbability"] == 0.0
            and gates["requiredProvisionalSandboxEntryProbability"] == 0.0
            and gates["requiredPlannerCommitAuthorizationCount"] == 0
            and not config["benefitThresholds"]["failureInvalidatesSafety"]
            and not config["strongDevelopmentThresholds"]["failureInvalidatesSafetyOrBenefit"]
            and config["decisionRule"]["retainNegativeOrMixedWithoutTuning"]
            and not config["decisionRule"]["passAuthorizesImmediateFreshConfirmation"]
            and not config["decisionRule"]["passAuthorizesChangingCodeCostDecoderOrGate"]
            and not config["decisionRule"]["passAuthorizesModelRegistrationRealServiceOrExecution"]
        ),
        "prelock_exposure_is_one_target_two_policies_and_zero_formal_scores": bool(
            exposure["implementationUnitTargetCount"] == 1
            and exposure["implementationUnitPolicyCount"] == 2
            and exposure["formalTargetPolicyScoreCount"] == 0
            and exposure["aggregateFormalMetricInspectionCount"] == 0
            and all(value == 0 for key, value in exposure.items() if key not in {"implementationUnitTargetCount", "implementationUnitPolicyCount", "formalTargetPolicyScoreCount", "aggregateFormalMetricInspectionCount"})
        ),
        "required_files_exist_and_formal_output_is_absent": bool(
            all(path.is_file() for path in (config_path, parent_path, source_v177_path, source_v171_path, source_v176_path, planner_path, V171_lock_path, states_path, eligible_path, targets_path, certificates_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {"schema_version": "180-triple-repetition-robust-planner-development-design-audit", "experiment": config["experiment"], "passed": passed, "decision": "freeze_and_authorize_one_formal_V180_development_run" if passed else "reject_V180_design", "checks": checks, "first_positive_block_horizon": first_positive, "prelock_exposure": exposure}
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {"config": config_path, "parent_V179_outcome": parent_path, "source_V177_outcome": source_v177_path, "source_V171_outcome": source_v171_path, "source_V171_lock": V171_lock_path, "source_V176_outcome": source_v176_path, "source_V167_planner_lock": planner_path, "constraint_states": states_path, "eligible_state_ids": eligible_path, "target_cases": targets_path, "target_certificate_results": certificates_path, "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "180-triple-repetition-robust-planner-development-lock", "experiment": config["experiment"], "config_payload": config, "V167_config_payload": planner["config_payload"], "V171_config_payload": V171_lock["V168_config_payload"], "authorization": {"modify_code_cost_horizon_objective_policies_population_gate_metrics_thresholds_or_decision": False, "run_formal_V180_development_once": True, "select_exclude_subsample_or_tune_after_scores": False, "run_confirmation_without_separate_population_and_lock": False, "change_decoder_or_unanimity_gate": False, "run_model_register_mutate_real_state_call_sensor_service_or_execute": False}}
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
