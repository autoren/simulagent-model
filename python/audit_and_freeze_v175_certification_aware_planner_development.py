#!/usr/bin/env python3
from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def as_fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v175-certification-aware-planner-development.json"
    parent_path = PROJECT_ROOT / "configs/v174-certificate-depth-feasibility-census-outcome-lock.json"
    source_v173_path = PROJECT_ROOT / "configs/v173-trusted-only-shadow-integration-outcome-lock.json"
    source_v171_path = PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-outcome-lock.json"
    source_v172_path = PROJECT_ROOT / "configs/v172-trusted-shadow-integration-population-outcome-lock.json"
    planner_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v173.md"
    plan_path = PROJECT_ROOT / "docs/v175-certification-aware-planner-development-plan.md"
    protocol_path = PROJECT_ROOT / "python/v175_certification_aware_planner_development.py"
    tests_path = PROJECT_ROOT / "python/test_v175_certification_aware_planner_development.py"
    runner_path = PROJECT_ROOT / "python/run_v175_certification_aware_planner_development.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v175_certification_aware_planner_development_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v175_certification_aware_planner_development.py"
    audit_path = PROJECT_ROOT / "outputs/v175-certification-aware-planner-development/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v175-certification-aware-planner-development-lock.json"
    output_root = PROJECT_ROOT / "outputs/v175-certification-aware-planner-development/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v175-certification-aware-planner-development-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V175 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    v173 = json.loads(source_v173_path.read_text())
    v171 = json.loads(source_v171_path.read_text())
    v172 = json.loads(source_v172_path.read_text())
    planner = json.loads(planner_path.read_text())
    v171_lock_path = PROJECT_ROOT / v171["confirmation_lock"]
    v171_lock = json.loads(v171_lock_path.read_text())
    states_path = PROJECT_ROOT / v172["constraint_states"]
    eligible_path = PROJECT_ROOT / v172["eligible_state_ids"]
    targets_path = PROJECT_ROOT / v172["target_cases"]
    certificates_path = PROJECT_ROOT / parent["target_certificate_results"]
    curve = parent["outcome"]["summary"]["adaptive_trusted_completion_by_horizon"]
    first_positive = next(
        int(horizon)
        for horizon, value in sorted(curve.items(), key=lambda item: int(item[0]))
        if as_fraction(value) > 0
    )
    objective = config["frozenRoutedObjective"]
    population = config["populationUse"]
    boundary = config["sandboxBoundary"]
    gates = config["integrityAndSafetyGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V174_is_frozen_exact_and_authorizes_separate_V175_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_feasibility_gates_passed"]
            and parent["authorization"]["use_structural_horizon_curve_to_preregister_V175"]
            and not parent["authorization"]["score_V175_without_separate_design_lock"]
            and not parent["authorization"]["modify_rerun_select_or_tune_V174"]
        ),
        "horizon_five_is_prospectively_selected_as_first_positive_certificate_horizon": bool(
            first_positive == 5
            and objective["maximumQueries"] == first_positive
            and all(as_fraction(curve[str(horizon)]) == 0 for horizon in range(5))
            and as_fraction(curve["5"]) > 0
        ),
        "V173_safe_nonbeneficial_outcome_is_retained_without_repair_or_rerun": bool(
            valid_lock(v173)
            and v173["outcome"]["passed"]
            and v173["outcome"]["scientific_safety_gates_passed"]
            and not v173["outcome"]["beneficial"]
            and not v173["authorization"]["modify_rerun_select_subsample_or_tune_V173"]
            and not v173["authorization"]["design_nonoverlapping_integration_confirmation"]
        ),
        "V171_sandbox_and_V167_prior_are_exact_frozen_sources": bool(
            valid_lock(v171)
            and v171["outcome"]["passed"]
            and v171["outcome"]["scientific_confirmation_gates_passed"]
            and valid_lock(v171_lock)
            and file_sha256(v171_lock_path) == v171["confirmation_lock_sha256"]
            and valid_lock(planner)
            and objective["queryCost"] == planner["config_payload"]["queryModel"]["queryCost"]
        ),
        "all_V172_development_states_targets_and_weights_are_used_without_selection": bool(
            valid_lock(v172)
            and v172["outcome"]["passed"]
            and population["V172IsDevelopment"]
            and population["useAllEligibleStates"]
            and population["requiredStateCount"] == 132
            and population["useAllTargets"]
            and population["requiredTargetCount"] == 4224
            and population["useFrozenClassBalancedWeights"]
            and not population["selectionSubsamplingOrExclusionAllowed"]
            and all(
                path.is_file() and file_sha256(path) == v172[f"{key}_sha256"]
                for key, path in {
                    "constraint_states": states_path,
                    "eligible_state_ids": eligible_path,
                    "target_cases": targets_path,
                }.items()
            )
            and certificates_path.is_file()
            and file_sha256(certificates_path) == parent["target_certificate_results_sha256"]
        ),
        "routed_objective_keeps_consensus_gate_and_noncompensatory_safety": bool(
            objective["maximumQueries"] == 5
            and objective["queryCost"] == "1/10"
            and objective["trustedCorrectRouteLoss"] == 0
            and objective["deferLossForEveryTrueClass"] == 2
            and not objective["falseTrustedRouteAllowed"]
            and not objective["provisionalSandboxEntryAllowed"]
            and objective["equalRiskPrefersStop"]
            and objective["equalQueryRiskPrefersLowestValuationIndex"]
            and boundary["reuseV171TrustedTypedRoutesAndDurabilityContract"]
            and boundary["deterministicConsensusGateRetainsAuthority"]
            and boundary["plannerMayNotAuthorizeCommit"]
            and not boundary["realServiceOrToolTargetExists"]
            and gates["requiredFalseTrustedRouteProbability"] == 0.0
            and gates["requiredProvisionalSandboxEntryProbability"] == 0.0
            and gates["requiredPlannerCommitAuthorizationCount"] == 0
        ),
        "benefit_and_strong_claims_are_separate_from_safety_and_negative_is_retained": bool(
            not config["benefitThresholds"]["failureInvalidatesSafety"]
            and not config["strongDevelopmentThresholds"]["failureInvalidatesSafetyOrBenefit"]
            and config["decisionRule"]["retainNegativeOrMixedWithoutTuning"]
            and not config["decisionRule"]["passAuthorizesImmediateFreshConfirmation"]
            and not config["decisionRule"]["passAuthorizesModelRegistrationRealServiceOrExecution"]
        ),
        "prelock_exposure_matches_one_target_two_policy_unit_test_and_no_formal_scores": bool(
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
        "required_files_exist_and_formal_output_is_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    parent_path,
                    source_v173_path,
                    source_v171_path,
                    source_v172_path,
                    planner_path,
                    v171_lock_path,
                    states_path,
                    eligible_path,
                    targets_path,
                    certificates_path,
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
        "schema_version": "175-certification-aware-planner-development-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_V175_development_run" if passed else "reject_V175_design",
        "checks": checks,
        "first_positive_certificate_horizon": first_positive,
        "prelock_exposure": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V174_outcome": parent_path,
        "source_V173_outcome": source_v173_path,
        "source_V171_outcome": source_v171_path,
        "source_V171_lock": v171_lock_path,
        "source_V172_outcome": source_v172_path,
        "source_V167_planner_lock": planner_path,
        "constraint_states": states_path,
        "eligible_state_ids": eligible_path,
        "target_cases": targets_path,
        "target_certificate_results": certificates_path,
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
        "schema_version": "175-certification-aware-planner-development-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "V167_config_payload": planner["config_payload"],
        "V171_config_payload": v171_lock["V168_config_payload"],
        "authorization": {
            "modify_objective_horizon_cost_gate_policies_population_metrics_thresholds_or_decision": False,
            "run_formal_V175_development_once": True,
            "select_exclude_subsample_or_tune_after_scores": False,
            "run_fresh_confirmation_without_separate_population_and_lock": False,
            "allow_planner_model_hidden_target_or_provisional_commit_authority": False,
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
