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
    config_path = PROJECT_ROOT / "configs/v174-certificate-depth-feasibility-census.json"
    parent_path = PROJECT_ROOT / "configs/v173-trusted-only-shadow-integration-outcome-lock.json"
    source_path = PROJECT_ROOT / "configs/v172-trusted-shadow-integration-population-outcome-lock.json"
    planner_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v173.md"
    plan_path = PROJECT_ROOT / "docs/v174-certificate-depth-feasibility-census-plan.md"
    protocol_path = PROJECT_ROOT / "python/v174_certificate_depth_feasibility_census.py"
    tests_path = PROJECT_ROOT / "python/test_v174_certificate_depth_feasibility_census.py"
    runner_path = PROJECT_ROOT / "python/run_v174_certificate_depth_feasibility_census.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v174_certificate_depth_feasibility_census_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v174_certificate_depth_feasibility_census.py"
    audit_path = PROJECT_ROOT / "outputs/v174-certificate-depth-feasibility-census/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v174-certificate-depth-feasibility-census-lock.json"
    output_root = PROJECT_ROOT / "outputs/v174-certificate-depth-feasibility-census/census"
    outcome_path = PROJECT_ROOT / "configs/v174-certificate-depth-feasibility-census-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V174 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    source = json.loads(source_path.read_text())
    planner = json.loads(planner_path.read_text())
    states_path = PROJECT_ROOT / source["constraint_states"]
    eligible_path = PROJECT_ROOT / source["eligible_state_ids"]
    targets_path = PROJECT_ROOT / source["target_cases"]
    population = config["populationUse"]
    certificate = config["certificateDefinition"]
    adaptive = config["adaptiveOpportunity"]
    gates = config["feasibilityGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V173_is_frozen_safe_nonbeneficial_and_not_open_for_confirmation_or_change": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_safety_gates_passed"]
            and not parent["outcome"]["beneficial"]
            and not parent["outcome"]["strong_integration"]
            and not parent["authorization"]["design_nonoverlapping_integration_confirmation"]
            and not parent["authorization"]["modify_rerun_select_subsample_or_tune_V173"]
        ),
        "V172_population_and_all_target_artifacts_are_exact": bool(
            valid_lock(source)
            and source["outcome"]["passed"]
            and source["outcome"]["summary"]["integration_eligible_state_count"] == 132
            and source["outcome"]["summary"]["target_case_count"] == 4224
            and all(
                path.is_file() and file_sha256(path) == source[f"{key}_sha256"]
                for key, path in {
                    "constraint_states": states_path,
                    "eligible_state_ids": eligible_path,
                    "target_cases": targets_path,
                }.items()
            )
        ),
        "population_is_declared_development_complete_and_unselected": bool(
            population["useAllV172EligibleStates"]
            and population["requiredStateCount"] == 132
            and population["useAllV172TargetCases"]
            and population["requiredTargetCount"] == 4224
            and population["useFrozenClassBalancedWeights"]
            and population["V172IsDevelopmentForV174"]
            and not population["selectionSubsamplingOrExclusionAllowed"]
        ),
        "certificate_definition_is_exact_complete_and_nonoperational": bool(
            certificate["maximumDepth"] == 5
            and certificate["enumerateEveryQuerySubset"]
            and certificate["targetIdentityMayChooseCertificate"]
            and certificate["targetInformedCertificateIsOnlyAnUpperBoundNotAnOperationalPolicy"]
        ),
        "adaptive_curve_is_cost_free_structural_and_target_blind": bool(
            adaptive["horizons"] == [0, 1, 2, 3, 4, 5]
            and not adaptive["queryCostUsed"]
            and adaptive["dynamicProgramMaximizesExpectedTrustedCompletion"]
            and adaptive["prior"].startswith("unchanged V167")
            and adaptive["hiddenTargetNotAvailableToAdaptivePolicy"]
            and adaptive["tieBreak"] == "lowest valuation index"
            and valid_lock(planner)
        ),
        "feasibility_gates_are_exact_and_exclude_policy_cost_or_sandbox_scoring": bool(
            gates["requiredStateCount"] == 132
            and gates["requiredTargetCount"] == 4224
            and gates["requiredTargetCoverage"] == 1.0
            and gates["requiredCertificateValidity"] == 1.0
            and gates["requiredCertificateMinimality"] == 1.0
            and gates["requiredFullDepthCertifiability"] == 1.0
            and gates["requiredHorizonMonotonicity"] == 1.0
            and gates["maximumPlannerRiskOrCostScoreCount"] == 0
            and gates["maximumSandboxTransactionCount"] == 0
        ),
        "prelock_model_authority_and_effect_boundaries_hold": bool(
            exposure["implementationUnitStateCount"] == 1
            and exposure["formalStateCount"] == 0
            and exposure["aggregateFormalMetricInspectionCount"] == 0
            and all(
                value == 0
                for key, value in exposure.items()
                if key not in {
                    "implementationUnitStateCount",
                    "formalStateCount",
                    "aggregateFormalMetricInspectionCount",
                }
            )
            and not config["decisionRule"]["passAuthorizesImmediatePlannerScoring"]
            and not config["decisionRule"]["passAuthorizesChangeToV173"]
            and not config["decisionRule"]["passAuthorizesModelRegistrationRealServiceOrExecution"]
        ),
        "required_locked_files_exist": all(
            path.is_file()
            for path in (
                config_path,
                parent_path,
                source_path,
                planner_path,
                roadmap_path,
                plan_path,
                protocol_path,
                tests_path,
                runner_path,
                verifier_path,
                auditor_path,
            )
        ),
        "formal_census_absent_before_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "174-certificate-depth-feasibility-census-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_certificate_census" if passed else "reject_V174_design",
        "checks": checks,
        "prelock_exposure": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V173_outcome": parent_path,
        "source_V172_outcome": source_path,
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
        "schema_version": "174-certificate-depth-feasibility-census-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "V167_config_payload": planner["config_payload"],
        "authorization": {
            "modify_population_certificate_definition_horizons_metrics_gates_or_decision": False,
            "run_formal_certificate_census_once": True,
            "score_planner_risk_or_cost_or_run_sandbox": False,
            "select_exclude_tune_or_change_V173": False,
            "run_model_register_mutate_state_call_service_or_execute": False,
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
