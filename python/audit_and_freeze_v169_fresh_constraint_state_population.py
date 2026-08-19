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
    config_path = PROJECT_ROOT / "configs/v169-fresh-constraint-state-population.json"
    parent_path = PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox-outcome-lock.json"
    source_path = PROJECT_ROOT / "configs/v166-model-free-factored-ontology-baselines-outcome-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v168.md"
    plan_path = PROJECT_ROOT / "docs/v169-fresh-constraint-state-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v169_fresh_constraint_state_population.py"
    tests_path = PROJECT_ROOT / "python/test_v169_fresh_constraint_state_population.py"
    runner_path = PROJECT_ROOT / "python/run_v169_fresh_constraint_state_population.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v169_fresh_constraint_state_population_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v169_fresh_constraint_state_population.py"
    audit_path = PROJECT_ROOT / "outputs/v169-fresh-constraint-state-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v169-fresh-constraint-state-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v169-fresh-constraint-state-population/population"
    outcome_path = PROJECT_ROOT / "configs/v169-fresh-constraint-state-population-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V169 is already preregistered, built, or frozen")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    source = json.loads(source_path.read_text())
    generator = config["generator"]
    eligibility = config["plannerEligibility"]
    gates = config["populationGates"]
    exposure = config["preLockExposure"]
    authority = config["authorityBoundary"]
    hidden_path = PROJECT_ROOT / source["hidden_records"]
    checks = {
        "V168_is_frozen_simulated_evidence_with_no_integration_authority": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["authorization"]["retain_V168_as_positive_project_authored_simulated_development_evidence"]
            and not parent["authorization"]["create_or_open_evaluation_population_without_separate_design"]
            and not parent["authorization"]["call_real_service_or_tool"]
        ),
        "V166_hidden_source_is_exact_and_zero_residual": bool(
            valid_lock(source) and source["outcome"]["passed"]
            and source["outcome"]["model_eligible_residual_count"] == 0
            and hidden_path.is_file() and file_sha256(hidden_path) == source["hidden_records_sha256"]
        ),
        "complete_nonoverlapping_generator_is_frozen_without_policy_selection": bool(
            generator["candidateTruthTableCount"] == 256
            and generator["valuationCount"] == 8
            and generator["constraintCountPerState"] == 2
            and generator["completeValuationPairAndOutcomeEnumeration"]
            and generator["requiredSourceStateCount"] == 112
            and generator["excludeEveryV165AmbiguousConstraintSignature"]
            and not generator["selectionUsesPlannerScoresOrPolicies"]
            and generator["retainEveryNonoverlappingSourceState"]
        ),
        "eligibility_is_structural_and_retains_ineligible_states": bool(
            eligibility["usesOnlyFrozenCandidateClassMetadata"]
            and eligibility["minimumEligibleStateCount"] >= 48
            and eligibility["ineligibleStatesRetainedInPopulation"]
            and not eligibility["policyScoringDuringEligibility"]
        ),
        "population_gates_require_exactness_and_zero_policy_access": bool(
            gates["requiredSourceStateCount"] == 112
            and gates["requiredOverlapWithV165AmbiguousSignatures"] == 0
            and gates["requiredCandidatesPerSelectedState"] == 64
            and gates["requiredVersionSpaceExactness"] == 1.0
            and gates["requiredAllNonoverlappingStatesRetained"] == 1.0
            and gates["maximumPlannerPolicyScoreCount"] == 0
        ),
        "prelock_and_authority_boundaries_are_closed": bool(
            exposure["implementationTestPopulationBuildCount"] == 1
            and exposure["formalPopulationBuildCount"] == 0
            and all(value == 0 for key, value in exposure.items() if key not in {"implementationTestPopulationBuildCount", "formalPopulationBuildCount"})
            and authority["populationContainsOnlyShadowVersionSpaces"]
            and authority["authoritativeOntologyAndStateImmutable"]
            and authority["plannerNotRun"] and authority["modelNotRun"]
            and not authority["provisionalRegistrationAllowed"]
            and not authority["actionOrExecutionAllowed"]
            and authority["realExecutionCount"] == 0
            and not config["decisionRule"]["passAuthorizesImmediatePlannerRun"]
            and not config["decisionRule"]["passAuthorizesModelRegistrationAuthorityActionOrExecution"]
        ),
        "required_locked_files_exist": all(path.is_file() for path in (
            config_path, parent_path, source_path, roadmap_path, plan_path, protocol_path,
            tests_path, runner_path, verifier_path, auditor_path,
        )),
        "formal_population_absent_before_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "169-fresh-constraint-state-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_population_build" if passed else "reject_V169_design",
        "checks": checks,
        "prelock_exposure": exposure,
        "planner_policy_score_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_V168_outcome": parent_path, "source_V166_outcome": source_path,
        "V165_hidden_records": hidden_path, "roadmap": roadmap_path, "plan": plan_path,
        "protocol": protocol_path, "tests": tests_path, "runner": runner_path,
        "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "169-fresh-constraint-state-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_generator_exclusions_eligibility_gates_or_decision": False,
            "build_formal_population_once": True,
            "score_planner_or_policy": False,
            "create_external_or_human_evidence_claim": False,
            "run_model_register_mutate_state_act_or_execute": False,
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
