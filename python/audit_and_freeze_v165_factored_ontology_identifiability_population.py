#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def valid_lock(value: dict[str, Any]) -> bool:
    return payload_hash(
        {key: item for key, item in value.items() if key != "lock_payload_sha256"}
    ) == value.get("lock_payload_sha256")


def main() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs/v165-factored-ontology-identifiability-population.json"
    )
    parent_path = (
        PROJECT_ROOT / "configs/v164-local-residual-open-set-transfer-outcome-lock.json"
    )
    roadmap_path = PROJECT_ROOT / "docs/research-branches-after-v161.md"
    plan_path = (
        PROJECT_ROOT
        / "docs/v165-factored-ontology-identifiability-population-plan.md"
    )
    protocol_path = (
        PROJECT_ROOT / "python/v165_factored_ontology_identifiability_population.py"
    )
    tests_path = (
        PROJECT_ROOT
        / "python/test_v165_factored_ontology_identifiability_population.py"
    )
    runner_path = (
        PROJECT_ROOT
        / "python/run_v165_factored_ontology_identifiability_population.py"
    )
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v165_factored_ontology_identifiability_outcome.py"
    )
    auditor_path = (
        PROJECT_ROOT
        / "python/audit_and_freeze_v165_factored_ontology_identifiability_population.py"
    )
    audit_path = (
        PROJECT_ROOT
        / "outputs/v165-factored-ontology-identifiability/implementation-audit.json"
    )
    lock_path = (
        PROJECT_ROOT
        / "configs/v165-factored-ontology-identifiability-population-lock.json"
    )
    output_root = (
        PROJECT_ROOT / "outputs/v165-factored-ontology-identifiability/population"
    )
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V165 design is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    dsl = config["typedDSL"]
    factorial = config["factorialDesign"]
    gates = config["populationGates"]
    authority = config["authorityBoundary"]
    exposure = config["preLockExposure"]
    checks = {
        "V164_is_exactly_frozen_negative_and_track_A_protocol_is_closed": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["quality_gate_pass"]
            and parent["authorization"][
                "close_residual_model_protocol_without_protected_test"
            ]
            and not parent["authorization"][
                "read_protected_transfer_before_separate_lock"
            ]
            and not parent["authorization"]["run_additional_local_or_API_models"]
            and not parent["authorization"]["induce_register_or_execute_capability"]
        ),
        "finite_typed_DSL_and_relative_provisional_semantics_are_exact": bool(
            dsl["entityType"] == "Device"
            and dsl["primitiveArity"] == 1
            and dsl["primitiveValueType"] == "Boolean"
            and dsl["primitiveCount"] == 3
            and dsl["valuationCount"] == 8
            and dsl["candidateTruthTableCount"] == 256
            and dsl["registeredAliasTruthTableCount"] == 3
            and dsl["registeredCompositionTruthTableCount"] == 6
            and dsl["provisionalTruthTableCount"] == 247
            and "not extensionally expressible" in dsl["provisionalMeaning"]
        ),
        "three_by_three_development_only_factorial_is_frozen": bool(
            len(factorial["expressibilityClasses"]) == 3
            and len(factorial["evidenceStatuses"]) == 3
            and factorial["logicalTargetGroupsPerCell"] == 4
            and factorial["renamingsPerLogicalTarget"] == 4
            and factorial["recordsPerCell"] == 16
            and factorial["cellCount"] == 9
            and factorial["recordCount"] == 144
            and factorial["split"] == "development_only"
            and not factorial["evaluationPopulationCreated"]
        ),
        "identifiability_and_access_gates_are_noncompensatory": bool(
            gates["requiredSufficientVersionSpaceSize"] == 1
            and gates["minimumAmbiguousVersionSpaceSize"] == 2
            and gates["requiredAmbiguousExpressibilityClassCoverage"] == 3
            and gates["requiredContradictoryVersionSpaceSize"] == 0
            and gates["requiredTargetRetentionWhenNonContradictory"] == 1.0
            and gates["requiredEvidenceStatusClassificationAccuracy"] == 1.0
            and gates["requiredRenamingVersionSpaceInvariance"] == 1.0
            and gates["maximumEvaluationRecordCount"] == 0
            and all(
                gates[key] == 0
                for key in (
                    "maximumModelLoadCount",
                    "maximumModelGenerationCount",
                    "maximumAPICallCount",
                    "maximumTrainingRunCount",
                    "maximumOntologyRegistrationCount",
                    "maximumActualExecutionCount",
                )
            )
        ),
        "public_hidden_projection_and_development_disclosure_are_frozen": bool(
            not (set(config["publicFields"]) & set(config["hiddenFields"]))
            and "record_id" in config["publicFields"]
            and "version_space_candidate_ids" in config["hiddenFields"]
            and gates["requiredProjectAuthoredDevelopmentDisclosure"]
        ),
        "authority_and_prelock_access_boundaries_are_closed": bool(
            authority["authoritativeOntologyImmutable"]
            and authority["allCandidateTruthTablesRemainShadowOnly"]
            and authority["observationallyEquivalentCandidatesRetained"]
            and not authority["provisionalPrimitiveCanBeRegistered"]
            and not authority["candidateCanUpdateAuthoritativeStateOrBelief"]
            and not authority["candidateCanSelectActionOrExecute"]
            and authority["realExecutionCount"] == 0
            and all(value == 0 for value in exposure.values())
            and not config["decisionRule"]["passAuthorizesImmediateBaselineScoring"]
            and not config["decisionRule"][
                "passAuthorizesModelEvaluationOrProtectedPopulation"
            ]
            and not config["decisionRule"][
                "passAuthorizesOntologyRegistrationAuthorityActionOrExecution"
            ]
        ),
        "required_locked_files_exist": all(
            path.is_file()
            for path in (
                config_path,
                parent_path,
                roadmap_path,
                plan_path,
                protocol_path,
                tests_path,
                runner_path,
                verifier_path,
                auditor_path,
            )
        ),
        "population_output_absent_before_design_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "165-factored-ontology-identifiability-design-audit",
        "experiment": "v165_factored_ontology_identifiability_design_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_one_model_free_population_build"
            if passed
            else "reject_V165_design"
        ),
        "checks": checks,
        "prelock_exposure": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_track_A_outcome": parent_path,
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
        "schema_version": "165-factored-ontology-identifiability-population-lock",
        "experiment": "v165_factored_ontology_identifiability_population_lock",
        "config_payload": config,
        "authorization": {
            "modify_DSL_factors_population_evidence_parser_version_space_gates_or_decision": False,
            "build_and_audit_population_once": True,
            "create_or_open_evaluation_population": False,
            "make_manual_judgments": False,
            "load_or_run_local_or_API_model": False,
            "train_or_fit_learned_component": False,
            "register_provisional_primitive": False,
            "grant_candidate_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
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
