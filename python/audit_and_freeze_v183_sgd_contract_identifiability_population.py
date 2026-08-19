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
    config_path = PROJECT_ROOT / "configs/v183-sgd-contract-identifiability-population.json"
    plan_path = PROJECT_ROOT / "docs/v183-sgd-contract-identifiability-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v183_sgd_contract_identifiability_population.py"
    tests_path = PROJECT_ROOT / "python/test_v183_sgd_contract_identifiability_population.py"
    runner_path = PROJECT_ROOT / "python/run_v183_sgd_contract_identifiability_population.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v183_sgd_contract_identifiability_population_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v183_sgd_contract_identifiability_population.py"
    audit_path = PROJECT_ROOT / "outputs/v183-sgd-contract-identifiability-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v183-sgd-contract-identifiability-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v183-sgd-contract-identifiability-population/population"
    outcome_path = PROJECT_ROOT / "configs/v183-sgd-contract-identifiability-population-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V183 is already preregistered, built, or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV182OutcomeLock"]
    source_outcome_path = PROJECT_ROOT / config["sourceV134OutcomeLock"]
    archive_path = PROJECT_ROOT / config["sourceArchive"]
    source_catalog_path = PROJECT_ROOT / config["sourceCatalog"]
    source_population_path = PROJECT_ROOT / config["sourcePopulation"]
    roadmap_path = PROJECT_ROOT / config["roadmap"]
    parent = json.loads(parent_path.read_text())
    source = json.loads(source_outcome_path.read_text())
    source_analysis_path = PROJECT_ROOT / source["analysis_lock"]
    source_analysis = json.loads(source_analysis_path.read_text())
    identity = config["capabilityIdentity"]
    protocol = config["hiddenIdentifiabilityProtocol"]
    split = config["roleSplit"]
    gates = config["populationGates"]
    exposure = config["preLockExposure"]
    authority = config["authorityBoundary"]
    decision = config["decisionRule"]
    checks = {
        "V182_is_valid_confirmed_and_closes_the_previous_track": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["strong_confirmation"]
            and parent["authorization"]["fixed_ontology_one_corruption_branch_closed_as_confirmed"]
            and not parent["authorization"]["run_model_register_mutate_real_state_call_sensor_service_or_execute"]
        ),
        "V134_is_valid_frozen_future_asset_and_exact_dependencies_match": bool(
            valid_lock(source)
            and source["outcome"]["passed"]
            and source["authorization"]["retain_as_future_benchmark_asset"]
            and not source["authorization"]["modify_rerun_reselect_or_redefine_V134"]
            and valid_lock(source_analysis)
            and file_sha256(source_analysis_path) == source["analysis_lock_sha256"]
            and file_sha256(archive_path) == source_analysis["source_archive_sha256"]
            and file_sha256(source_catalog_path) == source["choice_catalog_sha256"]
            and file_sha256(source_population_path) == source["fixture_population_sha256"]
        ),
        "full_contract_identity_deduplicates_version_labels_and_forbids_absolute_novelty": bool(
            identity["deduplicateExactCapabilityContractsAcrossSourceDefinitions"]
            and identity["crossTruthKindContractCollisionMeansConfounded"]
            and not identity["absoluteNoveltyClaimAllowed"]
            and "service version labels are excluded" in identity["capabilityContractIdentity"]
        ),
        "hidden_screen_is_conservative_complete_and_nonleaking": bool(
            protocol["sourceSemanticFrameIsHiddenAndNeverModelInput"]
            and protocol["targetContractMustAlwaysBeRetained"]
            and protocol["compatibilityMayNotUsePresentedCandidateOrTruthKind"]
            and protocol["multipleOrMixedCompatibleContracts"] == "INSUFFICIENT"
            and protocol["zeroCompatibleContracts"] == "INVALID_SOURCE_RECORD"
            and protocol["missingObservation"].startswith("INSUFFICIENT")
            and set(protocol["excludedFields"]) >= {
                "service_name", "domain", "source_definition_id", "truth_kind",
                "utterance", "dialogue_text", "slot_values", "character_spans",
            }
            and protocol["screeningDoesNotEstablishUtteranceLevelHumanOrModelIdentifiability"]
        ),
        "role_split_is_prospective_balanced_and_outcome_independent": bool(
            split["recordsPerTruthCandidateCell"] == 4
            and split["developmentRecordsPerCell"] == 2
            and split["protectedRecordsPerCell"] == 2
            and split["requiredCellCount"] == 66
            and split["requiredFixtureCountPerRole"] == 132
            and split["requiredSourceRecordCountPerRole"] == 120
            and split["requiredMissingControlCountPerRole"] == 12
            and split["selectionUsesOnlyFrozenIdentifiersTruthCandidateCellsAndSalt"]
            and split["selectionUsesNoLanguageCompatibilityOutcomeOrModelScore"]
        ),
        "population_gates_require_meaningful_each_class_identifiability": bool(
            gates["minimumIdentifiableSourceRecordCountPerRole"] == 48
            and gates["minimumIdentifiableKnownRecordCountPerRole"] == 24
            and gates["minimumIdentifiableProvisionalRecordCountPerRole"] == 12
            and gates["minimumIdentifiableUnsupportedRecordCountPerRole"] == 6
            and gates["requiredTargetContractRetentionRate"] == 1.0
            and gates["requiredMissingControlInsufficientRate"] == 1.0
            and gates["maximumInvalidSourceRecordCount"] == 0
            and gates["requiredCrossTruthKindContractCollisionCount"] == 0
        ),
        "prelock_exposure_and_authority_are_closed": bool(
            exposure["archiveSchemaShapeInspectionCount"] == 1
            and exposure["dialogueFrameShapeInspectionCount"] == 1
            and exposure["perRecordCompatibilityCensusCount"] == 0
            and exposure["formalPopulationBuildCount"] == 0
            and all(value == 0 for key, value in exposure.items() if key not in {
                "archiveSchemaShapeInspectionCount", "dialogueFrameShapeInspectionCount",
                "perRecordCompatibilityCensusCount", "formalPopulationBuildCount",
            })
            and authority["allCapabilityContractsAreShadowOnly"]
            and authority["authoritativeOntologyAndStateRemainImmutable"]
            and authority["presentedCandidateIsFallibleContextOnly"]
            and authority["compatibilitySetsNeverPruneAuthoritativeHypotheses"]
            and not authority["provisionalRegistrationAllowed"]
            and not authority["languageExtractionAllowedDuringV183"]
            and not authority["modelOrAPIAllowedDuringV183"]
            and not authority["actionOrExecutionAllowed"]
            and authority["realExecutionCount"] == 0
        ),
        "decision_requires_separate_successors_and_no_authority": bool(
            not decision["passAuthorizesImmediateLanguageExtraction"]
            and not decision["passAuthorizesDeterministicPolicyOrModelRun"]
            and not decision["passAuthorizesProtectedAccess"]
            and not decision["passAuthorizesOntologyRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_formal_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, parent_path, source_outcome_path, archive_path,
                source_analysis_path,
                source_catalog_path, source_population_path, roadmap_path, plan_path,
                protocol_path, tests_path, runner_path, verifier_path, auditor_path,
            ))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "183-sgd-contract-identifiability-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_V183_population_build" if passed else "reject_V183_design",
        "checks": checks,
        "prelock_exposure": exposure,
        "language_record_emission_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V182_outcome": parent_path,
        "source_V134_outcome": source_outcome_path,
        "source_V134_analysis_lock": source_analysis_path,
        "source_archive": archive_path,
        "source_catalog": source_catalog_path,
        "source_population": source_population_path,
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
        "schema_version": "183-sgd-contract-identifiability-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_source_membership_contract_identity_role_split_compatibility_gates_or_decision": False,
            "build_formal_population_once": True,
            "extract_or_emit_language": False,
            "run_deterministic_policy_model_API_or_training": False,
            "open_protected_language_register_mutate_state_call_service_act_or_execute": False,
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
