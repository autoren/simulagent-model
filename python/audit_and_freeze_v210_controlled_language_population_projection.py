#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v210-controlled-language-population-projection.json"
    plan_path = PROJECT_ROOT / "docs/v210-controlled-language-population-projection-plan.md"
    protocol_path = PROJECT_ROOT / "python/v210_controlled_language_population_projection.py"
    tests_path = PROJECT_ROOT / "python/test_v210_controlled_language_population_projection.py"
    runner_path = PROJECT_ROOT / "python/run_v210_controlled_language_population_projection.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v210_controlled_language_population_projection_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v210_controlled_language_population_projection.py"
    audit_path = PROJECT_ROOT / "outputs/v210-controlled-language-population-projection/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v210-controlled-language-population-projection-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v210-controlled-language-population-projection-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, outcome_path)):
        raise RuntimeError("V210 is already preregistered or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV209r1OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    population = config["population"]
    grammar = config["surfaceGrammar"]
    projection = config["projection"]
    pop_gates = config["populationGates"]
    proj_gates = config["projectionGates"]
    access = config["accessGates"]
    prelock = config["preLockExposure"]
    expected_groups = (
        len(population["semanticRegimes"])
        * len(population["taskStates"])
        * len(population["contexts"])
        * len(population["semanticObservationIds"])
    )
    expected_records = expected_groups * len(population["counterfactualTypes"])
    artifacts_absent = all(not (PROJECT_ROOT / value).exists() for value in config["artifacts"].values())
    checks = {
        "V209r1_is_valid_positive_and_authorizes_only_population_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_oracle_passed"]
            and parent["authorization"]["preregister_fresh_controlled_language_population_design_only"]
            and not parent["authorization"]["open_language_population_or_run_model"]
        ),
        "factorial_population_counts_are_fixed_before_generation": bool(
            config["schemaVersion"] == "210-controlled-language-population-projection-design"
            and expected_groups == population["groupsPerRole"] == pop_gates["requiredGroupsPerRole"] == 90
            and expected_records == population["recordsPerRole"] == pop_gates["requiredRecordsPerRole"] == 270
            and population["totalRecords"] == 2 * expected_records == 540
            and population["recordsPerCounterfactualGroup"] == pop_gates["requiredRecordsPerGroup"] == 3
        ),
        "seeds_identifiers_roles_and_truth_firewall_are_fixed": bool(
            population["generatorVersion"] == "v210-generator-1"
            and set(population["roleSeeds"]) == set(population["roles"]) == {"DEVELOPMENT", "PROTECTED"}
            and len(set(population["roleSeeds"].values())) == 2
            and set(population["recordIdHashes"]) == {"DEVELOPMENT", "PROTECTED", "ALL"}
            and all(len(value) == 64 for value in population["recordIdHashes"].values())
            and population["surfaceAndTruthArtifactsAreSeparate"]
            and population["fullFactorKeysMustBeUnique"]
            and population["protectedSurfaceTextSealedFromBaselineAndManualInspection"]
        ),
        "heldout_constructions_counterfactuals_and_lexicons_are_fixed": bool(
            set(population["counterfactualTypes"]) == {"DIRECT", "MATCHED_PARAPHRASE", "OPAQUE_RENAMING"}
            and set(grammar["constructionFamiliesByRole"]) == {"DEVELOPMENT", "PROTECTED"}
            and set(grammar["templates"]) == {"DEVELOPMENT", "PROTECTED"}
            and set(grammar["lexicons"]) == {"DEVELOPMENT", "PROTECTED"}
            and grammar["constructionFamiliesByRole"]["DEVELOPMENT"] != grammar["constructionFamiliesByRole"]["PROTECTED"]
            and grammar["roleTemplateSkeletonsAndLexicalLabelsMustBeDisjoint"]
            and grammar["surfaceContainsNoRegimeStateOrProbabilityField"]
        ),
        "projection_is_truth_blind_conservative_and_development_only": bool(
            projection["name"] == "EXPLICIT_DEVELOPMENT_MARKER_ONLY"
            and projection["inputFields"] == ["record_id", "utterance"]
            and projection["acceptedCounterfactualType"] == "DIRECT"
            and projection["paraphraseAndOpaqueDefault"] == "ABSTAIN"
            and not projection["readsTruthDuringPrediction"]
            and not projection["readsProtectedSurfaceDuringPrediction"]
            and not projection["readsProtectedTruthDuringPrediction"]
            and projection["residualDefinedFromPredictionOnly"]
            and proj_gates["requiredAcceptedCount"] + proj_gates["requiredResidualCount"]
            == proj_gates["requiredDevelopmentPredictionCount"]
        ),
        "population_projection_and_access_gates_are_exact_and_noncompensatory": bool(
            pop_gates["requiredUniqueRecordIdRate"] == 1.0
            and pop_gates["requiredUniqueFullFactorKeyRate"] == 1.0
            and pop_gates["requiredCounterfactualTruthMismatchCount"] == 0
            and pop_gates["requiredProbabilityNormalizationRate"] == 1.0
            and proj_gates["requiredAcceptedAccuracy"] == 1.0
            and proj_gates["requiredFalseAcceptanceCount"] == 0
            and access["requiredPopulationGenerationCount"] == 1
            and access["requiredDevelopmentProjectionEvaluationCount"] == 1
            and all(value == 0 for key, value in access.items() if not key.startswith("required"))
        ),
        "prelock_enumerated_only_identifiers_and_opened_no_records_or_evaluation": bool(
            prelock["identifierEnumerationCount"] == 540
            and prelock["identifierHashComputationCount"] == 3
            and all(
                prelock[key] == 0
                for key in (
                    "surfaceRecordGenerationCount",
                    "surfaceRecordReadCount",
                    "truthRecordGenerationCount",
                    "truthRecordReadCount",
                    "projectionEvaluationCount",
                    "modelLoadOrGenerationCount",
                    "APICallCount",
                    "trainingRunCount",
                    "actualExecutionCount",
                )
            )
        ),
        "pass_authorizes_only_separate_deterministic_baseline_design": bool(
            config["decisionRule"]["passAuthorizesSeparateDeterministicDevelopmentBaselineDesignOnly"]
            and not config["decisionRule"]["passAuthorizesModelRunOrProtectedOpening"]
            and not config["decisionRule"]["passAuthorizesAPITrainingRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_all_population_evaluation_artifacts_are_absent": bool(
            all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path, parent_path))
            and artifacts_absent
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "210-controlled-language-population-projection-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V210_population_generation_and_projection" if passed else "reject_V210_design",
        "checks": checks,
        "prelock_exposure": prelock,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V209r1_outcome": parent_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "210-controlled-language-population-projection-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_generator_seed_population_template_lexicon_projection_or_gates": False,
            "generate_population_and_run_development_projection_once": True,
            "manually_read_protected_surface_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
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
