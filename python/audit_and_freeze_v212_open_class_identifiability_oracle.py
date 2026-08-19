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
    paths = {
        "config": PROJECT_ROOT / "configs/v212-open-class-identifiability-oracle.json",
        "plan": PROJECT_ROOT / "docs/v212-open-class-identifiability-oracle-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v202.md",
        "research_direction": PROJECT_ROOT / "docs/research-direction.md",
        "open_world_direction": PROJECT_ROOT / "docs/open-world-language-research-direction.md",
        "protocol": PROJECT_ROOT / "python/v212_open_class_identifiability_oracle.py",
        "tests": PROJECT_ROOT / "python/test_v212_open_class_identifiability_oracle.py",
        "worker": PROJECT_ROOT / "python/v212_oracle_worker.py",
        "runner": PROJECT_ROOT / "python/run_v212_open_class_identifiability_oracle.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v212_open_class_identifiability_oracle_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v212_open_class_identifiability_oracle.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v212-representational-diagnosis/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v212-open-class-identifiability-oracle-lock.json"
    output_root = PROJECT_ROOT / "outputs/v212-representational-diagnosis/oracle"
    outcome_path = PROJECT_ROOT / "configs/v212-open-class-identifiability-oracle-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V212 is already audited, frozen, materialized, or outcome-frozen")

    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV211r1OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    domain = config["semanticDomain"]
    languages = config["representationLanguages"]
    population = config["population"]
    gates = config["oracleGates"]
    access = config["accessGates"]
    exposure = config["preLockExposure"]
    recipes = population["recipes"]
    public_fields = set(population["publicFields"])
    hidden_fields = set(population["hiddenFields"])
    worker_source = paths["worker"].read_text()
    checks = {
        "V211r1_is_frozen_zero_model_eligibility_and_authorizes_new_identifiable_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["branch"] == "ZERO_MODEL_ELIGIBILITY"
            and not parent["outcome"]["model_eligible"]
            and parent["authorization"]["design_new_identifiable_open_class_population"]
            and not parent["authorization"]["open_protected_or_run_model"]
        ),
        "complete_finite_behavioral_domain_and_languages_are_frozen": bool(
            domain["worldCount"] == 8
            and domain["completeBooleanBehaviorCount"] == 256
            and set(domain["registeredPrimitiveTruthTables"]) == {"P", "Q", "R"}
            and len(set(domain["registeredPrimitiveTruthTables"].values())) == 3
            and languages["baseLanguage"]["operators"] == ["PRIMITIVE", "IDENTITY", "AND", "OR"]
            and languages["diagnosticExtensionLanguage"]["addsOperators"] == ["NOT", "XOR"]
            and languages["expressibilityPrecedence"]
            == ["EXISTING_PRIMITIVE", "EXISTING_COMPOSITION", "MISSING_OPERATOR", "IRREDUCIBLE_PROVISIONAL"]
            and languages["allSyntacticProgramsCollapseByCompleteBehaviorBeforeScoring"]
            and languages["irreducibleMeansRelativeToFrozenLanguagesNotAbsoluteNovelty"]
        ),
        "fresh_oracle_population_recipes_and_roles_are_complete": bool(
            population["role"] == "development_oracle_only"
            and population["caseCount"] == 40
            and set(recipes) == set(population["conceptFamilyCounts"])
            and all(len(rows) == 4 for rows in recipes.values())
            and sum(len(rows) for rows in recipes.values()) == 40
            and population["naturalLanguageSurfaceRecordCount"] == 0
            and population["externalOntologyPayloadReadCount"] == 0
        ),
        "public_truth_schema_and_reference_ambiguity_contradiction_outside_cases_are_explicit": bool(
            public_fields & hidden_fields == {"case_id"}
            and population["publicHiddenFieldsDisjointExceptIdentifier"]
            and all(name in recipes for name in (
                "REFERENCE_GROUNDED_SYMBOL",
                "GENUINELY_AMBIGUOUS",
                "CONTRADICTORY",
                "OUTSIDE_DESCRIPTION",
                "MISSING_OPERATOR",
                "IRREDUCIBLE_RELATIVE_TO_LANGUAGES",
            ))
            and all("heldOutWorld" in row for row in recipes["GENUINELY_AMBIGUOUS"])
            and all(row["definition"]["kind"] == "OUTSIDE_DESCRIPTION" for row in recipes["OUTSIDE_DESCRIPTION"])
        ),
        "noncompensatory_exactness_witness_invariance_and_regret_gates_are_frozen": bool(
            gates["requiredExactCandidateSetAccuracy"] == 1.0
            and gates["requiredEvidenceStatusAccuracy"] == 1.0
            and gates["requiredExpressibilitySetAccuracy"] == 1.0
            and gates["requiredShadowActionAccuracy"] == 1.0
            and gates["requiredDistinctPairBoundaryWitnessCoverage"] == 1.0
            and gates["requiredReferenceFactNecessityRate"] == 1.0
            and gates["requiredCompleteObservationNecessityRate"] == 1.0
            and all(gates[name] == 1.0 for name in (
                "requiredVocabularyRenamingInvariance",
                "requiredEvidenceOrderInvariance",
                "requiredCommutativeOrderInvariance",
                "requiredEquivalentRewriteInvariance",
            ))
            and gates["maximumExactFalsePrimitiveRate"] == 0.0
            and gates["maximumExactFalseMergeRate"] == 0.0
            and gates["minimumNormalizedForcedNewPrimitiveRegret"] >= 0.50
            and gates["minimumNormalizedForcedNearestMergeRegret"] >= 0.50
        ),
        "truth_firewall_worker_has_only_public_semantics_cases_and_predictions": bool(
            "--semantics" in worker_source
            and "--public-cases" in worker_source
            and "--predictions" in worker_source
            and "--config" not in worker_source
            and "sealed-truth" not in worker_source
            and "expected_candidate" not in worker_source
            and "concept_family" not in worker_source
        ),
        "prelock_exposure_and_access_boundaries_are_zero": bool(
            exposure["exactCandidateSetEvaluationCount"] == 0
            and exposure["exactDecisionScoreCount"] == 0
            and exposure["naturalLanguageSurfaceReadCount"] == 0
            and exposure["externalOntologyPayloadReadCount"] == 0
            and exposure["protectedAccessCount"] == 0
            and exposure["modelLoadCount"] == 0
            and exposure["modelGenerationCount"] == 0
            and exposure["APICallCount"] == 0
            and exposure["trainingRunCount"] == 0
            and exposure["actualExecutionCount"] == 0
            and access["requiredModelFreeOracleRunCount"] == 1
            and all(value == 0 for key, value in access.items() if key != "requiredModelFreeOracleRunCount")
        ),
        "pass_authorizes_only_V213_design": bool(
            config["decisionRule"]["passAuthorizesV213DesignOnly"]
            and not config["decisionRule"]["passAuthorizesImmediatePopulationGenerationExternalPayloadOrModelRun"]
            and not config["decisionRule"]["passAuthorizesAPITrainingRegistrationAuthorityActionOrExecution"]
        ),
        "all_required_files_exist_and_no_formal_output_exists": bool(
            all(path.is_file() for path in (*paths.values(), parent_path))
            and not output_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "212-representational-diagnosis-oracle-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_model_free_V212_oracle" if passed else "reject_V212_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, "parent_V211r1_outcome": parent_path, "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "212-representational-diagnosis-oracle-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_one_model_free_oracle": True,
            "read_natural_language_or_external_ontology_payload": False,
            "read_protected_or_run_model_API_training": False,
            "mutate_register_call_act_or_execute": False,
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
