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
    config_path = PROJECT_ROOT / "configs/v186-typed-contract-question-codebook-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v186-typed-contract-question-codebook-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v186_typed_contract_question_codebook_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v186_typed_contract_question_codebook_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v186_typed_contract_question_codebook_feasibility.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v186_typed_contract_question_codebook_feasibility_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v186_typed_contract_question_codebook_feasibility.py"
    audit_path = PROJECT_ROOT / "outputs/v186-typed-contract-question-codebook-feasibility/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v186-typed-contract-question-codebook-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v186-typed-contract-question-codebook-feasibility/codebook"
    outcome_path = PROJECT_ROOT / "configs/v186-typed-contract-question-codebook-feasibility-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V186 is already preregistered, built, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV185OutcomeLock"]
    source_path = PROJECT_ROOT / config["sourceV183OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    source = json.loads(source_path.read_text())
    paths = {
        "contract_catalog": PROJECT_ROOT / config["contractCatalog"],
        "hidden_identifiability": PROJECT_ROOT / config["hiddenIdentifiability"],
        "development_identities": PROJECT_ROOT / config["developmentIdentities"],
        "protected_identities": PROJECT_ROOT / config["protectedIdentities"],
        "roadmap": PROJECT_ROOT / config["roadmap"],
    }
    families = config["questionFamilies"]
    equivalence = config["equivalenceRule"]
    binding = config["roleBinding"]
    gates = config["feasibilityGates"]
    exposure = config["preLockExposure"]
    decision = config["decisionRule"]
    checks = {
        "V185_is_valid_negative_and_closes_similarity_and_model_residual": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_evaluation_gates_passed"]
            and parent["outcome"]["decision"] == "freeze_V185_boundary_result_without_protected_or_model_access"
            and not parent["authorization"]["preregister_one_local_residual_protocol"]
            and not parent["authorization"]["read_protected_language"]
        ),
        "V183_contract_and_identity_artifacts_are_exact": bool(
            valid_lock(source)
            and source["outcome"]["passed"]
            and file_sha256(paths["contract_catalog"]) == source["contract_catalog_sha256"]
            and file_sha256(paths["hidden_identifiability"]) == source["hidden_identifiability_sha256"]
            and file_sha256(paths["development_identities"]) == source["development_identities_sha256"]
            and file_sha256(paths["protected_identities"]) == source["protected_identities_sha256"]
        ),
        "questions_are_finite_binary_semantic_and_outcome_independent": bool(
            set(families) >= {
                "intent_concept", "domain", "slot_any", "slot_required",
                "slot_result", "transactional", "removeInvariantQuestions",
                "binaryAnswers", "questionOrder",
            }
            and families["removeInvariantQuestions"]
            and families["binaryAnswers"] == [0, 1]
            and families["questionDerivationUsesOnlyFrozenSemanticPayload"]
            and families["questionDerivationMayNotUseServiceVersionSourceDefinitionTruthKindPresentedCandidateLanguageOrOutcomes"]
        ),
        "equivalence_rule_never_forces_hidden_distinctions": bool(
            equivalence["contractsEquivalentOnlyIfCompleteAnswerVectorsAreEqual"]
            and equivalence["neverForceSeparationWithinAnEqualVectorClass"]
            and equivalence["singletonClassesRequiredForFullIdentification"]
            and equivalence["pairwiseSeparationAuditedExhaustively"]
        ),
        "role_binding_is_complete_separate_and_language_free": bool(
            binding["bindEveryV183RecordByOpaqueIdentifier"]
            and binding["observedRecordTargetAnswerVectorFromFrozenContract"]
            and binding["missingRecordHasNoAnswerVectorAndRemainsINSUFFICIENT"]
            and binding["developmentAndProtectedBindingsSeparate"]
            and binding["protectedUtteranceLanguageReadCount"] == 0
            and binding["selectionOrQuestionDesignUsesNoRoleOutcome"]
        ),
        "identification_and_safety_gates_are_noncompensatory": bool(
            gates["requiredCapabilityContractCount"] == 14
            and gates["requiredContractPairCount"] == 91
            and gates["requiredUniqueAnswerVectorCount"] == 14
            and gates["requiredLargestEquivalenceClassSize"] == 1
            and gates["requiredPairwiseSeparationRate"] == 1.0
            and gates["requiredTargetVectorReconstructionRate"] == 1.0
            and gates["maximumPlannerPolicyScoreCount"] == 0
            and gates["maximumUtteranceOrDialogueLanguageReadCount"] == 0
        ),
        "prelock_and_successor_authority_are_closed": bool(
            all(value == 0 for value in exposure.values())
            and not decision["passAuthorizesImmediatePlannerScoring"]
            and not decision["passAuthorizesLanguageModelOrProtectedLanguageAccess"]
            and not decision["passAuthorizesRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, parent_path, source_path, plan_path, protocol_path,
                tests_path, runner_path, verifier_path, auditor_path, *paths.values(),
            ))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "186-typed-contract-question-codebook-feasibility-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_V186_codebook_build" if passed else "reject_V186_design",
        "checks": checks,
        "prelock_exposure": exposure,
        "language_read_count": 0,
        "planner_policy_score_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_V185_outcome": parent_path,
        "source_V183_outcome": source_path,
        **paths,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "186-typed-contract-question-codebook-feasibility-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_contracts_question_families_equivalence_binding_gates_or_decision": False,
            "build_codebook_once": True,
            "score_planner_or_read_utterance_language": False,
            "run_model_API_or_training": False,
            "register_mutate_call_service_act_or_execute": False,
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
