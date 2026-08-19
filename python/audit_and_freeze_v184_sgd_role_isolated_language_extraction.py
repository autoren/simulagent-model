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
    config_path = PROJECT_ROOT / "configs/v184-sgd-role-isolated-language-extraction.json"
    plan_path = PROJECT_ROOT / "docs/v184-sgd-role-isolated-language-extraction-plan.md"
    protocol_path = PROJECT_ROOT / "python/v184_sgd_role_isolated_language_extraction.py"
    tests_path = PROJECT_ROOT / "python/test_v184_sgd_role_isolated_language_extraction.py"
    runner_path = PROJECT_ROOT / "python/run_v184_sgd_role_isolated_language_extraction.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v184_sgd_role_isolated_language_extraction_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v184_sgd_role_isolated_language_extraction.py"
    audit_path = PROJECT_ROOT / "outputs/v184-sgd-role-isolated-language-extraction/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v184-sgd-role-isolated-language-extraction-lock.json"
    output_root = PROJECT_ROOT / "outputs/v184-sgd-role-isolated-language-extraction/extraction"
    outcome_path = PROJECT_ROOT / "configs/v184-sgd-role-isolated-language-extraction-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V184 is already preregistered, extracted, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV183OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    paths = {
        "source_archive": PROJECT_ROOT / config["sourceArchive"],
        "source_V134_catalog": PROJECT_ROOT / config["sourceV134Catalog"],
        "V183_contract_catalog": PROJECT_ROOT / config["V183ContractCatalog"],
        "V183_hidden_identifiability": PROJECT_ROOT / config["V183HiddenIdentifiability"],
        "V183_development_identities": PROJECT_ROOT / config["V183DevelopmentIdentities"],
        "V183_protected_identities": PROJECT_ROOT / config["V183ProtectedIdentities"],
    }
    observable = config["observableRecordContract"]
    catalog = config["declaredCatalogContract"]
    gates = config["extractionGates"]
    exposure = config["preLockExposure"]
    decision = config["decisionRule"]
    checks = {
        "V183_is_valid_positive_and_authorizes_only_separate_extraction": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_population_gates_passed"]
            and parent["authorization"]["preregister_role_isolated_language_extraction"]
            and not parent["authorization"]["extract_language_without_separate_lock"]
            and not parent["authorization"]["open_protected_language_during_development"]
        ),
        "all_source_artifacts_match_V183_outcome": bool(
            file_sha256(paths["V183_contract_catalog"]) == parent["contract_catalog_sha256"]
            and file_sha256(paths["V183_hidden_identifiability"]) == parent["hidden_identifiability_sha256"]
            and file_sha256(paths["V183_development_identities"]) == parent["development_identities_sha256"]
            and file_sha256(paths["V183_protected_identities"]) == parent["protected_identities_sha256"]
            and all(path.is_file() for path in paths.values())
        ),
        "observable_projection_excludes_every_gold_or_authority_field": bool(
            observable["conversationIncludesEveryTurnThroughSelectedUserTurn"]
            and observable["missingObservationConversation"] is None
            and set(observable["recordFields"]) == {
                "record_id", "role", "observation_available",
                "presented_candidate_choice_id", "conversation",
            }
            and set(observable["conversationTurnFields"]) == {"speaker", "utterance"}
            and set(observable["forbiddenRecordFields"]) >= {
                "source_candidate_id", "source_definition_id", "service", "active_intent",
                "domain", "truth_kind", "truth_contract_id", "compatible_contract_ids",
                "identifiability_status", "evaluation_choice", "frames", "state", "slot_values",
            }
        ),
        "declared_catalog_is_complete_known_only_and_non_authoritative": bool(
            catalog["includeOnlySixFrozenKnownChoices"]
            and not catalog["includeProvisionalOrUnsupportedSchemaLanguage"]
            and catalog["catalogIsDescriptiveAndNeverAuthorityGranting"]
            and gates["requiredDeclaredKnownChoiceCount"] == 6
        ),
        "role_isolation_and_exact_extraction_gates_are_noncompensatory": bool(
            gates["requiredDevelopmentFixtureCount"] == 132
            and gates["requiredProtectedFixtureCount"] == 132
            and gates["requiredDevelopmentSourceRecordCount"] == 120
            and gates["requiredProtectedSourceRecordCount"] == 120
            and gates["requiredDevelopmentMissingControlCount"] == 12
            and gates["requiredProtectedMissingControlCount"] == 12
            and gates["requiredRecordIdentifierReconstructionRate"] == 1.0
            and gates["requiredConversationPrefixExactness"] == 1.0
            and gates["requiredPublicProjectionExactness"] == 1.0
            and gates["maximumForbiddenFieldOccurrenceCount"] == 0
        ),
        "prelock_and_successor_access_are_closed": bool(
            all(value == 0 for value in exposure.values())
            and not decision["passAuthorizesImmediateDevelopmentLanguageScoring"]
            and not decision["passAuthorizesProtectedLanguageReading"]
            and not decision["passAuthorizesModelAPITrainingOntologyRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, parent_path, plan_path, protocol_path, tests_path,
                runner_path, verifier_path, auditor_path, *paths.values(),
            ))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "184-sgd-role-isolated-language-extraction-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_exact_V184_extraction" if passed else "reject_V184_design",
        "checks": checks,
        "prelock_exposure": exposure,
        "selected_conversation_read_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_V183_outcome": parent_path,
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
        "schema_version": "184-sgd-role-isolated-language-extraction-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_records_projection_catalog_roles_gates_or_decision": False,
            "extract_once_automatically": True,
            "print_or_manually_inspect_development_or_protected_language": False,
            "score_policy_run_model_API_or_training": False,
            "open_protected_during_development_register_mutate_call_service_act_or_execute": False,
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
