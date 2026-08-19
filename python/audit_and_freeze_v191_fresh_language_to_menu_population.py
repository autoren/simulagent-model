#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v191-fresh-language-to-menu-population.json"
    plan_path = PROJECT_ROOT / "docs/v191-fresh-language-to-menu-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v191_fresh_language_to_menu_population.py"
    tests_path = PROJECT_ROOT / "python/test_v191_fresh_language_to_menu_population.py"
    runner_path = PROJECT_ROOT / "python/run_v191_fresh_language_to_menu_population.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v191_fresh_language_to_menu_population_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v191_fresh_language_to_menu_population.py"
    audit_path = PROJECT_ROOT / "outputs/v191-fresh-language-to-menu-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v191-fresh-language-to-menu-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v191-fresh-language-to-menu-population/population"
    outcome_path = PROJECT_ROOT / "configs/v191-fresh-language-to-menu-population-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V191 is already preregistered, evaluated, or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV190OutcomeLock"]
    source_path = PROJECT_ROOT / config["sourceV124OutcomeLock"]
    v183_path = PROJECT_ROOT / config["sourceV183OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    source = json.loads(source_path.read_text())
    v183 = json.loads(v183_path.read_text())
    inputs = {
        "source_archive": PROJECT_ROOT / config["sourceArchive"],
        "source_inventory": PROJECT_ROOT / config["sourceInventory"],
        "contract_catalog": PROJECT_ROOT / config["contractCatalog"],
        "previous_hidden_population": PROJECT_ROOT / config["previousHiddenPopulation"],
        "roadmap": PROJECT_ROOT / config["roadmap"],
    }
    population = config["population"]
    gates = config["populationGates"]
    pre = config["preLockExposure"]
    boundary = config["authorityBoundary"]
    decision = config["decisionRule"]
    checks = {
        "parent_V190_is_valid_narrow_confirmation": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_confirmation_gates_passed"]
            and parent["authorization"]["claim_finite_oracle_menu_compression_only"]
            and not parent["authorization"]["claim_adaptive_language_model_UI_or_human_result"]
        ),
        "source_V124_and_V183_locks_and_artifacts_are_exact": bool(
            valid_lock(source)
            and source["outcome"]["passed"]
            and valid_lock(v183)
            and v183["outcome"]["passed"]
            and file_sha256(inputs["source_archive"]) == source["source_archive_sha256"]
            and file_sha256(inputs["source_inventory"]) == source["inventory_sha256"]
            and file_sha256(inputs["contract_catalog"]) == v183["contract_catalog_sha256"]
            and file_sha256(inputs["previous_hidden_population"]) == v183["hidden_identifiability_sha256"]
        ),
        "population_and_freshness_gates_are_coherent": bool(
            population["requiredContractCount"] == 14
            and population["sourceRecordsPerContract"] == 6
            and population["requiredSourceRecordCount"] == 84
            and population["requiredMissingControlCount"] == 14
            and population["requiredFixtureCount"] == 98
            and gates["minimumUnusedDialogueCountPerContract"] == 6
            and gates["requiredV183SourceRecordOverlap"] == 0
            and gates["requiredV183DialogueOverlap"] == 0
            and gates["requiredWithinV191DialogueOverlap"] == 0
        ),
        "selection_is_text_free_and_excludes_both_V183_roles": bool(
            config["freshnessContract"]["excludeEveryDialogueAppearingInEitherV183Role"]
            and config["freshnessContract"]["excludeEveryV183SourceCandidate"]
            and config["freshnessContract"]["selectedDialoguesMustBeGloballyUnique"]
            and config["freshnessContract"]["selectionUsesNoUtteranceDialogueTextSlotValueFramePredictionOrOutcome"]
        ),
        "prelock_exposure_is_metadata_only": bool(
            pre["sourceMetadataInventoryReadCount"] == 1
            and pre["priorUsedIdentityReadCount"] == 1
            and pre["aggregatePerContractUnusedDialogueCensusCount"] == 1
            and pre["selectedIdentityBuildCount"] == 0
            and all(
                pre[key] == 0
                for key in (
                    "utteranceOrDialogueTextReadOrEmissionCount",
                    "manualLanguageInspectionCount",
                    "protectedLanguageReadCount",
                    "policyScoreCount",
                    "modelLoadCount",
                    "modelGenerationCount",
                    "APICallCount",
                    "trainingRunCount",
                    "actualExecutionCount",
                )
            )
        ),
        "successor_authority_is_closed": bool(
            boundary["allTargetsAndFutureProposalsAreShadowOnly"]
            and boundary["authoritativeHypothesisUniverseRemainsComplete"]
            and not any(
                boundary[key]
                for key in (
                    "languageExtractionAllowedDuringV191",
                    "protectedLanguageAccessAllowed",
                    "modelAPIOrTrainingAllowed",
                    "ontologyRegistrationOrPruningAllowed",
                    "trustedMutationActionOrExecutionAllowed",
                )
            )
            and not decision["passAuthorizesImmediateLanguageExtraction"]
            and not decision["passAuthorizesInterfaceScoringOrModelRun"]
            and not decision["passAuthorizesProtectedAccessRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    parent_path,
                    source_path,
                    v183_path,
                    plan_path,
                    protocol_path,
                    tests_path,
                    runner_path,
                    verifier_path,
                    auditor_path,
                    *inputs.values(),
                )
            )
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "191-fresh-language-to-menu-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_text_free_V191_population_build" if passed else "reject_V191_design",
        "checks": checks,
        "prelock_exposure": pre,
        "selected_identity_build_count": 0,
        "utterance_or_dialogue_text_read_or_emission_count": 0,
        "protected_language_read_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V190_outcome": parent_path,
        "source_V124_outcome": source_path,
        "source_V183_outcome": v183_path,
        **inputs,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "191-fresh-language-to-menu-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_selection_exclusions_population_gates_or_decision": False,
            "run_text_free_population_build_once": True,
            "read_or_emit_utterance_language": False,
            "read_protected_language_or_run_model_API_training": False,
            "register_prune_mutate_call_service_act_or_execute": False,
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
