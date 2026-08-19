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
    config_path = PROJECT_ROOT / "configs/v193-shadow-menu-interface-frontier.json"
    plan_path = PROJECT_ROOT / "docs/v193-shadow-menu-interface-frontier-plan.md"
    protocol_path = PROJECT_ROOT / "python/v193_shadow_menu_interface_frontier.py"
    tests_path = PROJECT_ROOT / "python/test_v193_shadow_menu_interface_frontier.py"
    runner_path = PROJECT_ROOT / "python/run_v193_shadow_menu_interface_frontier.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v193_shadow_menu_interface_frontier_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v193_shadow_menu_interface_frontier.py"
    audit_path = PROJECT_ROOT / "outputs/v193-shadow-menu-interface-frontier/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v193-shadow-menu-interface-frontier-lock.json"
    output_root = PROJECT_ROOT / "outputs/v193-shadow-menu-interface-frontier/interface"
    outcome_path = PROJECT_ROOT / "configs/v193-shadow-menu-interface-frontier-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V193 is already preregistered, evaluated, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV192OutcomeLock"]
    v190_path = PROJECT_ROOT / config["sourceV190OutcomeLock"]
    v186_path = PROJECT_ROOT / config["sourceV186OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v190 = json.loads(v190_path.read_text())
    v186 = json.loads(v186_path.read_text())
    v186_lock = json.loads((PROJECT_ROOT / v186["codebook_lock"]).read_text())
    inputs = {
        "contract_catalog": PROJECT_ROOT / config["contractCatalog"],
        "development_bindings": PROJECT_ROOT / config["developmentBindings"],
    }
    grammar = config["proposalGrammar"]
    controller = config["trustedController"]
    economics = config["economics"]
    pre = config["preLockExposure"]
    decision = config["decisionRule"]
    checks = {
        "parent_and_source_outcomes_are_valid": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_shadow_menu_interface_and_oracle_frontier_only"]
            and valid_lock(v190)
            and v190["outcome"]["passed"]
            and valid_lock(v186)
            and v186["outcome"]["passed"]
        ),
        "contract_catalog_and_development_prior_are_exact": bool(
            valid_lock(v186_lock)
            and file_sha256(inputs["contract_catalog"]) == v186_lock["contract_catalog_sha256"]
            and file_sha256(inputs["development_bindings"]) == v186["development_bindings_sha256"]
        ),
        "grammar_is_bounded_exact_and_fail_closed": bool(
            grammar["ranked"]["minimumLength"] == 1
            and grammar["ranked"]["maximumLength"] == 3
            and grammar["ranked"]["distinctKnownOptionIdsOnly"]
            and grammar["missingMalformedTruncatedUnknownDuplicateExtraKeyOrWrongType"] == "INSUFFICIENT"
            and not grammar["confidenceFieldAllowed"]
            and not grammar["freeTextAllowed"]
            and not grammar["retryOrRepairGenerationAllowed"]
        ),
        "controller_is_trusted_answer_only_and_never_prunes": bool(
            controller["policies"] == ["TOP1_PLUS_OTHER", "TOP3_PLUS_OTHER"]
            and controller["bitCost"] == 0.10
            and controller["outsideMenuGenericClarificationCost"] == 0.40
            and controller["proposalNeverDeterminesTerminalState"]
            and controller["proposalNeverPrunesCandidateUniverse"]
            and controller["trustedAnswerRequiredForEveryExactTerminal"]
        ),
        "economics_and_material_thresholds_are_prospective": bool(
            economics["fixedV190MeanCost"] == 0.38
            and economics["alwaysGenericCost"] == 0.40
            and economics["minimumMaterialImprovement"] == 0.02
            and economics["maximumQualifyingMeanCost"] == 0.36
            and economics["top1AnalyticMaterialRecallInclusive"] == 0.35
            and economics["top3AnalyticMaterialRecallInclusive"] == 0.60
        ),
        "prelock_exposure_is_text_free_and_unscored": bool(
            pre["contractMetadataReadCount"] == 1
            and pre["V190CostSummaryReadCount"] == 1
            and all(
                pre[key] == 0
                for key in (
                    "interfaceEvaluationCount",
                    "utteranceOrDialogueLanguageReadCount",
                    "protectedLanguageReadCount",
                    "deterministicLanguageScoreCount",
                    "modelLoadCount",
                    "modelGenerationCount",
                    "APICallCount",
                    "trainingRunCount",
                    "actualExecutionCount",
                )
            )
        ),
        "successor_authority_is_closed": bool(
            not decision["passAuthorizesImmediateLanguageScoring"]
            and not decision["passAuthorizesModelOrAPIRun"]
            and not decision["passAuthorizesProtectedAccessRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    parent_path,
                    v190_path,
                    v186_path,
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
        "schema_version": "193-shadow-menu-interface-frontier-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_text_free_V193_interface_run" if passed else "reject_V193_design",
        "checks": checks,
        "prelock_exposure": pre,
        "utterance_or_dialogue_language_read_count": 0,
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
        "parent_V192_outcome": parent_path,
        "source_V190_outcome": v190_path,
        "source_V186_outcome": v186_path,
        "source_V186_codebook_lock": PROJECT_ROOT / v186["codebook_lock"],
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
        "schema_version": "193-shadow-menu-interface-frontier-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_menu_grammar_controller_costs_gates_or_decision": False,
            "run_text_free_interface_frontier_once": True,
            "read_or_score_utterance_or_protected_language": False,
            "run_model_API_or_training": False,
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
