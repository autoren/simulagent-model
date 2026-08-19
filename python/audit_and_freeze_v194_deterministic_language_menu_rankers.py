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
    config_path = PROJECT_ROOT / "configs/v194-deterministic-language-menu-rankers.json"
    plan_path = PROJECT_ROOT / "docs/v194-deterministic-language-menu-rankers-plan.md"
    protocol_path = PROJECT_ROOT / "python/v194_deterministic_language_menu_rankers.py"
    tests_path = PROJECT_ROOT / "python/test_v194_deterministic_language_menu_rankers.py"
    runner_path = PROJECT_ROOT / "python/run_v194_deterministic_language_menu_rankers.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v194_deterministic_language_menu_rankers_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v194_deterministic_language_menu_rankers.py"
    audit_path = PROJECT_ROOT / "outputs/v194-deterministic-language-menu-rankers/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v194-deterministic-language-menu-rankers-lock.json"
    output_root = PROJECT_ROOT / "outputs/v194-deterministic-language-menu-rankers/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v194-deterministic-language-menu-rankers-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V194 is already preregistered, evaluated, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV193OutcomeLock"]
    v192_path = PROJECT_ROOT / config["sourceV192OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v192 = json.loads(v192_path.read_text())
    v192_lock = json.loads((PROJECT_ROOT / v192["extraction_lock"]).read_text())
    inputs = {
        "development_language": PROJECT_ROOT / config["developmentLanguage"],
        "hidden_targets": PROJECT_ROOT / config["hiddenTargets"],
        "visible_menu": PROJECT_ROOT / config["visibleMenu"],
        "hidden_option_map": PROJECT_ROOT / config["hiddenOptionMap"],
        "primary_prior": PROJECT_ROOT / config["primaryPrior"],
        "fixed_hierarchy_target_costs": PROJECT_ROOT / config["fixedHierarchyTargetCosts"],
    }
    pre = config["preLockExposure"]
    decision = config["decisionRule"]
    checks = {
        "parent_V193_and_source_V192_are_valid": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_one_deterministic_language_ranker_evaluation_only"]
            and valid_lock(v192)
            and v192["outcome"]["passed"]
        ),
        "language_targets_menu_prior_and_cost_artifacts_are_exact": bool(
            file_sha256(inputs["development_language"]) == v192["development_language_sha256"]
            and file_sha256(inputs["hidden_targets"]) == v192_lock["hidden_targets_sha256"]
            and file_sha256(inputs["visible_menu"]) == parent["visible_menu_sha256"]
            and file_sha256(inputs["hidden_option_map"]) == parent["hidden_option_map_sha256"]
            and file_sha256(inputs["primary_prior"]) == parent["primary_prior_sha256"]
            and file_sha256(inputs["fixed_hierarchy_target_costs"]) == parent["fixed_hierarchy_target_costs_sha256"]
        ),
        "rankers_and_queries_are_fixed_without_calibration": bool(
            [row["ranker_id"] for row in config["rankers"]]
            == ["CHAR_LAST", "CHAR_ALL", "TOKEN_ALL", "RRF_ALL"]
            and config["observableQuery"]["systemUtterancesExcluded"]
            and config["observableQuery"]["goldTargetsSourceMetadataTruthKindsAndHiddenMapExcludedFromRanking"]
            and config["normalization"]["tieBreak"] == "ascending option_id"
        ),
        "evaluation_costs_signal_gate_and_champion_rule_are_prospective": bool(
            config["evaluation"]["rankedOutputLength"] == 3
            and config["evaluation"]["top1QuestionCost"] == 0.10
            and config["evaluation"]["top3QuestionCost"] == 0.20
            and config["evaluation"]["missThenGenericAdditionalCost"] == 0.40
            and config["evaluationGates"]["minimumBestPrimaryTop3RecallForSignal"] == 0.30
            and config["evaluationGates"]["minimumBestMacroTop3RecallForSignal"] == 0.30
            and config["decisionRule"]["deterministicMaterialValueIsReportedNotRequiredForModelAuthorization"]
        ),
        "prelock_language_model_and_execution_access_is_zero": all(value == 0 for value in pre.values()),
        "successor_authority_and_API_fallback_are_closed": bool(
            not decision["passAuthorizesImmediateModelRun"]
            and not decision["passAuthorizesAPIFallback"]
            and not decision["passAuthorizesProtectedAccessRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    parent_path,
                    v192_path,
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
        "schema_version": "194-deterministic-language-menu-rankers-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V194_deterministic_evaluation" if passed else "reject_V194_design",
        "checks": checks,
        "prelock_exposure": pre,
        "development_language_read_count": 0,
        "deterministic_language_score_count": 0,
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
        "parent_V193_outcome": parent_path,
        "source_V192_outcome": v192_path,
        "source_V192_extraction_lock": PROJECT_ROOT / v192["extraction_lock"],
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
        "schema_version": "194-deterministic-language-menu-rankers-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_queries_rankers_costs_gates_champion_or_decision": False,
            "run_deterministic_development_evaluation_once": True,
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
