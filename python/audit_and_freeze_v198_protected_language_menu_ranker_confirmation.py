#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from audit_and_freeze_v195_bounded_local_language_menu_ranker import manifest_snapshot_is_exact
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v198-protected-language-menu-ranker-confirmation.json"
    plan_path = PROJECT_ROOT / "docs/v198-protected-language-menu-ranker-confirmation-plan.md"
    protocol_path = PROJECT_ROOT / "python/v198_protected_language_menu_ranker_confirmation.py"
    tests_path = PROJECT_ROOT / "python/test_v198_protected_language_menu_ranker_confirmation.py"
    runner_path = PROJECT_ROOT / "python/run_v198_protected_language_menu_ranker_confirmation.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v198_protected_language_menu_ranker_confirmation_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v198_protected_language_menu_ranker_confirmation.py"
    v195_protocol_path = PROJECT_ROOT / "python/v195_bounded_local_language_menu_ranker.py"
    v194_protocol_path = PROJECT_ROOT / "python/v194_deterministic_language_menu_rankers.py"
    bounded_helper_path = PROJECT_ROOT / "python/v154_adaptive_local_question_order.py"
    audit_path = PROJECT_ROOT / "outputs/v198-protected-language-menu-ranker-confirmation/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v198-protected-language-menu-ranker-confirmation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v198-protected-language-menu-ranker-confirmation/model-realization"
    outcome_path = PROJECT_ROOT / "configs/v198-protected-language-menu-ranker-confirmation-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V198 is already preregistered, run, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV197OutcomeLock"]
    v195_path = PROJECT_ROOT / config["sourceV195OutcomeLock"]
    v194_path = PROJECT_ROOT / config["sourceV194OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v195 = json.loads(v195_path.read_text())
    v194 = json.loads(v194_path.read_text())
    v195_lock_path = PROJECT_ROOT / v195["experiment_lock"]
    v195_lock = json.loads(v195_lock_path.read_text())
    v194_lock_path = PROJECT_ROOT / v194["evaluation_lock"]
    v194_lock = json.loads(v194_lock_path.read_text())
    inputs = {
        "confirmation_language": PROJECT_ROOT / config["confirmationLanguage"],
        "hidden_targets": PROJECT_ROOT / config["hiddenTargets"],
        "visible_menu": PROJECT_ROOT / config["visibleMenu"],
        "hidden_option_map": PROJECT_ROOT / config["hiddenOptionMap"],
        "primary_prior": PROJECT_ROOT / config["primaryPrior"],
        "fixed_hierarchy_target_costs": PROJECT_ROOT / config["fixedHierarchyTargetCosts"],
        "model_manifest": PROJECT_ROOT / config["modelManifest"],
        "roadmap": PROJECT_ROOT / config["roadmap"],
    }
    old = v195_lock["config_payload"]
    old_char = next(row for row in v194_lock["config_payload"]["rankers"] if row["ranker_id"] == "CHAR_LAST")
    inherited_numeric_keys = (
        "top1QuestionCost", "top3QuestionCost", "missThenGenericAdditionalCost", "missingObservationCost",
        "minimumIncrementalPrimaryImprovement", "maximumQualifyingPrimaryTop3MeanCost",
        "maximumQualifyingMacroTop3MeanCost",
    )
    manifest = json.loads(inputs["model_manifest"].read_text())
    checks = {
        "V197_authorizes_only_a_separate_unchanged_policy_confirmation": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_unchanged_V195_policy_confirmation_only"]
            and not parent["authorization"]["run_model_without_separate_lock"]
        ),
        "V195_and_V194_sources_are_valid": bool(
            valid_lock(v195) and v195["outcome"]["passed"] and v195["outcome"]["scientific_qualification_gates_passed"]
            and valid_lock(v194) and v194["outcome"]["passed"]
            and valid_lock(v195_lock) and valid_lock(v194_lock)
        ),
        "confirmation_language_targets_menu_prior_costs_and_model_are_exact": bool(
            file_sha256(inputs["confirmation_language"]) == parent["confirmation_language_sha256"]
            and file_sha256(inputs["hidden_targets"]) == parent["hidden_targets_sha256"]
            and all(
                file_sha256(inputs[key]) == v195_lock[f"{key}_sha256"]
                for key in ("visible_menu", "hidden_option_map", "primary_prior", "fixed_hierarchy_target_costs", "model_manifest")
            )
            and manifest_snapshot_is_exact(manifest, config)
        ),
        "model_prompt_and_qualification_gates_are_unchanged_from_V195": bool(
            config["model"] == old["model"]
            and config["prompt"] == old["prompt"]
            and config["qualificationGates"] == old["qualificationGates"]
            and all(config["trustedEvaluation"][key] == old["trustedEvaluation"][key] for key in inherited_numeric_keys)
        ),
        "CHAR_LAST_comparator_is_unchanged_from_V194": bool(
            {key: config["deterministicComparator"][key] for key in old_char} == old_char
            and config["deterministicComparator"]["rankedOutputLength"] == v194_lock["config_payload"]["evaluation"]["rankedOutputLength"]
        ),
        "development_reference_values_are_exact": bool(
            config["trustedEvaluation"]["developmentModelPrimaryTop3MeanCost"] == v195["outcome"]["summary"]["primary_top3_mean_cost"]
            and config["trustedEvaluation"]["developmentModelMacroTop3MeanCost"] == v195["outcome"]["summary"]["macro_top3_mean_cost"]
            and config["trustedEvaluation"]["developmentDeterministicChampionPrimaryTop3MeanCost"] == v194["outcome"]["summary"]["champion_primary_top3_mean_cost"]
        ),
        "prelock_language_model_and_execution_access_is_zero": all(value == 0 for value in config["preLockExposure"].values()),
        "API_model_selection_authority_action_and_execution_are_closed": bool(
            not config["decisionRule"]["passAuthorizesOntologyPromotionAuthorityActionOrExecution"]
            and not config["decisionRule"]["passAuthorizesAPIOrAdditionalModel"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path,
                v195_protocol_path, v194_protocol_path, bounded_helper_path, parent_path, v195_path, v194_path,
                v195_lock_path, v194_lock_path, *inputs.values(),
            )) and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "198-protected-language-menu-ranker-confirmation-design-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_and_authorize_one_V198_confirmation_realization" if passed else "reject_V198_design",
        "checks": checks, "prelock_exposure": config["preLockExposure"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_V197_outcome": parent_path, "source_V195_outcome": v195_path,
        "source_V194_outcome": v194_path, "source_V195_lock": v195_lock_path, "source_V194_lock": v194_lock_path,
        **inputs, "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "V195_protocol": v195_protocol_path, "V194_protocol": v194_protocol_path,
        "bounded_reasoning_helper": bounded_helper_path, "runner": runner_path, "verifier": verifier_path,
        "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "198-protected-language-menu-ranker-confirmation-lock",
        "experiment": config["experiment"], "config_payload": config,
        "authorization": {
            "modify_population_model_prompt_budget_parser_comparator_costs_or_gates": False,
            "run_exact_single_confirmation_realization": True,
            "generate_on_missing_fixtures_or_retry": False,
            "persist_or_manually_inspect_raw_model_outputs": False,
            "run_API_training_additional_model_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
