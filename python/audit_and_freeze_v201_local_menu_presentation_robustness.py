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
    config_path = PROJECT_ROOT / "configs/v201-local-menu-presentation-robustness.json"
    plan_path = PROJECT_ROOT / "docs/v201-local-menu-presentation-robustness-plan.md"
    protocol_path = PROJECT_ROOT / "python/v201_local_menu_presentation_robustness.py"
    tests_path = PROJECT_ROOT / "python/test_v201_local_menu_presentation_robustness.py"
    runner_path = PROJECT_ROOT / "python/run_v201_local_menu_presentation_robustness.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v201_local_menu_presentation_robustness_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v201_local_menu_presentation_robustness.py"
    v195_protocol_path = PROJECT_ROOT / "python/v195_bounded_local_language_menu_ranker.py"
    bounded_helper_path = PROJECT_ROOT / "python/v154_adaptive_local_question_order.py"
    audit_path = PROJECT_ROOT / "outputs/v201-local-menu-presentation-robustness/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v201-local-menu-presentation-robustness-lock.json"
    output_root = PROJECT_ROOT / "outputs/v201-local-menu-presentation-robustness/model-realization"
    outcome_path = PROJECT_ROOT / "configs/v201-local-menu-presentation-robustness-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V201 is already preregistered, run, or frozen")
    config = json.loads(config_path.read_text())
    v200_path = PROJECT_ROOT / config["parentV200OutcomeLock"]
    v199_path = PROJECT_ROOT / config["sourceV199OutcomeLock"]
    v195_path = PROJECT_ROOT / config["sourceV195OutcomeLock"]
    v200 = json.loads(v200_path.read_text()); v199 = json.loads(v199_path.read_text()); v195 = json.loads(v195_path.read_text())
    v199_lock_path = PROJECT_ROOT / v199["experiment_lock"]; v199_lock = json.loads(v199_lock_path.read_text())
    v195_lock_path = PROJECT_ROOT / v195["experiment_lock"]; v195_lock = json.loads(v195_lock_path.read_text())
    inputs = {
        "development_language": PROJECT_ROOT / config["developmentLanguage"],
        "hidden_targets": PROJECT_ROOT / config["hiddenTargets"],
        "visible_menu_variants": PROJECT_ROOT / config["visibleMenuVariants"],
        "hidden_variant_maps": PROJECT_ROOT / config["hiddenVariantMaps"],
        "canonical_hidden_option_map": PROJECT_ROOT / config["canonicalHiddenOptionMap"],
        "canonical_model_census": PROJECT_ROOT / config["canonicalModelCensus"],
        "transformed_CHAR_LAST_summary": PROJECT_ROOT / config["transformedCHARLASTSummary"],
        "primary_prior": PROJECT_ROOT / config["primaryPrior"],
        "fixed_hierarchy_target_costs": PROJECT_ROOT / config["fixedHierarchyTargetCosts"],
        "model_manifest": PROJECT_ROOT / config["modelManifest"],
    }
    old = v195_lock["config_payload"]
    manifest = json.loads(inputs["model_manifest"].read_text())
    checks = {
        "V200_authorizes_only_a_separate_unchanged_local_development_test": bool(
            valid_lock(v200) and v200["outcome"]["passed"]
            and v200["authorization"]["preregister_separate_unchanged_local_model_development_robustness_only"]
            and not v200["authorization"]["immediate_model_run_or_protected_access"]
        ),
        "V199_and_V195_sources_are_valid": bool(
            valid_lock(v199) and v199["outcome"]["passed"] and valid_lock(v199_lock)
            and valid_lock(v195) and v195["outcome"]["passed"] and valid_lock(v195_lock)
        ),
        "population_transformations_canonical_outputs_and_controls_are_exact": bool(
            file_sha256(inputs["development_language"]) == v195_lock["development_language_sha256"]
            and file_sha256(inputs["hidden_targets"]) == v195_lock["hidden_targets_sha256"]
            and file_sha256(inputs["visible_menu_variants"]) == v199["visible_menu_variants_sha256"]
            and file_sha256(inputs["hidden_variant_maps"]) == v199["hidden_variant_maps_sha256"]
            and file_sha256(inputs["canonical_hidden_option_map"]) == v195_lock["hidden_option_map_sha256"]
            and file_sha256(inputs["canonical_model_census"]) == v195["census_result_sha256"]
            and file_sha256(inputs["transformed_CHAR_LAST_summary"]) == v200["summary_sha256"]
            and file_sha256(inputs["primary_prior"]) == v195_lock["primary_prior_sha256"]
            and file_sha256(inputs["fixed_hierarchy_target_costs"]) == v195_lock["fixed_hierarchy_target_costs_sha256"]
        ),
        "model_snapshot_prompt_decode_and_parser_policy_are_unchanged": bool(
            config["model"] == old["model"] and config["prompt"] == old["prompt"]
            and manifest_snapshot_is_exact(manifest, config)
        ),
        "prospective_V199_qualification_gates_are_inherited_verbatim": config["qualificationGates"] == v199_lock["config_payload"]["futurePairedDevelopmentGates"],
        "trusted_controller_costs_are_unchanged": bool(
            config["trustedEvaluation"]["top1QuestionCost"] == old["trustedEvaluation"]["top1QuestionCost"]
            and config["trustedEvaluation"]["top3QuestionCost"] == old["trustedEvaluation"]["top3QuestionCost"]
            and config["trustedEvaluation"]["missThenGenericAdditionalCost"] == old["trustedEvaluation"]["missThenGenericAdditionalCost"]
        ),
        "prelock_transformed_model_and_execution_access_is_zero": all(value == 0 for value in config["preLockExposure"].values()),
        "protected_API_model_selection_authority_action_and_execution_are_closed": bool(
            not config["decisionRule"]["passAuthorizesImmediateProtectedRun"]
            and not config["decisionRule"]["passAuthorizesAPIAdditionalModelOrSyntheticLanguageShift"]
            and not config["decisionRule"]["passAuthorizesRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path,
                v195_protocol_path, bounded_helper_path, v200_path, v199_path, v195_path, v199_lock_path,
                v195_lock_path, *inputs.values(),
            )) and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "201-local-menu-presentation-robustness-design-audit", "experiment": config["experiment"],
        "passed": passed, "decision": "freeze_and_authorize_one_V201_local_development_robustness_realization" if passed else "reject_V201_design",
        "checks": checks, "prelock_exposure": config["preLockExposure"],
    }
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_V200_outcome": v200_path, "source_V199_outcome": v199_path,
        "source_V195_outcome": v195_path, "source_V199_lock": v199_lock_path, "source_V195_lock": v195_lock_path,
        **inputs, "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "V195_protocol": v195_protocol_path, "bounded_reasoning_helper": bounded_helper_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "201-local-menu-presentation-robustness-lock", "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_population_transformations_model_prompt_budget_parser_comparator_costs_or_gates": False,
            "run_exact_single_local_development_robustness_realization": True,
            "generate_on_missing_records_retry_or_select_outputs": False,
            "persist_or_manually_inspect_raw_model_outputs": False,
            "read_protected_language_or_run_API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
