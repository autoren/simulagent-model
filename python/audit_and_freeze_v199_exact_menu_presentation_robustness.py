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
    config_path = PROJECT_ROOT / "configs/v199-exact-menu-presentation-robustness.json"
    plan_path = PROJECT_ROOT / "docs/v199-exact-menu-presentation-robustness-plan.md"
    protocol_path = PROJECT_ROOT / "python/v199_exact_menu_presentation_robustness.py"
    tests_path = PROJECT_ROOT / "python/test_v199_exact_menu_presentation_robustness.py"
    runner_path = PROJECT_ROOT / "python/run_v199_exact_menu_presentation_robustness.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v199_exact_menu_presentation_robustness_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v199_exact_menu_presentation_robustness.py"
    audit_path = PROJECT_ROOT / "outputs/v199-exact-menu-presentation-robustness/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v199-exact-menu-presentation-robustness-lock.json"
    output_root = PROJECT_ROOT / "outputs/v199-exact-menu-presentation-robustness/census"
    outcome_path = PROJECT_ROOT / "configs/v199-exact-menu-presentation-robustness-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V199 is already preregistered, run, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV198OutcomeLock"]
    v195_path = PROJECT_ROOT / config["sourceV195OutcomeLock"]
    v193_path = PROJECT_ROOT / config["sourceV193OutcomeLock"]
    v191_path = PROJECT_ROOT / config["sourceV191OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v195 = json.loads(v195_path.read_text())
    v193 = json.loads(v193_path.read_text())
    v191 = json.loads(v191_path.read_text())
    inputs = {
        "development_identities": PROJECT_ROOT / config["developmentIdentities"],
        "hidden_targets": PROJECT_ROOT / config["hiddenTargets"],
        "canonical_visible_menu": PROJECT_ROOT / config["canonicalVisibleMenu"],
        "canonical_hidden_option_map": PROJECT_ROOT / config["canonicalHiddenOptionMap"],
        "roadmap": PROJECT_ROOT / config["roadmap"],
    }
    future = config["futurePairedDevelopmentGates"]
    checks = {
        "V198_authorizes_a_separate_model_free_robustness_design": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_confirmation_gates_passed"]
            and parent["authorization"]["update_roadmap_and_preregister_separate_model_free_robustness_design"]
        ),
        "V195_V193_and_V191_sources_are_valid": bool(
            valid_lock(v195) and v195["outcome"]["passed"]
            and valid_lock(v193) and v193["outcome"]["passed"]
            and valid_lock(v191) and v191["outcome"]["passed"]
        ),
        "identity_target_and_menu_inputs_match_frozen_sources": bool(
            file_sha256(inputs["development_identities"]) == v191["development_identities_sha256"]
            and file_sha256(inputs["hidden_targets"]) == v191["hidden_targets_sha256"]
            and file_sha256(inputs["canonical_visible_menu"]) == v193["visible_menu_sha256"]
            and file_sha256(inputs["canonical_hidden_option_map"]) == v193["hidden_option_map_sha256"]
        ),
        "prospective_robustness_gates_are_fixed_and_nontrivial": bool(
            future["canonicalPrimaryTop3Recall"] == v195["outcome"]["summary"]["primary_top3_recall"]
            and future["canonicalMacroTop3Recall"] == v195["outcome"]["summary"]["macro_top3_recall"]
            and future["canonicalPrimaryTop3MeanCost"] == v195["outcome"]["summary"]["primary_top3_mean_cost"]
            and future["maximumPerVariantPrimaryTop3RecallDrop"] <= 0.05
            and future["maximumPerVariantPrimaryTop3CostIncrease"] <= 0.02
            and future["minimumPerVariantTop1ContractAgreementWithCanonical"] >= 0.80
            and future["minimumPerVariantMeanTop3ContractSetJaccardWithCanonical"] >= 0.80
            and future["minimumPerVariantIncrementalPrimaryImprovementOverTransformedCHAR_LAST"] >= 0.01
        ),
        "transformations_are_text_blind_and_exact_by_contract": bool(
            config["transformationContract"]["assignmentInputs"] == ["fixed salt", "record_id", "canonical option_id"]
            and config["transformationContract"]["utteranceConversationTruthKindAndPriorNeverUsedForAssignment"]
            and config["transformationContract"]["sameSemanticDomainAndIntentMultisetRequired"]
            and config["transformationContract"]["optionToContractBijectionRequired"]
            and config["transformationContract"]["hiddenTargetExpressibleExactlyOnceRequired"]
        ),
        "prelock_language_model_and_execution_access_is_zero": all(value == 0 for value in config["preLockExposure"].values()),
        "immediate_model_protected_API_authority_action_and_execution_are_closed": bool(
            not config["decisionRule"]["passAuthorizesImmediateLanguageScoringOrModelRun"]
            and not config["decisionRule"]["passAuthorizesProtectedAccessAPITrainingRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path,
                parent_path, v195_path, v193_path, v191_path, *inputs.values(),
            )) and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "199-exact-menu-presentation-robustness-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_text_free_V199_census" if passed else "reject_V199_design",
        "checks": checks,
        "prelock_exposure": config["preLockExposure"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_V198_outcome": parent_path, "source_V195_outcome": v195_path,
        "source_V193_outcome": v193_path, "source_V191_outcome": v191_path, **inputs,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path,
        "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "199-exact-menu-presentation-robustness-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_population_transformations_metrics_or_gates": False,
            "run_exact_single_text_free_census": True,
            "read_or_score_language_or_run_model": False,
            "run_protected_API_training_registration_authority_action_or_execution": False,
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
