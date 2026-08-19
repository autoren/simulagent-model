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
    config_path = PROJECT_ROOT / "configs/v200-transformed-char-last-controls.json"
    plan_path = PROJECT_ROOT / "docs/v200-transformed-char-last-controls-plan.md"
    protocol_path = PROJECT_ROOT / "python/v200_transformed_char_last_controls.py"
    tests_path = PROJECT_ROOT / "python/test_v200_transformed_char_last_controls.py"
    runner_path = PROJECT_ROOT / "python/run_v200_transformed_char_last_controls.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v200_transformed_char_last_controls_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v200_transformed_char_last_controls.py"
    v194_protocol_path = PROJECT_ROOT / "python/v194_deterministic_language_menu_rankers.py"
    audit_path = PROJECT_ROOT / "outputs/v200-transformed-char-last-controls/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v200-transformed-char-last-controls-lock.json"
    output_root = PROJECT_ROOT / "outputs/v200-transformed-char-last-controls/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v200-transformed-char-last-controls-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V200 is already preregistered, run, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV199OutcomeLock"]
    v194_path = PROJECT_ROOT / config["sourceV194OutcomeLock"]
    v193_path = PROJECT_ROOT / config["sourceV193OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v194 = json.loads(v194_path.read_text())
    v193 = json.loads(v193_path.read_text())
    v194_lock_path = PROJECT_ROOT / v194["evaluation_lock"]
    v194_lock = json.loads(v194_lock_path.read_text())
    inputs = {
        "development_language": PROJECT_ROOT / config["developmentLanguage"],
        "hidden_targets": PROJECT_ROOT / config["hiddenTargets"],
        "visible_menu_variants": PROJECT_ROOT / config["visibleMenuVariants"],
        "hidden_variant_maps": PROJECT_ROOT / config["hiddenVariantMaps"],
        "canonical_hidden_option_map": PROJECT_ROOT / config["canonicalHiddenOptionMap"],
        "canonical_CHAR_LAST_predictions": PROJECT_ROOT / config["canonicalCHARLASTPredictions"],
        "primary_prior": PROJECT_ROOT / config["primaryPrior"],
    }
    old_char = next(row for row in v194_lock["config_payload"]["rankers"] if row["ranker_id"] == "CHAR_LAST")
    checks = {
        "V199_authorizes_only_separate_deterministic_development_evaluation": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_separate_deterministic_development_evaluation_only"]
            and not parent["authorization"]["immediate_language_scoring_or_model_run"]
        ),
        "V194_and_V193_sources_are_valid": bool(
            valid_lock(v194) and v194["outcome"]["passed"] and valid_lock(v194_lock)
            and valid_lock(v193) and v193["outcome"]["passed"]
        ),
        "language_variants_maps_predictions_and_prior_are_exact": bool(
            file_sha256(inputs["development_language"]) == v194_lock["development_language_sha256"]
            and file_sha256(inputs["visible_menu_variants"]) == parent["visible_menu_variants_sha256"]
            and file_sha256(inputs["hidden_variant_maps"]) == parent["hidden_variant_maps_sha256"]
            and file_sha256(inputs["canonical_hidden_option_map"]) == v193["hidden_option_map_sha256"]
            and file_sha256(inputs["canonical_CHAR_LAST_predictions"]) == v194["shadow_predictions_sha256"]
            and file_sha256(inputs["primary_prior"]) == v193["primary_prior_sha256"]
            and inputs["hidden_targets"].is_file()
        ),
        "CHAR_LAST_and_controller_are_unchanged": bool(
            {key: config["ranker"][key] for key in old_char} == old_char
            and config["ranker"]["rankedOutputLength"] == v194_lock["config_payload"]["evaluation"]["rankedOutputLength"]
            and config["evaluation"]["top3QuestionCost"] == v194_lock["config_payload"]["evaluation"]["top3QuestionCost"]
            and config["evaluation"]["missThenGenericAdditionalCost"] == v194_lock["config_payload"]["evaluation"]["missThenGenericAdditionalCost"]
        ),
        "prelock_language_scoring_model_and_execution_access_is_zero": all(value == 0 for value in config["preLockExposure"].values()),
        "model_protected_API_authority_action_and_execution_are_closed": bool(
            not config["decisionRule"]["passAuthorizesImmediateModelRunOrProtectedAccess"]
            and not config["decisionRule"]["passAuthorizesAPITrainingRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path,
                v194_protocol_path, parent_path, v194_path, v193_path, v194_lock_path, *inputs.values(),
            )) and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "200-transformed-char-last-controls-design-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_and_authorize_one_V200_deterministic_evaluation" if passed else "reject_V200_design",
        "checks": checks, "prelock_exposure": config["preLockExposure"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_V199_outcome": parent_path, "source_V194_outcome": v194_path,
        "source_V193_outcome": v193_path, "source_V194_lock": v194_lock_path, **inputs,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "V194_protocol": v194_protocol_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "200-transformed-char-last-controls-lock", "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_population_transformations_ranker_metrics_or_gates": False,
            "run_exact_single_deterministic_development_evaluation": True,
            "run_model_or_read_protected_language": False,
            "run_API_training_registration_authority_action_or_execution": False,
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
