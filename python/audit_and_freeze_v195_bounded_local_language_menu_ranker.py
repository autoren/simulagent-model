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


def manifest_snapshot_is_exact(manifest: dict[str, Any], config: dict[str, Any]) -> bool:
    model = config["model"]
    snapshot = Path(manifest["snapshot_path"])
    if not (
        manifest["repository"] == model["repository"]
        and manifest["revision"] == model["revision"]
        and manifest["quantization_bits"] == model["quantizationBits"]
        and snapshot.is_dir()
        and len(manifest["files"]) == manifest["file_count"]
    ):
        return False
    for row in manifest["files"]:
        path = snapshot / row["path"]
        if not path.is_file() or path.stat().st_size != row["size"]:
            return False
        if row.get("sha256") and file_sha256(path) != row["sha256"]:
            return False
    return sum(row["size"] for row in manifest["files"] if row["path"].endswith(".safetensors")) == manifest["weight_bytes"]


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v195-bounded-local-language-menu-ranker.json"
    plan_path = PROJECT_ROOT / "docs/v195-bounded-local-language-menu-ranker-plan.md"
    protocol_path = PROJECT_ROOT / "python/v195_bounded_local_language_menu_ranker.py"
    tests_path = PROJECT_ROOT / "python/test_v195_bounded_local_language_menu_ranker.py"
    runner_path = PROJECT_ROOT / "python/run_v195_bounded_local_language_menu_ranker.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v195_bounded_local_language_menu_ranker_outcome.py"
    bounded_helper_path = PROJECT_ROOT / "python/v154_adaptive_local_question_order.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v195_bounded_local_language_menu_ranker.py"
    audit_path = PROJECT_ROOT / "outputs/v195-bounded-local-language-menu-ranker/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v195-bounded-local-language-menu-ranker-lock.json"
    realization_path = PROJECT_ROOT / "outputs/v195-bounded-local-language-menu-ranker/model-realization"
    outcome_path = PROJECT_ROOT / "configs/v195-bounded-local-language-menu-ranker-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, realization_path, outcome_path)):
        raise RuntimeError("V195 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV194OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_evaluation_lock_path = PROJECT_ROOT / parent["evaluation_lock"]
    parent_evaluation_lock = json.loads(parent_evaluation_lock_path.read_text())
    inputs = {
        "development_language": PROJECT_ROOT / config["developmentLanguage"],
        "hidden_targets": PROJECT_ROOT / config["hiddenTargets"],
        "visible_menu": PROJECT_ROOT / config["visibleMenu"],
        "hidden_option_map": PROJECT_ROOT / config["hiddenOptionMap"],
        "primary_prior": PROJECT_ROOT / config["primaryPrior"],
        "fixed_hierarchy_target_costs": PROJECT_ROOT / config["fixedHierarchyTargetCosts"],
        "deterministic_ranker_results": PROJECT_ROOT / config["deterministicRankerResults"],
        "model_manifest": PROJECT_ROOT / config["modelManifest"],
    }
    manifest = json.loads(inputs["model_manifest"].read_text())
    source_keys = (
        "development_language", "hidden_targets", "visible_menu", "hidden_option_map",
        "primary_prior", "fixed_hierarchy_target_costs",
    )
    checks = {
        "V194_is_valid_and_authorizes_exactly_one_bounded_local_comparator": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_one_bounded_local_model_shadow_comparator"]
            and not parent["authorization"]["immediate_model_run_or_API_fallback"]
            and valid_lock(parent_evaluation_lock)
        ),
        "fresh_language_targets_menu_prior_and_cost_inputs_match_V194_lock": all(
            inputs[key].is_file()
            and file_sha256(inputs[key]) == parent_evaluation_lock[f"{key}_sha256"]
            for key in source_keys
        ),
        "deterministic_champion_artifact_matches_V194_outcome": bool(
            inputs["deterministic_ranker_results"].is_file()
            and file_sha256(inputs["deterministic_ranker_results"]) == parent["ranker_results_sha256"]
        ),
        "model_snapshot_revision_and_files_are_exact": manifest_snapshot_is_exact(manifest, config),
        "single_bounded_low_reasoning_condition_is_fixed": bool(
            config["model"]["samplesPerPrompt"] == 1
            and config["model"]["retryCount"] == 0
            and config["model"]["modelLoadLimit"] == 1
            and config["model"]["enableThinking"]
            and config["model"]["reasoningEffort"] == "low"
            and config["model"]["reasoningPhaseMaximumTokens"] == 48
            and config["model"]["finalPhaseMaximumTokens"] == 64
            and config["model"]["mechanicallyForceCloseThinkingBeforeFinalPhase"]
            and config["model"]["temperature"] == 0.0
        ),
        "incremental_gate_is_stricter_than_deterministic_champion": bool(
            config["qualificationGates"]["minimumIncrementalPrimaryImprovementOverDeterministicChampion"] == 0.01
            and config["qualificationGates"]["maximumPrimaryTop3MeanCost"]
            == config["trustedEvaluation"]["deterministicChampionPrimaryTop3MeanCost"] - 0.01
        ),
        "prelock_language_model_and_execution_access_is_zero": all(
            value == 0 for value in config["preLockExposure"].values()
        ),
        "API_protected_authority_action_and_execution_remain_closed": bool(
            not config["decisionRule"]["passAuthorizesImmediateConfirmation"]
            and not config["decisionRule"]["passAuthorizesAPIFallbackOrAdditionalModelCondition"]
            and not config["decisionRule"]["passAuthorizesProtectedAccessRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, parent_path, parent_evaluation_lock_path, plan_path, protocol_path,
                tests_path, runner_path, verifier_path, bounded_helper_path, auditor_path,
                *inputs.values(),
            ))
            and not realization_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "195-bounded-local-language-menu-ranker-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V195_local_development_realization" if passed else "reject_V195_design",
        "checks": checks,
        "prelock_exposure": config["preLockExposure"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V194_outcome": parent_path,
        "parent_V194_evaluation_lock": parent_evaluation_lock_path,
        **inputs,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "bounded_reasoning_helper": bounded_helper_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "195-bounded-local-language-menu-ranker-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_population_prompt_model_decode_budgets_parser_costs_or_gates": False,
            "run_exact_single_bounded_local_development_realization": True,
            "generate_on_missing_fixtures_or_retry": False,
            "persist_or_manually_inspect_raw_model_outputs": False,
            "run_API_training_protected_access_registration_authority_action_or_execution": False,
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
