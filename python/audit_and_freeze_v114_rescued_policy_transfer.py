#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v114_rescued_policy_transfer import (
    merged_excluded_population, population_gates, select_v114_population,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v114-rescued-policy-transfer.json"
    plan_path = PROJECT_ROOT / "docs/v114-rescued-policy-transfer-plan.md"
    protocol_path = PROJECT_ROOT / "python/v114_rescued_policy_transfer.py"
    tests_path = PROJECT_ROOT / "python/test_v114_rescued_policy_transfer.py"
    runner_path = PROJECT_ROOT / "python/run_v114_rescued_policy_transfer.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v114_rescued_policy_transfer_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v114_rescued_policy_transfer.py"
    audit_path = PROJECT_ROOT / "outputs/v114-rescued-policy-transfer/design-audit.json"
    population_path = PROJECT_ROOT / "outputs/v114-rescued-policy-transfer/design/fresh-population.json"
    lock_path = PROJECT_ROOT / "configs/v114-rescued-policy-transfer-lock.json"
    language_path = PROJECT_ROOT / "outputs/v114-rescued-policy-transfer/fresh-language/development-transfer-2.jsonl"
    model_output = PROJECT_ROOT / "outputs/v114-rescued-policy-transfer/model-policy-transfer"
    if any(path.exists() for path in (audit_path, population_path, lock_path, language_path, model_output)):
        raise RuntimeError("V114 design is already frozen, extracted, or evaluated")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV113OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    r1_lock_path = PROJECT_ROOT / parent_lock["V112r1_lock"]
    r1_lock = json.loads(r1_lock_path.read_text())
    v112_lock_path = PROJECT_ROOT / r1_lock["parent_lock"]
    v112_lock = json.loads(v112_lock_path.read_text())
    v112_config = v112_lock["config_payload"]

    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    archive_path = PROJECT_ROOT / config["sourceArchive"]
    excluded_paths = [PROJECT_ROOT / path for path in config["excludedPopulations"]]
    visible_catalog_path = PROJECT_ROOT / v112_config["visibleCatalog"]
    choice_catalog_path = PROJECT_ROOT / v112_config["choiceCatalog"]
    model_manifest_path = PROJECT_ROOT / v112_config["modelManifest"]
    baseline_lock_path = PROJECT_ROOT / v112_lock["baseline_lock"]
    v109_result_path = PROJECT_ROOT / v112_lock["V109_result"]

    inventory = json.loads(inventory_path.read_text())
    excluded_populations = [json.loads(path.read_text()) for path in excluded_paths]
    merged_exclusions = merged_excluded_population(excluded_populations)
    population = select_v114_population(inventory, merged_exclusions, config)
    pop_checks = population_gates(population, config)
    write_json(population_path, population)
    choices = json.loads(choice_catalog_path.read_text())
    manifest = json.loads(model_manifest_path.read_text())
    selected_rule = parent["outcome"]["selected"]["rule"]
    spec = config["pairedRescueEvaluation"]

    v101_ids = {row["candidate_id"] for row in excluded_populations[0]["selected_population"]}
    v112_ids = {row["candidate_id"] for row in excluded_populations[1]["selected_population"]}
    selected_ids = {row["candidate_id"] for row in population["selected_population"]}
    checks = {
        "V113_is_exact_feasible_historical_only_and_authorizes_new_population": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["feasible_rescue_exists"]
            and parent["authorization"]["preregister_selected_rescue_on_new_disjoint_population"]
            and not parent["authorization"]["read_protected_test_before_separate_lock"]
            and file_sha256(parent_lock_path) == parent["analysis_lock_sha256"]
            and valid_lock(parent_lock) and valid_lock(r1_lock) and valid_lock(v112_lock)
        ),
        "selected_V113_rule_and_confidence_are_frozen_exactly": bool(
            selected_rule == config["selectedRescueRule"]
            and config["selectedRescueRule"] == {
                "family": "proposal_score_and_gap", "minimum_score": 0.6,
                "maximum_gap": 0.15, "complexity": 2,
            }
            and config["rescueActionConfidence"] == 0.75
        ),
        "fresh_population_is_balanced_scenario_covering_language_free_and_disjoint": all(pop_checks.values()),
        "fresh_population_uses_only_test_and_excludes_V101_protected_and_V112": bool(
            all(row["source_partition"] == "test" for row in population["selected_population"])
            and all(row["role"] == config["extraction"]["role"] for row in population["selected_population"])
            and not (selected_ids & v101_ids) and not (selected_ids & v112_ids)
            and config["freshPopulation"]["excludeEveryPriorAndProtectedIdentifier"]
            and config["freshPopulation"]["selectionBeforeLanguageExtractionOrModelGeneration"]
        ),
        "paired_full_policy_and_mechanism_decisions_are_prospectively_separate": bool(
            config["decisionRule"]["qualificationLayers"] == [
                "absolute full-policy transfer", "paired V113 rescue-mechanism transfer",
            ]
            and spec["minimumEligibleDisagreementCountForMechanismConclusion"] == 8
            and spec["minimumTriggeredRescueCountForMechanismConclusion"] == 4
            and spec["minimumRescuePrecision"] == 0.75
            and spec["minimumNetCorrectedErrors"] == 1
            and spec["requireExactNovelEvidenceIdentity"]
        ),
        "typed_choice_prompt_model_decoding_policy_costs_and_seventeen_gates_are_exact_V112": bool(
            parent_lock["V112_config_payload"] == v112_config
            and parent_lock["baseline_config_payload"] == v112_lock["baseline_config_payload"]
            and choices["choice_count"] == v112_config["typedChoiceInterface"]["requiredChoiceCount"] == 17
            and manifest["repository"] == v112_config["condition"]["repository"]
            and manifest["revision"] == v112_config["condition"]["revision"]
            and manifest["weight_bytes"] == v112_config["condition"]["weightBytes"]
            and not v112_config["decoding"]["retryOnMalformedOutput"]
            and v112_config["condition"]["totalGenerationCount"] == 240
            and len(v112_config["qualityGates"]) == 17
        ),
        "authority_protected_API_training_and_execution_boundaries_remain_closed": bool(
            config["authorityBoundary"]["modelIsPermanentlyNonAuthoritativeEvidenceOnly"]
            and config["authorityBoundary"]["completeSafeHypothesisUniverseAlwaysRetained"]
            and config["authorityBoundary"]["novelCandidateStillAsksAndCannotDefineCapability"]
            and config["authorityBoundary"]["realExecutionCount"] == 0
            and not config["decisionRule"]["passAuthorizesProtectedTestImmediately"]
            and not config["decisionRule"]["passAuthorizesSchemaInductionImmediately"]
            and not config["decisionRule"]["passAuthorizesSequentialPlanningAPITrainingActionOrExecution"]
            and v112_config["accessGates"]["maximumProtectedTestLanguageReadCount"] == 0
            and v112_config["accessGates"]["maximumLLMAPICallCount"] == 0
            and v112_config["accessGates"]["maximumAdapterTrainingRunCount"] == 0
        ),
        "locked_code_runtime_snapshot_and_output_absence_hold": bool(
            all(path.is_file() for path in (
                plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path,
            ))
            and metadata.version("numpy") == "2.5.1"
            and metadata.version("scikit-learn") == "1.9.0"
            and metadata.version("mlx-lm") == "0.31.3"
            and Path(manifest["snapshot_path"]).is_dir()
            and not language_path.exists() and not model_output.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "114-rescued-policy-transfer-design-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_population_and_authorize_one_paired_transfer_run" if passed else "reject_V114_transfer",
        "checks": checks, "population_gates": pop_checks,
        "population_summary": {key: population[key] for key in (
            "selected_record_count", "class_counts", "scenario_counts",
            "excluded_identifier_overlap_count", "contains_language", "selected_population_sha256",
        )},
        "prelock_access": {
            "source_inventory_automatic_read_count": 1, "source_archive_read_count": 0,
            "fresh_language_extraction_count": 0, "protected_test_language_read_count": 0,
            "manual_language_or_raw_response_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path, "parent_outcome": parent_path,
        "parent_analysis_lock": parent_lock_path, "V112r1_lock": r1_lock_path,
        "V112_lock": v112_lock_path, "source_inventory": inventory_path,
        "source_archive": archive_path, "V101_population": excluded_paths[0],
        "V112_population": excluded_paths[1], "visible_catalog": visible_catalog_path,
        "choice_catalog": choice_catalog_path, "model_manifest": model_manifest_path,
        "baseline_lock": baseline_lock_path, "V109_result": v109_result_path,
        "fresh_population": population_path, "plan": plan_path, "protocol": protocol_path,
        "tests": tests_path, "runner": runner_path, "verifier": verifier_path,
        "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "114-rescued-policy-transfer-lock",
        "experiment": config["experiment"], "config_payload": config,
        "V112_config_payload": v112_config,
        "baseline_config_payload": v112_lock["baseline_config_payload"],
        "selected_rescue_rule": selected_rule,
        "authorization": {
            "modify_population_model_prompt_interface_policy_rescue_thresholds_costs_metrics_gates_or_decision": False,
            "extract_exact_selected_fresh_language_once": True,
            "run_one_240_generation_local_transfer_once": True,
            "feed_each_model_response_to_both_paired_policies": True,
            "manually_inspect_language_or_raw_response": False,
            "read_original_protected_test_language": False,
            "run_API_model_or_train_adapter": False,
            "prune_hypotheses_define_capability_or_grant_belief_action_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
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
