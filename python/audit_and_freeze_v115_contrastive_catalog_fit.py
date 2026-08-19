#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v115_contrastive_catalog_fit import (
    merged_excluded_population, population_gates, select_v115_population,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v115-contrastive-catalog-fit.json"
    plan_path = PROJECT_ROOT / "docs/v115-contrastive-catalog-fit-plan.md"
    protocol_path = PROJECT_ROOT / "python/v115_contrastive_catalog_fit.py"
    tests_path = PROJECT_ROOT / "python/test_v115_contrastive_catalog_fit.py"
    runner_path = PROJECT_ROOT / "python/run_v115_contrastive_catalog_fit.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v115_contrastive_catalog_fit_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v115_contrastive_catalog_fit.py"
    audit_path = PROJECT_ROOT / "outputs/v115-contrastive-catalog-fit/design-audit.json"
    population_path = PROJECT_ROOT / "outputs/v115-contrastive-catalog-fit/design/fresh-population.json"
    lock_path = PROJECT_ROOT / "configs/v115-contrastive-catalog-fit-lock.json"
    language_path = PROJECT_ROOT / "outputs/v115-contrastive-catalog-fit/fresh-language/development-contrastive.jsonl"
    model_output = PROJECT_ROOT / "outputs/v115-contrastive-catalog-fit/model-contrastive"
    if any(path.exists() for path in (audit_path, population_path, lock_path, language_path, model_output)):
        raise RuntimeError("V115 design is already frozen, extracted, or evaluated")

    config = json.loads(config_path.read_text())
    parent_outcome_path = PROJECT_ROOT / config["parentV114OutcomeLock"]
    parent_outcome = json.loads(parent_outcome_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent_outcome["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    v112_lock_path = PROJECT_ROOT / parent_lock["V112_lock"]
    v112_lock = json.loads(v112_lock_path.read_text())
    v112_config = parent_lock["V112_config_payload"]
    baseline_config = parent_lock["baseline_config_payload"]

    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    archive_path = PROJECT_ROOT / config["sourceArchive"]
    excluded_paths = [PROJECT_ROOT / path for path in config["excludedPopulations"]]
    visible_catalog_path = PROJECT_ROOT / parent_lock["visible_catalog"]
    choice_catalog_path = PROJECT_ROOT / parent_lock["choice_catalog"]
    model_manifest_path = PROJECT_ROOT / parent_lock["model_manifest"]
    baseline_lock_path = PROJECT_ROOT / parent_lock["baseline_lock"]
    v109_result_path = PROJECT_ROOT / parent_lock["V109_result"]

    inventory = json.loads(inventory_path.read_text())
    excluded_populations = [json.loads(path.read_text()) for path in excluded_paths]
    merged_exclusions = merged_excluded_population(excluded_populations)
    population = select_v115_population(inventory, merged_exclusions, config)
    pop_checks = population_gates(population, config)
    write_json(population_path, population)

    selected_ids = {row["candidate_id"] for row in population["selected_population"]}
    excluded_sets = [
        {row["candidate_id"] for row in source["selected_population"]}
        for source in excluded_populations
    ]
    choices = json.loads(choice_catalog_path.read_text())
    manifest = json.loads(model_manifest_path.read_text())
    evidence_gates = config["contrastiveEvidenceGates"]
    checks = {
        "V114_is_valid_negative_and_authorizes_no_induction_or_protected_access": bool(
            valid_lock(parent_outcome) and valid_lock(parent_lock) and valid_lock(v112_lock)
            and parent_outcome["outcome"]["passed"]
            and not parent_outcome["outcome"]["novel_evidence_pass"]
            and not parent_outcome["authorization"]["preregister_sandboxed_typed_induction_feasibility"]
            and not parent_outcome["authorization"]["open_original_protected_test"]
            and not parent_outcome["authorization"]["run_API_model_or_train_adapter"]
            and file_sha256(parent_lock_path) == parent_outcome["analysis_lock_sha256"]
        ),
        "fresh_population_is_balanced_scenario_covering_language_free_and_disjoint": all(pop_checks.values()),
        "fresh_population_is_test_only_and_excludes_V101_V112_and_V114": bool(
            len(excluded_populations) == 3
            and all(row["source_partition"] == "test" for row in population["selected_population"])
            and all(row["role"] == config["extraction"]["role"] for row in population["selected_population"])
            and all(not (selected_ids & excluded) for excluded in excluded_sets)
            and config["freshPopulation"]["excludeEveryPriorAndProtectedIdentifier"]
            and config["freshPopulation"]["selectionBeforeLanguageExtractionOrModelGeneration"]
        ),
        "V115_changes_evidence_without_retuning_V114": bool(
            config["condition"]["generationCountPerFixture"] == 2
            and config["condition"]["totalGenerationCount"] == 480
            and config["condition"]["retryCount"] == 0
            and set(config["contrastiveInterface"]["verdicts"]) == {"C00", "O00", "N00", "U00", "A00"}
            and config["contrastiveInterface"]["completeDeclaredCatalogShown"]
            and not config["contrastiveInterface"]["hiddenRecordGroundTruthClassIntentAndScenarioShown"]
            and evidence_gates["minimumExplicitNovelPrecision"] == 0.70
            and evidence_gates["minimumExplicitNovelRecall"] == 0.50
            and evidence_gates["maximumExplicitNovelNonNovelFalsePositiveRate"] == 0.10
            and evidence_gates["maximumExplicitNovelECE"] == 0.15
            and config["reuseExactV112CombinedPolicyQualityGates"]
        ),
        "first_pass_model_prompt_decoder_policy_and_costs_are_exact_V112": bool(
            parent_lock["V112_config_payload"] == v112_lock["config_payload"] == v112_config
            and parent_lock["baseline_config_payload"] == v112_lock["baseline_config_payload"] == baseline_config
            and choices["choice_count"] == v112_config["typedChoiceInterface"]["requiredChoiceCount"] == 17
            and manifest["repository"] == v112_config["condition"]["repository"]
            and manifest["revision"] == v112_config["condition"]["revision"]
            and manifest["weight_bytes"] == v112_config["condition"]["weightBytes"]
            and not v112_config["decoding"]["retryOnMalformedOutput"]
            and len(v112_config["qualityGates"]) == 17
        ),
        "authority_protected_API_training_induction_and_execution_boundaries_are_closed": bool(
            config["authorityBoundary"]["modelIsPermanentlyNonAuthoritativeEvidenceOnly"]
            and config["authorityBoundary"]["completeSafeHypothesisUniverseAlwaysRetained"]
            and config["authorityBoundary"]["realExecutionCount"] == 0
            and not config["decisionRule"]["passAuthorizesOriginalProtectedTest"]
            and not config["decisionRule"]["passAuthorizesSchemaInduction"]
            and not config["decisionRule"]["passAuthorizesRicherSequentialPlanning"]
            and not config["decisionRule"]["passAuthorizesAPITrainingActionOrExecution"]
            and config["accessGates"]["maximumProtectedTestLanguageReadCount"] == 0
            and config["accessGates"]["maximumLLMAPICallCount"] == 0
            and config["accessGates"]["maximumAdapterTrainingRunCount"] == 0
        ),
        "locked_code_runtime_snapshot_and_output_absence_hold": bool(
            all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path))
            and metadata.version("numpy") == "2.5.1"
            and metadata.version("scikit-learn") == "1.9.0"
            and metadata.version("mlx-lm") == "0.31.3"
            and Path(manifest["snapshot_path"]).is_dir()
            and not language_path.exists() and not model_output.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "115-contrastive-catalog-fit-design-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_population_and_authorize_one_two_pass_run" if passed else "reject_V115_design",
        "checks": checks, "population_gates": pop_checks,
        "population_summary": {key: population[key] for key in (
            "selected_record_count", "class_counts", "scenario_counts",
            "excluded_identifier_overlap_count", "contains_language", "selected_population_sha256",
        )},
        "prelock_access": {
            "source_inventory_automatic_read_count": 1, "source_archive_read_count": 0,
            "fresh_language_extraction_count": 0, "protected_test_language_read_count": 0,
            "manual_language_or_raw_response_inspection_count": 0, "model_load_count": 0,
            "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path, "parent_outcome": parent_outcome_path,
        "parent_analysis_lock": parent_lock_path, "V112_lock": v112_lock_path,
        "source_inventory": inventory_path, "source_archive": archive_path,
        "V101_population": excluded_paths[0], "V112_population": excluded_paths[1],
        "V114_population": excluded_paths[2], "visible_catalog": visible_catalog_path,
        "choice_catalog": choice_catalog_path, "model_manifest": model_manifest_path,
        "baseline_lock": baseline_lock_path, "V109_result": v109_result_path,
        "fresh_population": population_path, "plan": plan_path, "protocol": protocol_path,
        "tests": tests_path, "runner": runner_path, "verifier": verifier_path,
        "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "115-contrastive-catalog-fit-lock",
        "experiment": config["experiment"], "config_payload": config,
        "V112_config_payload": v112_config, "baseline_config_payload": baseline_config,
        "authorization": {
            "modify_population_model_prompts_interfaces_policy_thresholds_costs_metrics_gates_or_decision": False,
            "extract_exact_selected_fresh_language_once": True,
            "run_one_480_generation_two_pass_local_condition_once": True,
            "manually_inspect_language_or_raw_response": False,
            "read_original_protected_test_language": False,
            "run_API_model_or_train_adapter": False,
            "begin_schema_induction_or_richer_sequential_planning": False,
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
