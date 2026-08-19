#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v112_open_world_full_policy_transfer import population_gates, select_fresh_population


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v112-open-world-full-policy-transfer.json"
    parent_path = PROJECT_ROOT / "configs/v111-open-world-separability-audit-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v112-open-world-full-policy-transfer-plan.md"
    protocol_path = PROJECT_ROOT / "python/v112_open_world_full_policy_transfer.py"
    tests_path = PROJECT_ROOT / "python/test_v112_open_world_full_policy_transfer.py"
    runner_path = PROJECT_ROOT / "python/run_v112_open_world_full_policy_transfer.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v112_full_policy_transfer_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v112_full_policy_transfer.py"
    audit_path = PROJECT_ROOT / "outputs/v112-open-world-full-policy-transfer/design-audit.json"
    population_path = PROJECT_ROOT / "outputs/v112-open-world-full-policy-transfer/design/fresh-population.json"
    lock_path = PROJECT_ROOT / "configs/v112-open-world-full-policy-transfer-lock.json"
    language_path = PROJECT_ROOT / "outputs/v112-open-world-full-policy-transfer/fresh-language/development-transfer.jsonl"
    model_output = PROJECT_ROOT / "outputs/v112-open-world-full-policy-transfer/model-policy-transfer"
    if audit_path.exists() or population_path.exists() or lock_path.exists() or language_path.exists() or model_output.exists():
        raise RuntimeError("V112 design is already frozen, extracted, or evaluated")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    archive_path = PROJECT_ROOT / config["sourceArchive"]
    excluded_path = PROJECT_ROOT / config["excludedV101Population"]
    catalog_path = PROJECT_ROOT / config["visibleCatalog"]
    choices_path = PROJECT_ROOT / config["choiceCatalog"]
    manifest_path = PROJECT_ROOT / config["modelManifest"]
    inventory = json.loads(inventory_path.read_text())
    excluded = json.loads(excluded_path.read_text())
    choices = json.loads(choices_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    population = select_fresh_population(inventory, excluded, config)
    pop_checks = population_gates(population, config)
    write_json(population_path, population)

    constants = config["frozenPolicy"]
    checks = {
        "V111_positive_separability_outcome_is_exact_and_authorizes_full_policy_preregistration": bool(
            valid_lock(parent) and parent["outcome"]["passed"] and parent["outcome"]["quality_gate_pass"]
            and parent["outcome"]["selected_rule"]["family"] == "llm_abstain_only"
            and parent["authorization"]["preregister_full_deterministic_development_policy"]
            and not parent["authorization"]["read_protected_test_before_separate_lock"]
            and file_sha256(parent_lock_path) == parent["analysis_lock_sha256"]
        ),
        "fresh_population_is_balanced_scenario_covering_language_free_and_disjoint": all(pop_checks.values()),
        "fresh_population_uses_only_validation_and_excludes_all_V101_identifiers": bool(
            all(row["source_partition"] == "validation" for row in population["selected_population"])
            and config["freshPopulation"]["excludeEveryV101DevelopmentAndProtectedIdentifier"]
            and config["freshPopulation"]["selectionBeforeFreshLanguageExtraction"]
        ),
        "historical_only_policy_and_calibration_constants_are_exactly_frozen": bool(
            constants["knownActionConfidence"] == 56 / 61
            and constants["unsupportedActionConfidence"] == 28 / 32
            and constants["positiveNovelEvidenceProbability"] == 16 / 21
            and constants["negativeNovelEvidenceProbability"] == 16 / 107
            and constants["completeSafeHypothesisUniverseAlwaysRetained"]
            and constants["novelEvidenceCannotDefineSchemaCapabilityActionOrExecution"]
        ),
        "typed_choice_model_and_no_retry_condition_are_exact": bool(
            choices["choice_count"] == config["typedChoiceInterface"]["requiredChoiceCount"] == 17
            and len({row["choice_id"] for row in choices["choices"]}) == 17
            and manifest["repository"] == config["condition"]["repository"]
            and manifest["revision"] == config["condition"]["revision"]
            and manifest["weight_bytes"] == config["condition"]["weightBytes"]
            and not config["decoding"]["retryOnMalformedOutput"]
            and config["condition"]["totalGenerationCount"] == 240
        ),
        "quality_gates_are_joint_and_novel_evidence_is_not_action_authority": bool(
            config["qualityGates"]["minimumNovelEvidencePrecision"] == 0.70
            and config["qualityGates"]["minimumNovelEvidenceRecall"] == 0.50
            and config["qualityGates"]["maximumNovelEvidenceNonNovelFalsePositiveRate"] == 0.10
            and config["qualityGates"]["maximumPolicyFalseKnownAcceptanceRate"] == 0.10
            and config["qualityGates"]["maximumPolicyMeanDecisionRegret"] == 1.125
            and config["authorityBoundary"]["modelIsPermanentlyNonAuthoritativeEvidenceOnly"]
            and config["authorityBoundary"]["novelCandidateDoesNotEnterCapabilityCatalog"]
            and config["authorityBoundary"]["realExecutionCount"] == 0
        ),
        "protected_test_API_training_and_execution_remain_closed": bool(
            not config["decisionRule"]["passAuthorizesProtectedTestImmediately"]
            and not config["decisionRule"]["passAuthorizesSchemaInductionImmediately"]
            and not config["decisionRule"]["passAuthorizesSequentialPlanningAPITrainingCapabilityActionOrExecution"]
            and config["accessGates"]["maximumProtectedTestLanguageReadCount"] == 0
            and config["accessGates"]["maximumLLMAPICallCount"] == 0
            and config["accessGates"]["maximumAdapterTrainingRunCount"] == 0
        ),
        "locked_code_runtime_model_snapshot_and_output_absence_hold": bool(
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
        "schema_version": "112-open-world-full-policy-transfer-design-audit",
        "experiment": "v112_fresh_massive_non_authoritative_novelty_evidence_policy_design_audit",
        "passed": passed,
        "decision": "freeze_population_policy_and_authorize_one_fresh_transfer_run" if passed else "reject_V112_transfer",
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
        "config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path,
        "source_inventory": inventory_path, "source_archive": archive_path,
        "excluded_population": excluded_path, "visible_catalog": catalog_path,
        "choice_catalog": choices_path, "model_manifest": manifest_path,
        "baseline_lock": PROJECT_ROOT / parent_lock["baseline_lock"],
        "V109_result": PROJECT_ROOT / parent_lock["V109_result"],
        "fresh_population": population_path, "plan": plan_path, "protocol": protocol_path,
        "tests": tests_path, "runner": runner_path, "verifier": verifier_path,
        "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "112-open-world-full-policy-transfer-lock",
        "experiment": "v112_fresh_massive_non_authoritative_novelty_evidence_policy_transfer_lock",
        "config_payload": config,
        "baseline_config_payload": parent_lock["baseline_config_payload"],
        "authorization": {
            "modify_population_prompt_model_policy_confidences_comparators_metrics_gates_or_decision": False,
            "extract_exact_fresh_selected_development_language_once": True,
            "run_one_240_generation_local_transfer_once": True,
            "manually_inspect_language_or_raw_response": False,
            "read_protected_test_language": False,
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
