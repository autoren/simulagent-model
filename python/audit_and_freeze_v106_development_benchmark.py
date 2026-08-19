#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

import mlx_lm
import numpy
import sklearn

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v106-open-world-development-benchmark.json"
    parent_path = PROJECT_ROOT / "configs/v105-open-world-interface-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v106-open-world-development-benchmark-plan.md"
    protocol_path = PROJECT_ROOT / "python/v106_open_world_benchmark.py"
    tests_path = PROJECT_ROOT / "python/test_v106_open_world_benchmark.py"
    runner_path = PROJECT_ROOT / "python/run_v106_open_world_development_benchmark.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v106_development_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v106_development_benchmark.py"
    failed_audit_path = PROJECT_ROOT / "outputs/v106-open-world-development/development-design-audit.json"
    technical_path = PROJECT_ROOT / "configs/v106-open-world-development-benchmark-technical-outcome-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v106-open-world-development/development-design-audit-r1.json"
    lock_path = PROJECT_ROOT / "configs/v106-open-world-development-benchmark-lock.json"
    output_root = PROJECT_ROOT / "outputs/v106-open-world-development/development-baselines"
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V106 development benchmark is already frozen or materialized")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    technical = json.loads(technical_path.read_text())
    interface_lock = json.loads((PROJECT_ROOT / parent["interface_lock"]).read_text())
    language_outcome = json.loads((PROJECT_ROOT / interface_lock["parent_language_outcome"]).read_text())
    catalog_path = PROJECT_ROOT / config["visibleCatalog"]
    hypotheses_path = PROJECT_ROOT / config["safeHypothesisUniverse"]
    controls_path = PROJECT_ROOT / config["controlledInsufficientIdentifiers"]
    catalog = json.loads(catalog_path.read_text())
    hypotheses = json.loads(hypotheses_path.read_text())
    controls = json.loads(controls_path.read_text())
    split = config["developmentSplit"]
    baselines = config["deterministicBaselines"]
    baseline_gates = config["baselineOutcomeGates"]
    model = config["futureLocalModelCondition"]
    model_gates = config["futureModelDevelopmentGates"]
    authority = config["authorityBoundary"]
    exposure = config["preLockExposure"]
    checks = {
        "exact_zero_access_technical_repair_is_authorized": bool(
            valid_lock(technical) and technical["outcome"]["passed"]
            and not technical["outcome"]["scientific_benchmark_ran"]
            and technical["outcome"]["development_language_read_count"] == 0
            and technical["outcome"]["protected_test_language_read_count"] == 0
            and technical["outcome"]["model_load_count"] == 0
            and technical["authorization"]["repair_audit_expected_model_gate_count_from_14_to_15"]
            and file_sha256(failed_audit_path) == technical["failed_audit_sha256"]
            and file_sha256(config_path) == technical["config_sha256"]
        ),
        "V105_outcome_is_exact_and_authorizes_baseline_preregistration": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_baselines_metrics_costs_calibration_and_one_local_model"]
            and not parent["authorization"]["read_protected_test_before_baseline_and_development_outcomes"]
            and not parent["authorization"]["load_model_before_baseline_and_benchmark_lock"]
        ),
        "development_and_protected_identities_match_frozen_V104_metadata_without_test_read": bool(
            config["developmentLanguage"] == language_outcome["development_language"]
            and config["developmentLanguageSha256"] == language_outcome["development_language_sha256"]
            and config["protectedTestLanguage"] == language_outcome["protected_test_language"]
            and config["protectedTestLanguageSha256"] == language_outcome["protected_test_language_sha256"]
        ),
        "V105_interface_artifacts_are_exact_and_complete": bool(
            file_sha256(catalog_path) == parent["visible_catalog_sha256"]
            and file_sha256(hypotheses_path) == parent["safe_hypothesis_universe_sha256"]
            and file_sha256(controls_path) == parent["controlled_insufficient_identifiers_sha256"]
            and len(catalog["scenarios"]) == 3 and len(catalog["intents"]) == 12
            and len(hypotheses["hypotheses"]) == 17
            and controls["role_counts"]["development"] == 64
        ),
        "balanced_hash_only_calibration_evaluation_split_is_frozen": bool(
            split["calibrationCountPerClass"] == 32
            and split["evaluationCountPerClass"] == 32
            and len(split["classes"]) == 4
            and split["splitUsesOnlyFrozenRecordIdentifierAndClass"]
            and not split["manualLanguageInspection"]
        ),
        "deterministic_controls_threshold_grid_costs_and_metrics_are_frozen": bool(
            set(baselines) == {
                "complete_safe_enumeration", "ask_always", "identifier_grammar",
                "character_ngram_retrieval", "oracle",
            }
            and baselines["character_ngram_retrieval"]["knownThresholdGrid"]
            and baselines["character_ngram_retrieval"]["unsupportedThresholdGrid"]
            and config["decisionCosts"]["known"]["wrong_known"] == 10.0
            and config["decisionCosts"]["insufficient"]["abstain"] == 0.0
            and len(config["registeredMetrics"]) >= 10
        ),
        "baseline_integrity_gates_are_nonperformance_and_protected_model_closed": bool(
            baseline_gates["requiredCalibrationRecordCount"] == 128
            and baseline_gates["requiredEvaluationRecordCount"] == 128
            and baseline_gates["requiredOracleExactDecisionAccuracy"] == 1.0
            and baseline_gates["requiredOracleMeanRegret"] == 0.0
            and baseline_gates["maximumProtectedTestLanguageReadCount"] == 0
            and baseline_gates["maximumModelLoadCount"] == 0
            and baseline_gates["maximumModelGenerationCount"] == 0
        ),
        "one_local_27B_4bit_shadow_condition_and_quality_gates_are_frozen": bool(
            model["id"] == "qwen38_27b_4bit"
            and model["revision"] == "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
            and (PROJECT_ROOT / model["localSnapshot"]).is_dir()
            and model["temperature"] == 0.0 and model["samplesPerRecord"] == 1
            and not model["retryOnMalformedOutput"] and not model["thinking"]
            and model["developmentGenerationCountIfAuthorized"] == 192
            and len(model_gates) == 15
        ),
        "deterministic_runtime_versions_match": bool(
            numpy.__version__ == config["deterministicRuntime"]["numpyVersion"]
            and sklearn.__version__ == config["deterministicRuntime"]["scikitLearnVersion"]
            and mlx_lm.__version__ == config["deterministicRuntime"]["mlxLmVersionReservedForFutureModelStage"]
        ),
        "authority_and_access_boundaries_remain_closed": bool(
            authority["allActionsAreCounterfactualShadowActions"]
            and authority["realExecutionCount"] == 0
            and authority["safeHypothesisUniverseMayNotBePruned"]
            and authority["authoritativeCapabilityStatePosteriorAndPolicyAreImmutable"]
            and authority["modelMayNotSelectActionOrExecuteTool"]
            and all(value == 0 for value in exposure.values())
            and not config["decisionRule"]["baselinePassAuthorizesProtectedTestAccess"]
            and not config["decisionRule"]["baselinePassAuthorizesAPITrainingPlanningOrExecution"]
        ),
        "plan_and_locked_code_exist": all(
            path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "106r1-open-world-development-benchmark-design-audit",
        "experiment": "v106r1_open_world_development_benchmark_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_development_deterministic_baselines" if passed else "reject_V106_benchmark",
        "checks": checks,
        "prelock_access": exposure,
        "protected_test_file_open_count": 0,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_interface_outcome": parent_path,
        "technical_outcome": technical_path, "failed_design_audit": failed_audit_path,
        "interface_lock": PROJECT_ROOT / parent["interface_lock"],
        "parent_language_outcome": PROJECT_ROOT / interface_lock["parent_language_outcome"],
        "visible_catalog": catalog_path, "safe_hypothesis_universe": hypotheses_path,
        "controlled_identifiers": controls_path, "plan": plan_path, "protocol": protocol_path,
        "tests": tests_path, "runner": runner_path, "verifier": verifier_path,
        "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "106-open-world-development-benchmark-lock",
        "experiment": "v106_open_world_development_benchmark_lock",
        "config_payload": config,
        "development_language": config["developmentLanguage"],
        "development_language_sha256": config["developmentLanguageSha256"],
        "source_archive": config["sourceArchive"],
        "source_archive_sha256": config["sourceArchiveSha256"],
        "protected_test_identity_only": {
            "path": config["protectedTestLanguage"], "sha256": config["protectedTestLanguageSha256"],
            "file_open_count": 0,
        },
        "authorization": {
            "modify_split_baselines_metrics_costs_model_or_gates": False,
            "automatically_read_development_language_once": True,
            "automatically_read_declared_training_language_once": True,
            "read_protected_test_language": False,
            "manually_inspect_any_utterance": False,
            "run_deterministic_development_baselines_once": True,
            "load_or_run_local_model": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
