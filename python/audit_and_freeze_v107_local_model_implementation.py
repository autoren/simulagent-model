#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v107-open-world-local-model.json"
    parent_path = PROJECT_ROOT / "configs/v106-open-world-development-benchmark-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v107-open-world-local-model-plan.md"
    protocol_path = PROJECT_ROOT / "python/v107_open_world_local_model.py"
    tests_path = PROJECT_ROOT / "python/test_v107_open_world_local_model.py"
    runner_path = PROJECT_ROOT / "python/run_v107_open_world_local_model.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v107_local_model_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v107_local_model_implementation.py"
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    audit_path = PROJECT_ROOT / "outputs/v107-open-world-local-model/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v107-open-world-local-model-implementation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v107-open-world-local-model/development-evaluation"
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V107 implementation is already frozen or evaluated")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    baseline_lock = json.loads((PROJECT_ROOT / parent["benchmark_lock"]).read_text())
    interface_outcome = json.loads((PROJECT_ROOT / baseline_lock["parent_interface_outcome"]).read_text())
    interface_lock = json.loads((PROJECT_ROOT / interface_outcome["interface_lock"]).read_text())
    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_path"])
    condition = config["condition"]
    baseline_model = parent["outcome"]["future_local_model_condition"]
    baseline_gates = parent["outcome"]["future_model_development_gates"]
    runtime_versions = {
        package: metadata.version(package)
        for package in ("mlx", "mlx-lm", "huggingface-hub", "transformers")
    }
    present = {row["path"] for row in manifest["files"]}
    checks = {
        "V106_outcome_is_exact_and_authorizes_one_development_model_run": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["authorization"]["audit_and_run_one_pinned_local_model_on_development_evaluation_and_missing_controls"]
            and parent["authorization"]["model_condition_count"] == 1
            and parent["authorization"]["model_generation_limit"] == 192
            and not parent["authorization"]["read_protected_test_before_model_development_outcome"]
            and not parent["authorization"]["combine_small_and_large_models"]
        ),
        "V107_model_and_quality_gates_exactly_match_V106_preregistration": bool(
            condition["id"] == baseline_model["id"]
            and condition["repository"] == baseline_model["repository"]
            and condition["revision"] == baseline_model["revision"]
            and condition["generationLimit"] == baseline_model["developmentGenerationCountIfAuthorized"]
            and config["developmentGates"] == baseline_gates
        ),
        "pinned_local_snapshot_manifest_is_exact_and_present": bool(
            file_sha256(manifest_path) == config["modelManifestSha256"]
            and manifest["condition_id"] == condition["id"]
            and manifest["repository"] == condition["repository"]
            and manifest["revision"] == condition["revision"]
            and manifest["quantization_bits"] == condition["quantizationBits"]
            and manifest["weight_bytes"] == condition["weightBytes"]
            and manifest["model_load_count"] == 0 and manifest["model_generation_count"] == 0
            and snapshot.is_dir() and snapshot.name == condition["revision"]
            and {"config.json", "tokenizer.json", "tokenizer_config.json"} <= present
        ),
        "exact_128_observed_plus_64_missing_no_retry_corpus_is_frozen": bool(
            config["corpus"]["observedDevelopmentEvaluationCount"] == 128
            and config["corpus"]["controlledMissingObservationCount"] == 64
            and config["corpus"]["totalGenerationCount"] == 192
            and config["corpus"]["excludeDevelopmentCalibrationRecords"]
            and config["corpus"]["excludeProtectedTestRecords"]
            and config["prompt"]["demonstrationCount"] == 0
            and config["prompt"]["missingConditionMayNotIncludeSourceUtterance"]
            and config["decoding"]["temperature"] == 0.0
            and not config["decoding"]["retryOnMalformedOutput"]
            and not config["decoding"]["enableThinking"]
        ),
        "invalid_fallback_raw_confidence_metrics_and_noncompensatory_gates_are_frozen": bool(
            config["validation"]["useExactV105ResponseValidator"]
            and config["validation"]["invalidOutputMapsToV105ZeroConfidenceAbstain"]
            and config["metricDefinitions"]["confidenceCalibration"].startswith("raw model confidence")
            and len(config["developmentGates"]) == 15
            and config["developmentGates"]["maximumRegretAboveBestNonOracleDeterministicBaseline"] == 0.25
        ),
        "model_has_no_capability_belief_action_tool_or_execution_authority": bool(
            config["authorityBoundary"]["permanentlyNonAuthoritative"]
            and config["authorityBoundary"]["completeSafeHypothesisUniverseRetained"]
            and not config["authorityBoundary"]["modelDefinesOrPrunesCapabilities"]
            and not config["authorityBoundary"]["modelUpdatesPosterior"]
            and not config["authorityBoundary"]["modelSelectsAction"]
            and not config["authorityBoundary"]["modelExecutesTool"]
            and config["authorityBoundary"]["realExecutionCount"] == 0
        ),
        "development_inputs_and_V105_interface_are_frozen_without_protected_access": bool(
            file_sha256(PROJECT_ROOT / baseline_lock["visible_catalog"]) == baseline_lock["visible_catalog_sha256"]
            and file_sha256(PROJECT_ROOT / baseline_lock["controlled_identifiers"]) == baseline_lock["controlled_identifiers_sha256"]
            and interface_lock["config_payload"]["safeHypothesisUniverse"]["totalHypothesisCount"] == 17
            and baseline_lock["protected_test_identity_only"]["file_open_count"] == 0
        ),
        "runner_dependencies_and_runtime_versions_are_frozen": bool(
            all(path.is_file() for path in (
                plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path, harness_path,
            )) and runtime_versions["mlx-lm"] == "0.31.3"
        ),
        "evaluation_output_absent_before_inference_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "107-open-world-local-model-implementation-audit",
        "experiment": "v107_open_world_local_model_implementation_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_local_development_shadow_run" if passed else "reject_V107_model_inference",
        "checks": checks,
        "runtime_versions": runtime_versions,
        "access": {
            "development_language_read_count": 0, "protected_test_language_read_count": 0,
            "manual_utterance_inspection_count": 0, "model_load_count": 0,
            "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_baseline_outcome": parent_path,
        "baseline_lock": PROJECT_ROOT / parent["benchmark_lock"],
        "interface_outcome": PROJECT_ROOT / baseline_lock["parent_interface_outcome"],
        "interface_lock": PROJECT_ROOT / interface_outcome["interface_lock"],
        "visible_catalog": PROJECT_ROOT / baseline_lock["visible_catalog"],
        "controlled_identifiers": PROJECT_ROOT / baseline_lock["controlled_identifiers"],
        "model_manifest": manifest_path, "plan": plan_path, "protocol": protocol_path,
        "tests": tests_path, "runner": runner_path, "verifier": verifier_path,
        "auditor": auditor_path, "census_harness": harness_path, "implementation_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "107-open-world-local-model-implementation-lock",
        "experiment": "v107_open_world_local_model_implementation_lock",
        "config_payload": config,
        "baseline_config_payload": baseline_lock["config_payload"],
        "interface_config_payload": interface_lock["config_payload"],
        "development_language": baseline_lock["development_language"],
        "development_language_sha256": baseline_lock["development_language_sha256"],
        "protected_test_identity_only": baseline_lock["protected_test_identity_only"],
        "runtime_versions": runtime_versions,
        "authorization": {
            "modify_corpus_prompt_model_decoding_validation_metrics_costs_gates_or_decisions": False,
            "run_one_local_development_shadow_condition_once": True,
            "maximum_model_load_count": 1, "maximum_model_generation_count": 192,
            "retry_failed_malformed_or_negative_fixture": False,
            "read_protected_test_language": False,
            "manually_inspect_source_or_model_language": False,
            "run_API_model_or_train_adapter": False,
            "prune_safe_hypotheses_or_grant_model_belief_action_or_execution_authority": False,
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
