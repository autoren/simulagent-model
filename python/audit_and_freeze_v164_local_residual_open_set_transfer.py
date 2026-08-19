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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash(
        {key: value for key, value in payload.items() if key != "lock_payload_sha256"}
    ) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v164-local-residual-open-set-transfer.json"
    parent_path = (
        PROJECT_ROOT
        / "configs/v163-deterministic-open-set-transfer-baselines-outcome-lock.json"
    )
    interface_outcome_path = (
        PROJECT_ROOT / "configs/v105-open-world-interface-outcome-lock.json"
    )
    direct_evidence_path = (
        PROJECT_ROOT / "configs/v154r1-outcome-verifier-repair-outcome-lock.json"
    )
    plan_path = PROJECT_ROOT / "docs/v164-local-residual-open-set-transfer-plan.md"
    protocol_path = PROJECT_ROOT / "python/v164_local_residual_open_set_transfer.py"
    tests_path = PROJECT_ROOT / "python/test_v164_local_residual_open_set_transfer.py"
    runner_path = PROJECT_ROOT / "python/run_v164_local_residual_open_set_transfer.py"
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v164_local_residual_open_set_transfer_outcome.py"
    )
    auditor_path = (
        PROJECT_ROOT
        / "python/audit_and_freeze_v164_local_residual_open_set_transfer.py"
    )
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    audit_path = (
        PROJECT_ROOT
        / "outputs/v164-local-residual-open-set-transfer/implementation-audit.json"
    )
    lock_path = (
        PROJECT_ROOT / "configs/v164-local-residual-open-set-transfer-lock.json"
    )
    output_root = (
        PROJECT_ROOT / "outputs/v164-local-residual-open-set-transfer/development"
    )
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V164 implementation is already frozen or evaluated")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["benchmark_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    interface_outcome = json.loads(interface_outcome_path.read_text())
    interface_lock_path = PROJECT_ROOT / interface_outcome["interface_lock"]
    interface_lock = json.loads(interface_lock_path.read_text())
    direct_evidence = json.loads(direct_evidence_path.read_text())
    residual_path = PROJECT_ROOT / parent["model_eligible_residual"]
    residual = json.loads(residual_path.read_text())
    predictions_path = PROJECT_ROOT / parent["baseline_predictions"]
    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_path"])
    condition = config["condition"]
    runtime_versions = {
        package: metadata.version(package)
        for package in ("mlx", "mlx-lm", "huggingface-hub", "transformers")
    }
    present = {row["path"] for row in manifest["files"]}
    parent_summary = parent["outcome"]["development_summary"]
    exposure = config["preLockExposure"]
    authority = config["authorityBoundary"]
    checks = {
        "V163_outcome_is_exact_and_authorizes_one_residual_model_protocol": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["residual_qualified"]
            and parent["authorization"][
                "preregister_one_pinned_local_model_on_frozen_residual_only"
            ]
            and not parent["authorization"][
                "run_model_without_separate_residual_protocol_lock"
            ]
            and not parent["authorization"][
                "read_protected_transfer_before_fresh_development_outcome"
            ]
        ),
        "residual_and_baseline_predictions_match_frozen_V163": bool(
            file_sha256(residual_path) == parent["model_eligible_residual_sha256"]
            and file_sha256(predictions_path) == parent["baseline_predictions_sha256"]
            and residual["payload_sha256"]
            == config["corpus"]["residualManifestPayloadSha256"]
            == parent_summary["residual_summary"]["manifest_payload_sha256"]
            and len(residual["records"])
            == config["corpus"]["modelEligibleResidualCount"]
            == 76
            and not residual["membership_uses_truth_or_language"]
            and config["corpus"]["deterministicNonResidualCount"] == 20
        ),
        "historical_typed_interface_and_complete_universe_are_exact": bool(
            valid_lock(interface_outcome)
            and interface_outcome["outcome"]["passed"]
            and interface_lock["config_payload"]["safeHypothesisUniverse"][
                "totalHypothesisCount"
            ]
            == 17
            and file_sha256(PROJECT_ROOT / parent_lock["visible_catalog"])
            == parent_lock["visible_catalog_sha256"]
            and file_sha256(PROJECT_ROOT / parent_lock["safe_hypothesis_universe"])
            == parent_lock["safe_hypothesis_universe_sha256"]
        ),
        "direct_no_thinking_is_the_single_evidence_based_condition": bool(
            valid_lock(direct_evidence)
            and direct_evidence["outcome"]["direct_summary"]["metrics"][
                "structural_validity"
            ]
            > direct_evidence["outcome"]["bounded_low_reasoning_summary"]["metrics"][
                "structural_validity"
            ]
            and direct_evidence["outcome"]["direct_summary"]["metrics"][
                "query_top1_accuracy"
            ]
            > direct_evidence["outcome"]["bounded_low_reasoning_summary"]["metrics"][
                "query_top1_accuracy"
            ]
            and config["decoding"]["temperature"] == 0.0
            and config["decoding"]["samplesPerRecord"] == 1
            and config["decoding"]["retryCount"] == 0
            and not config["decoding"]["enableThinking"]
            and config["decoding"]["reasoningEffort"] is None
        ),
        "pinned_local_snapshot_manifest_is_exact_and_present": bool(
            file_sha256(manifest_path) == config["modelManifestSha256"]
            and manifest["repository"] == condition["repository"]
            and manifest["revision"] == condition["revision"]
            and manifest["quantization_bits"] == condition["quantizationBits"]
            and manifest["weight_bytes"] == condition["weightBytes"]
            and manifest["model_load_count"] == 0
            and manifest["model_generation_count"] == 0
            and snapshot.is_dir()
            and snapshot.name == condition["revision"]
            and {"config.json", "tokenizer.json", "tokenizer_config.json"} <= present
        ),
        "one_residual_only_no_retry_corpus_and_prompt_are_frozen": bool(
            config["corpus"]["frozenEvaluationCount"] == 96
            and config["corpus"]["totalModelGenerationCount"] == 76
            and config["corpus"]["excludeCalibrationRecords"]
            and config["corpus"]["excludeDeterministicNonResidualRecordsFromModel"]
            and config["corpus"]["excludeProtectedTransferRecords"]
            and not config["corpus"]["manualUtteranceInspection"]
            and config["prompt"]["demonstrationCount"] == 0
            and set(config["prompt"]["hiddenFields"])
            >= {"record_id", "class_label", "truth", "deterministic_predictions"}
        ),
        "quality_access_and_decision_gates_are_noncompensatory": bool(
            len(config["qualityGates"]) == 19
            and config["qualityGates"]["maximumResidualFalseKnownAcceptanceRate"]
            == 0.10
            and config["qualityGates"][
                "minimumCombinedRegretImprovementOverFrozenConsensus"
            ]
            == 0.10
            and config["accessGates"]["requiredResidualFixtureCount"] == 76
            and config["accessGates"]["maximumProtectedLanguageReadCount"] == 0
            and config["accessGates"]["maximumModelLoadCount"] == 1
            and config["accessGates"]["maximumModelGenerationCount"] == 76
            and not config["decisionRule"]["passAuthorizesProtectedAccessImmediately"]
            and not config["decisionRule"][
                "passAuthorizesAPITrainingOntologyPlanningActionOrExecution"
            ]
        ),
        "authority_and_prelock_access_boundaries_remain_closed": bool(
            authority["permanentlyNonAuthoritative"]
            and authority["completeSafeHypothesisUniverseRetained"]
            and authority["authoritativeCapabilityStatePosteriorAndPolicyImmutable"]
            and not authority["modelDefinesOrRegistersCapabilities"]
            and not authority["modelUpdatesPosterior"]
            and not authority["modelSelectsAction"]
            and not authority["modelExecutesTool"]
            and authority["allCostsAndActionsAreCounterfactualShadowQuantities"]
            and authority["realExecutionCount"] == 0
            and all(value == 0 for value in exposure.values())
        ),
        "runner_dependencies_and_runtime_are_present": bool(
            all(
                path.is_file()
                for path in (
                    plan_path,
                    protocol_path,
                    tests_path,
                    runner_path,
                    verifier_path,
                    auditor_path,
                    harness_path,
                )
            )
            and runtime_versions["mlx-lm"] == "0.31.3"
        ),
        "evaluation_output_absent_before_inference_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "164-local-residual-open-set-transfer-implementation-audit",
        "experiment": "v164_local_residual_open_set_transfer_implementation_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_one_local_residual_development_shadow_run"
            if passed
            else "reject_V164_model_inference"
        ),
        "checks": checks,
        "runtime_versions": runtime_versions,
        "prelock_access": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_deterministic_outcome": parent_path,
        "parent_baseline_lock": parent_lock_path,
        "historical_interface_outcome": interface_outcome_path,
        "historical_interface_lock": interface_lock_path,
        "direct_decoding_evidence": direct_evidence_path,
        "visible_catalog": PROJECT_ROOT / parent_lock["visible_catalog"],
        "safe_hypothesis_universe": PROJECT_ROOT
        / parent_lock["safe_hypothesis_universe"],
        "residual_manifest": residual_path,
        "baseline_predictions": predictions_path,
        "model_manifest": manifest_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "census_harness": harness_path,
        "implementation_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "164-local-residual-open-set-transfer-lock",
        "experiment": "v164_local_residual_open_set_transfer_lock",
        "config_payload": config,
        "baseline_config_payload": parent_lock["config_payload"],
        "interface_config_payload": interface_lock["config_payload"],
        "development_language": parent_lock["development_language"],
        "development_language_sha256": parent_lock["development_language_sha256"],
        "protected_language_identity_only": parent_lock[
            "protected_language_identity_only"
        ],
        "runtime_versions": runtime_versions,
        "authorization": {
            "modify_residual_prompt_model_decoding_validation_costs_metrics_gates_or_decision": False,
            "run_one_local_residual_development_shadow_condition_once": True,
            "maximum_model_load_count": 1,
            "maximum_model_generation_count": 76,
            "retry_or_rerun_any_fixture": False,
            "run_model_on_nonresidual_or_missing_control": False,
            "read_protected_transfer_language": False,
            "manually_inspect_source_or_model_language": False,
            "run_API_model_or_train_adapter": False,
            "induce_register_or_execute_capability": False,
            "prune_hypotheses_or_grant_model_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(lock_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(lock_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
