#!/usr/bin/env python3
"""Verify acquired snapshots and freeze V90 before any model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    acquisition_lock_path = PROJECT_ROOT / "configs/v90-capacity-generation-acquisition-lock.json"
    acquisition_result_path = PROJECT_ROOT / "outputs/v90-capacity-generation/model-acquisition/result.json"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v90_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v90-capacity-generation/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v90-capacity-generation-implementation-lock.json"
    evaluation_root = PROJECT_ROOT / "outputs/v90-capacity-generation/evaluation"
    if audit_path.exists() or lock_path.exists() or evaluation_root.exists():
        raise RuntimeError("V90 implementation is already frozen or evaluated")
    acquisition_lock = json.loads(acquisition_lock_path.read_text())
    lock_payload = {key: value for key, value in acquisition_lock.items() if key != "lock_payload_sha256"}
    acquisition_result = json.loads(acquisition_result_path.read_text())
    conditions = acquisition_lock["config_payload"]["modelConditions"]
    manifest_checks: dict[str, bool] = {}
    model_manifests: dict[str, Any] = {}
    for condition in conditions:
        spec = acquisition_result["condition_manifests"].get(condition["id"])
        if not spec:
            manifest_checks[condition["id"]] = False
            continue
        path = PROJECT_ROOT / spec["path"]
        manifest = json.loads(path.read_text())
        snapshot = Path(manifest["snapshot_path"])
        present = {row["path"] for row in manifest["files"]}
        valid = bool(
            file_sha256(path) == spec["sha256"]
            and manifest["repository"] == condition["repository"]
            and manifest["revision"] == condition["revision"]
            and manifest["quantization_bits"] == condition["quantizationBits"]
            and manifest["weight_bytes"] == condition["weightBytes"] == spec["weight_bytes"]
            and snapshot.is_dir()
            and snapshot.name == condition["revision"]
            and {"config.json", "tokenizer.json", "tokenizer_config.json"} <= present
            and manifest["model_load_count"] == 0
            and manifest["model_generation_count"] == 0
        )
        manifest_checks[condition["id"]] = valid
        model_manifests[condition["id"]] = spec
    checks = {
        "acquisition_lock_and_every_frozen_dependency_are_exact": bool(
            payload_hash(lock_payload) == acquisition_lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / acquisition_lock[key]) == acquisition_lock[f"{key}_sha256"] for key in (
                "design_lock", "corpus_seal", "corpus", "protocol", "tests", "downloader", "runner",
                "census_harness", "acquisition_auditor", "implementation_auditor",
            ))
        ),
        "all_four_exact_snapshot_manifests_and_weight_sizes_validate": bool(
            acquisition_result["passed"]
            and len(model_manifests) == 4
            and all(manifest_checks.values())
        ),
        "acquisition_loaded_or_generated_from_no_model": bool(
            acquisition_result["access"]["model_load_count"] == 0
            and acquisition_result["access"]["model_generation_count"] == 0
            and acquisition_result["access"]["LLM_API_call_count"] == 0
        ),
        "runtime_versions_are_pinned_in_audit": True,
        "evaluation_outputs_do_not_exist_before_inference_lock": not evaluation_root.exists(),
    }
    passed = all(checks.values())
    try:
        import importlib.metadata as metadata
        runtime_versions = {
            package: metadata.version(package)
            for package in ("mlx", "mlx-lm", "huggingface-hub", "transformers")
        }
    except Exception as error:  # pragma: no cover - fail closed in audit
        runtime_versions = {"error": str(error)}
        checks["runtime_versions_are_pinned_in_audit"] = False
        passed = False
    audit = {
        "schema_version": "90-capacity-generation-implementation-audit",
        "experiment": "v90_capacity_generation_implementation_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_four_independent_local_shadow_conditions_once_each" if passed else "reject_V90_inference",
        "checks": checks,
        "manifest_checks": manifest_checks,
        "runtime_versions": runtime_versions,
        "access": {
            "model_snapshot_count": len(model_manifests),
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "manual_utterance_inspection_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "90-capacity-generation-implementation-lock",
        "experiment": "v90_capacity_generation_implementation_lock",
        "design_lock": acquisition_lock["design_lock"],
        "design_lock_sha256": acquisition_lock["design_lock_sha256"],
        "config_payload": acquisition_lock["config_payload"],
        "corpus_seal": acquisition_lock["corpus_seal"],
        "corpus_seal_sha256": acquisition_lock["corpus_seal_sha256"],
        "corpus": acquisition_lock["corpus"],
        "corpus_sha256": acquisition_lock["corpus_sha256"],
        "protocol": acquisition_lock["protocol"],
        "protocol_sha256": acquisition_lock["protocol_sha256"],
        "tests": acquisition_lock["tests"],
        "tests_sha256": acquisition_lock["tests_sha256"],
        "runner": acquisition_lock["runner"],
        "runner_sha256": acquisition_lock["runner_sha256"],
        "census_harness": acquisition_lock["census_harness"],
        "census_harness_sha256": acquisition_lock["census_harness_sha256"],
        "acquisition_result": str(acquisition_result_path.relative_to(PROJECT_ROOT)),
        "acquisition_result_sha256": file_sha256(acquisition_result_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "model_manifests": model_manifests,
        "runtime_versions": runtime_versions,
        "authorization": {
            "modify_corpus_models_prompt_protocol_runner_decoding_gates_or_decisions": False,
            "run_each_registered_condition_once": [item["id"] for item in conditions],
            "maximum_model_load_count_per_condition": 1,
            "maximum_model_generation_count_per_condition": 48,
            "rerun_failed_malformed_or_negative_condition": False,
            "deploy_or_execute_any_model_output": False,
            "grant_model_belief_or_action_authority": False,
            "run_API_model_or_train_adapter": False,
            "manually_inspect_source_language": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
