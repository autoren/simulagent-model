#!/usr/bin/env python3
"""Audit and freeze the V85 offline local adversarial-generator design."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v85-local-adversarial-generator-design.json"
    parent_path = PROJECT_ROOT / "configs/v84-schema-grounded-shadow-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v85-local-adversarial-generator-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v85_adversarial_design.py"
    audit_path = PROJECT_ROOT / "outputs/v85-local-adversarial-generator/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v85-local-adversarial-generator-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V85 adversarial design is already frozen")
    if (PROJECT_ROOT / "outputs/v85-local-adversarial-generator/evaluation").exists():
        raise RuntimeError("V85 outcome exists before design lock")
    config = json.loads(design_path.read_text())
    parent = json.loads(parent_path.read_text())
    parent_payload = {key: value for key, value in parent.items() if key != "lock_payload_sha256"}
    records = config["records"]
    schema_counts = Counter(row["schemaId"] for row in records)
    targets = {(row["schemaId"], row["kind"], row["slotId"]) for row in records}
    target_profile_counts = Counter((row["schemaId"], row["kind"], row["slotId"]) for row in records)
    checks = {
        "positive_V84_parent_exact_and_authorizes_only_preregistration": bool(
            payload_hash(parent_payload) == parent["lock_payload_sha256"]
            and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_offline_non_deployable_local_adversarial_generator_test"]
            and not parent["authorization"]["access_local_model_before_successor_lock"]
            and not parent["authorization"]["deploy_model_or_untrusted_generated_surface"]
        ),
        "complete_balanced_record_population": bool(
            len(records) == config["gates"]["requiredRecordCount"] == 24
            and len(schema_counts) == config["gates"]["requiredSchemaCount"] == 4
            and set(schema_counts.values()) == {6}
            and len(targets) == config["gates"]["requiredTypedTargetCount"] == 12
            and set(target_profile_counts.values()) == {2}
            and all({row["profile"] for row in records if (row["schemaId"], row["kind"], row["slotId"]) == target} == set(config["profilesInOrder"]) for target in targets)
        ),
        "pinned_local_model_and_deterministic_no_retry_decoding": bool(
            config["model"]["provider"] == "local_mlx"
            and config["model"]["revision"] == "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
            and config["model"]["adapterPath"] is None
            and config["decoding"]["temperature"] == 0.0
            and not config["decoding"]["retryOnMalformedOutput"]
            and not config["outputContract"]["retryOnMalformedOutput"]
        ),
        "permanent_non_deployable_provenance_is_noncompensatory": bool(
            config["permanentProvenance"]["source"] == "local_model_adversarial"
            and not config["permanentProvenance"]["deployable"]
            and not config["permanentProvenance"]["mayEnterCurrentV84Suite"]
            and not config["permanentProvenance"]["maySelectSchemaBeliefActionOrTool"]
            and config["gates"]["minimumPermanentNonDeployableRate"] == 1.0
        ),
        "bounded_single_local_run_and_zero_external_access_gates": bool(
            config["gates"]["maximumModelLoadCount"] == 1
            and config["gates"]["maximumModelGenerationCount"] == 24
            and all(config["gates"][key] == 0 for key in (
                "maximumAPICallCount", "maximumAdapterTrainingRunCount",
                "maximumHumanRecordAccessCount", "maximumOriginalUserLanguageAccessCount",
                "maximumRealToolCallCount", "maximumExternalSideEffectCount"
            ))
        ),
        "design_stage_has_no_model_human_tool_or_side_effect_authority": bool(
            config["stageAuthorization"]["auditAndFreezeDesign"]
            and not config["stageAuthorization"]["runLocalModel"]
            and not config["stageAuthorization"]["runAPIModel"]
            and not config["stageAuthorization"]["trainAdapter"]
            and not config["stageAuthorization"]["collectHumanOrOriginalUserLanguage"]
            and not config["stageAuthorization"]["performRealToolCall"]
            and not config["stageAuthorization"]["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "85-local-adversarial-generator-design-audit",
        "experiment": "v85_local_adversarial_generator_design_audit",
        "passed": passed,
        "decision": "freeze_design_and_authorize_corpus_and_runner_implementation" if passed else "reject_V85_design",
        "checks": checks,
        "access": {
            "model_load_count": 0, "model_generation_count": 0,
            "API_call_count": 0, "adapter_training_run_count": 0,
            "human_record_access_count": 0, "original_user_language_access_count": 0,
            "real_tool_call_count": 0, "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    lock = {
        "schema_version": "85-local-adversarial-generator-design-lock",
        "experiment": "v85_local_adversarial_generator_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)),
        "design_sha256": file_sha256(design_path),
        "config_payload": config,
        "parent_V84_outcome_lock": str(parent_path.relative_to(PROJECT_ROOT)),
        "parent_V84_outcome_lock_sha256": file_sha256(parent_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_prompt_records_model_decoding_or_gates": False,
            "construct_and_seal_corpus": True,
            "implement_and_audit_runner": True,
            "run_local_model": False,
            "run_API_model": False,
            "train_adapter": False,
            "collect_human_or_original_user_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
