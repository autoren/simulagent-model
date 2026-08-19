#!/usr/bin/env python3
"""Audit and freeze the untouched V91 SGD source extension before payload access."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v91-rank-only-source.json"
    source_authority_path = PROJECT_ROOT / "configs/v87-external-language-source-outcome-lock.json"
    parent_model_path = PROJECT_ROOT / "configs/v90-capacity-generation-outcome-lock.json"
    previous_source_path = PROJECT_ROOT / "configs/v90-capacity-generation-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v91-rank-only-plan.md"
    module_path = PROJECT_ROOT / "python/v87_external_source_inventory.py"
    runner_path = PROJECT_ROOT / "python/run_v91_source_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v91_source_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v91_source.py"
    audit_path = PROJECT_ROOT / "outputs/v91-rank-only/source-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v91-rank-only-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v91-rank-only/source"
    if audit_path.exists() or lock_path.exists() or source_root.exists():
        raise RuntimeError("V91 source design is already frozen or materialized")

    config = json.loads(config_path.read_text())
    source_authority = json.loads(source_authority_path.read_text())
    parent_model = json.loads(parent_model_path.read_text())
    previous_source = json.loads(previous_source_path.read_text())
    source_authority_payload = {
        key: value
        for key, value in source_authority.items()
        if key != "lock_payload_sha256"
    }
    parent_model_payload = {
        key: value
        for key, value in parent_model.items()
        if key != "lock_payload_sha256"
    }
    previous_source_payload = {
        key: value
        for key, value in previous_source.items()
        if key != "lock_payload_sha256"
    }
    exposure = config["preLockExposure"]
    shard = config["newDialogueShard"]
    schema = config["schemaDependency"]
    schema_path = PROJECT_ROOT / schema["localPath"]
    checks = {
        "positive_source_authority_is_exact": bool(
            payload_hash(source_authority_payload)
            == source_authority["lock_payload_sha256"]
            and source_authority["outcome"]["passed"]
        ),
        "V90_model_free_decision_is_exact_and_rank_only_compatible": bool(
            payload_hash(parent_model_payload) == parent_model["lock_payload_sha256"]
            and parent_model["outcome"]["decision"]
            == "retain_model_free_authoritative_boundary"
            and parent_model["authorization"][
                "retain_qwen35_4b_only_as_frozen_shadow_baseline"
            ]
            and not parent_model["authorization"]["grant_any_model_belief_or_action_authority"]
        ),
        "previous_source_outcome_is_exact": bool(
            payload_hash(previous_source_payload)
            == previous_source["lock_payload_sha256"]
            and previous_source["outcome"]["passed"]
        ),
        "same_repository_revision_and_license_are_frozen": bool(
            config["repository"]
            == "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue"
            and config["revision"]
            == "e852981ae34990f4358979625854259302feaa78"
            and config["license"] == "CC-BY-SA-4.0"
        ),
        "existing_schema_dependency_is_exact": bool(
            schema["path"] == "dev/schema.json"
            and schema["gitBlobSha1"]
            == source_authority["source_files"]["dev/schema.json"]["git_blob_sha1"]
            and schema["localSha256"]
            == source_authority["source_files"]["dev/schema.json"]["local_sha256"]
            and schema_path.is_file()
            and file_sha256(schema_path) == schema["localSha256"]
        ),
        "new_shard_metadata_is_exact_and_unmaterialized": bool(
            shard["path"] == "dev/dialogues_003.json"
            and shard["byteSize"] == 1965028
            and shard["gitBlobSha1"] == "6abbd79ddd2c58c386ca9cfb748bb55a4efd7be5"
            and config["revision"] in shard["rawUrl"]
            and shard["path"] not in source_authority["source_files"]
            and shard["gitBlobSha1"] != previous_source["source_file_git_blob_sha1"]
            and not source_root.exists()
        ),
        "zero_payload_language_model_or_API_access_before_lock": all(
            exposure[key] == 0
            for key in (
                "dialoguePayloadAccessCount",
                "individualUtteranceAccessCount",
                "manualUtteranceInspectionCount",
                "newModelWeightDownloadCount",
                "modelLoadCount",
                "modelGenerationCount",
                "LLMAPICallCount",
            )
        ),
        "inventory_is_text_free_and_fail_closed": bool(
            config["inventoryProtocol"]["verifyByteSizeAndGitBlobBeforeParsing"]
            and config["inventoryProtocol"]["reusePinnedV87SchemaOnly"]
            and not config["inventoryProtocol"]["manualUtteranceInspection"]
            and not config["inventoryProtocol"]["emitLanguageOrSlotValues"]
        ),
        "source_gates_are_noncompensatory_and_model_free": bool(
            config["sourceGates"]["minimumEligibleRecordCount"] >= 64
            and config["sourceGates"]["minimumEligibleActiveRecordCount"] >= 32
            and config["sourceGates"]["minimumEligibleNoneRecordCount"] >= 32
            and all(
                config["sourceGates"][key] == 0
                for key in (
                    "maximumManualUtteranceInspectionCount",
                    "maximumNewModelWeightDownloadCount",
                    "maximumModelLoadCount",
                    "maximumModelGenerationCount",
                    "maximumLLMAPICallCount",
                    "maximumRealServiceCallCount",
                    "maximumExternalSideEffectCount",
                )
            )
        ),
        "rank_only_plan_locks_complete_enumeration_before_source_access": bool(
            plan_path.is_file()
            and "cannot prune the search space" in plan_path.read_text()
            and "Full enumeration" in plan_path.read_text()
        ),
        "locked_source_code_and_plan_exist": all(
            path.is_file()
            for path in (
                plan_path,
                module_path,
                runner_path,
                verifier_path,
                auditor_path,
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "91-rank-only-source-design-audit",
        "experiment": "v91_rank_only_source_design_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_one_pinned_text_free_source_inventory"
            if passed
            else "reject_V91_source_extension"
        ),
        "checks": checks,
        "access": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "91-rank-only-source-lock",
        "experiment": "v91_rank_only_source_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "source_authority_lock": str(source_authority_path.relative_to(PROJECT_ROOT)),
        "source_authority_lock_sha256": file_sha256(source_authority_path),
        "parent_model_decision_lock": str(parent_model_path.relative_to(PROJECT_ROOT)),
        "parent_model_decision_lock_sha256": file_sha256(parent_model_path),
        "previous_source_outcome_lock": str(previous_source_path.relative_to(PROJECT_ROOT)),
        "previous_source_outcome_lock_sha256": file_sha256(previous_source_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "inventory_module": str(module_path.relative_to(PROJECT_ROOT)),
        "inventory_module_sha256": file_sha256(module_path),
        "runner": str(runner_path.relative_to(PROJECT_ROOT)),
        "runner_sha256": file_sha256(runner_path),
        "verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "verifier_sha256": file_sha256(verifier_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_source_metadata_code_or_gates": False,
            "download_and_inventory_pinned_shard_once": True,
            "print_or_manually_inspect_utterances": False,
            "select_rank_only_population": False,
            "load_local_or_API_model": False,
            "train_adapter": False,
            "grant_model_pruning_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False
        },
    }
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
