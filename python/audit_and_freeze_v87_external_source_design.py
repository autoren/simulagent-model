#!/usr/bin/env python3
"""Audit and freeze the V87 external human-language source gate."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v87-external-language-source-audit.json"
    parent_path = PROJECT_ROOT / "configs/v86-partial-option-validator-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v87-external-language-source-audit-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v87_external_source_design.py"
    audit_path = PROJECT_ROOT / "outputs/v87-external-language-source-audit/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v87-external-language-source-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V87 source design is already frozen")
    if (PROJECT_ROOT / "outputs/v87-external-language-source-audit/source").exists():
        raise RuntimeError("V87 source payload exists before the source design lock")

    config = json.loads(design_path.read_text())
    parent = json.loads(parent_path.read_text())
    parent_payload = {key: value for key, value in parent.items() if key != "lock_payload_sha256"}
    candidates = config["candidates"]
    gates = config["noncompensatorySourceGates"]
    passing = [item for item in candidates if all(item["sourceGateResults"][gate] for gate in gates)]
    selected = [item for item in candidates if item["selected"]]
    source = config["selectedSource"]
    exposure = config["auditExposure"]
    protocol = config["postLockStructuralInventoryProtocol"]
    stage = config["stageAuthorization"]

    checks = {
        "positive_V86_parent_exact_and_authorizes_external_shadow": bool(
            payload_hash(parent_payload) == parent["lock_payload_sha256"]
            and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_fresh_independently_authored_language_shadow_evaluation"]
            and not parent["authorization"]["access_local_or_API_model_or_train_adapter"]
        ),
        "metadata_only_prelock_exposure_is_explicit_and_zero_payload": bool(
            exposure["dialoguePayloadAccessCount"] == 0
            and exposure["schemaPayloadAccessCount"] == 0
            and exposure["individualUtteranceAccessCount"] == 0
            and exposure["modelLoadCount"] == 0
            and exposure["modelGenerationCount"] == 0
            and exposure["APICallCount"] == 0
            and "dialogue payload bytes" in exposure["forbiddenBeforeDesignLock"]
            and "individual utterances" in exposure["forbiddenBeforeDesignLock"]
        ),
        "every_candidate_has_complete_noncompensatory_gate_vector": bool(
            len(candidates) == 3
            and all(set(item["sourceGateResults"]) == set(gates) for item in candidates)
        ),
        "exactly_one_source_passes_and_is_selected": bool(
            len(passing) == 1
            and len(selected) == 1
            and passing[0]["id"] == selected[0]["id"] == source["candidateId"]
            and source["candidateId"] == "schema_guided_dialogue"
        ),
        "selected_source_revision_license_and_files_are_pinned": bool(
            selected[0]["revision"] == "e852981ae34990f4358979625854259302feaa78"
            and selected[0]["license"] == "CC-BY-SA-4.0"
            and selected[0]["licenseBlobSha1"] == "e100aff6acc25dad98b144f76c4903419c5e99c1"
            and source["maximumAcquisitionBytes"] == sum(item["byteSize"] for item in source["files"])
            and [item["path"] for item in source["files"]]
            == ["dev/schema.json", "dev/dialogues_001.json"]
            and all(len(item["gitBlobSha1"]) == 40 for item in source["files"])
        ),
        "license_attribution_and_sharealike_handling_is_locked": bool(
            all(source["licenseHandling"].values())
        ),
        "structural_inventory_is_presemantic_code_only_and_nonexecuting": bool(
            protocol["verifyBytesByGitBlobSha1BeforeParsing"]
            and protocol["allowOnlyPinnedFiles"]
            and not protocol["manualUtteranceInspectionBeforeSubsetSeal"]
            and protocol["allowCodeOnlyStructuralParsing"]
            and protocol["selectionBeforeLanguageScoring"]
            and not protocol["modelAccessDuringInventory"]
            and not protocol["benchmarkScoringDuringInventory"]
            and not protocol["realServiceCalls"]
            and len(protocol["eligibleRecordRequirements"]) == 6
            and len(protocol["excludedServicePrefixes"]) == 6
        ),
        "design_stage_has_no_payload_model_training_or_execution_authority": bool(
            stage["auditAndFreezeSourceDesign"]
            and not stage["acquirePinnedPayloadFiles"]
            and not stage["parseStructuralInventory"]
            and not stage["inspectUtterancesManually"]
            and not stage["selectBenchmarkSubset"]
            and not stage["accessLocalOrAPIModel"]
            and not stage["trainAdapter"]
            and not stage["performRealServiceCall"]
            and not stage["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "87-external-language-source-design-audit",
        "experiment": "v87_external_language_source_design_audit",
        "passed": passed,
        "decision": (
            "freeze_SGD_source_and_authorize_pinned_code_only_structural_inventory"
            if passed else "reject_V87_source_design_and_defer_external_language_branch"
        ),
        "checks": checks,
        "passing_candidate_ids": [item["id"] for item in passing],
        "selected_candidate_ids": [item["id"] for item in selected],
        "access": {
            "dialogue_payload_access_count": 0,
            "schema_payload_access_count": 0,
            "individual_utterance_access_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0
        }
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "87-external-language-source-design-lock",
        "experiment": "v87_external_language_source_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)),
        "design_sha256": file_sha256(design_path),
        "config_payload": config,
        "parent_V86_outcome_lock": str(parent_path.relative_to(PROJECT_ROOT)),
        "parent_V86_outcome_lock_sha256": file_sha256(parent_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_source_candidates_gates_revision_files_or_protocol": False,
            "acquire_only_pinned_SGD_files": True,
            "parse_code_only_structural_inventory_once": True,
            "manually_inspect_utterance_text_before_subset_lock": False,
            "select_or_score_benchmark_subset": False,
            "access_local_or_API_model": False,
            "train_adapter": False,
            "perform_real_service_call_or_external_side_effect": False
        }
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
