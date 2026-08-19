#!/usr/bin/env python3
"""Audit and freeze the non-authoritative V82 clarification-surface design."""
from __future__ import annotations

from collections import Counter
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
    design_path = PROJECT_ROOT / "configs/v82-local-clarification-surface-design.json"
    predecessor_path = PROJECT_ROOT / "configs/v81-factorized-local-candidate-outcome-lock.json"
    v79_lock_path = PROJECT_ROOT / "configs/v79-terminal-utility-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v82-local-clarification-surface-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v82_surface_design.py"
    audit_path = PROJECT_ROOT / "outputs/v82-local-clarification-surface/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v82-local-clarification-surface-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V82 design is already frozen")
    if (PROJECT_ROOT / "outputs/v82-local-clarification-surface/evaluation").exists():
        raise RuntimeError("V82 outcome exists before design lock")
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    design = json.loads(design_path.read_text())
    predecessor = json.loads(predecessor_path.read_text())
    predecessor_payload = {
        key: value for key, value in predecessor.items() if key != "lock_payload_sha256"
    }
    v79_lock = json.loads(v79_lock_path.read_text())
    v79_payload = {
        key: value for key, value in v79_lock.items() if key != "lock_payload_sha256"
    }
    v79_result_path = PROJECT_ROOT / v79_lock["result"]
    v79_result = json.loads(v79_result_path.read_text())
    reachable = [
        node
        for fixture in v79_result["fixtures"].values()
        for node in fixture["exact"]["policy_nodes"]
        if node["action"].startswith("ask_")
    ]
    reachable_counts = Counter(node["action"] for node in reachable)
    complete_reachable_counts = {
        code: int(reachable_counts.get(code, 0))
        for code in design["clarificationCodesInRequiredOrder"]
    }
    records = design["records"]
    record_counts = Counter(record["clarificationCode"] for record in records)
    anchors = design["lexicalAnchors"]
    canonical = design["canonicalSurfaces"]
    canonical_exact = bool(
        all(text.endswith("?") for text in canonical.values())
        and all(
            canonical["ask_operation"].count(anchor) == 1
            for anchor in anchors["operation"]
        )
        and all(anchor not in canonical["ask_operation"] for anchor in anchors["recipient"])
        and all(
            canonical["ask_recipient"].count(anchor) == 1
            for anchor in anchors["recipient"]
        )
        and all(anchor not in canonical["ask_recipient"] for anchor in anchors["operation"])
        and all(
            canonical["ask_full_details"].count(anchor) == 1
            for anchor in anchors["operation"] + anchors["recipient"]
        )
    )
    checks = {
        "V81_closure_authorizes_only_materially_different_non_authoritative_role": bool(
            payload_hash(predecessor_payload) == predecessor["lock_payload_sha256"]
            and not predecessor["authorization"]["continue_local_model_candidate_integration"]
            and predecessor["authorization"]["design_materially_different_non_authoritative_LLM_role"]
        ),
        "frozen_V79_parent_and_reachable_action_census_exact": bool(
            payload_hash(v79_payload) == v79_lock["lock_payload_sha256"]
            and file_sha256(v79_result_path) == v79_lock["result_sha256"]
            and len(reachable) == design["parentDecisionCore"]["reachableClarificationNodeCount"]
            and complete_reachable_counts
            == design["parentDecisionCore"]["reachableActionCounts"]
        ),
        "model_role_cannot_receive_language_or_change_decision_state": bool(
            "{instruction}" not in design["userPromptTemplate"]
            and "{clarificationCode}" in design["userPromptTemplate"]
            and not design["stageAuthorization"]["accessOriginalUserLanguage"]
            and not design["decisionRule"]["passAuthorizesCandidateGeneration"]
            and not design["decisionRule"]["passAuthorizesBeliefAssignment"]
            and not design["decisionRule"]["passAuthorizesActionSelection"]
        ),
        "canonical_surfaces_satisfy_exact_anchor_partition": canonical_exact,
        "fresh_complete_unique_balanced_population": bool(
            len(records) == 24
            and len({record["id"] for record in records}) == 24
            and len({(record["clarificationCode"], record["styleHint"]) for record in records}) == 24
            and dict(record_counts) == design["gates"]["requiredCodeCounts"]
        ),
        "local_frozen_deterministic_no_retry_configuration": bool(
            design["model"]["provider"] == "local_mlx"
            and design["model"]["frozen"]
            and not design["model"]["APIRequired"]
            and design["model"]["adapterPath"] is None
            and design["decoding"]["temperature"] == 0.0
            and design["decoding"]["samplesPerRecord"] == 1
            and not design["decoding"]["retryOnMalformedOutput"]
        ),
        "fail_closed_and_noncompensatory_safety_gates": bool(
            design["outputContract"]["invalidOutputHandling"].startswith("discard complete")
            and design["gates"]["minimumFinalSemanticValidityRate"] == 1.0
            and design["gates"]["minimumFinalActionCodePreservationRate"] == 1.0
            and design["gates"]["minimumUnsafeMutationRejectionRate"] == 1.0
            and design["gates"]["minimumReachableV79ClarificationActionInvarianceRate"] == 1.0
            and design["gates"]["maximumV79PolicyValueAbsoluteError"] <= 1e-12
        ),
        "design_stage_has_zero_model_API_training_human_language_tool_or_side_effect_authority": bool(
            design["stageAuthorization"]["auditAndFreezeProtocol"]
            and not design["stageAuthorization"]["runLocalModel"]
            and not design["stageAuthorization"]["runAPIModel"]
            and not design["stageAuthorization"]["trainAdapter"]
            and not design["stageAuthorization"]["collectHumanLanguage"]
            and not design["stageAuthorization"]["performRealToolCall"]
            and not design["stageAuthorization"]["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "82-local-clarification-surface-design-audit",
        "experiment": "v82_local_clarification_surface_design_audit",
        "passed": passed,
        "decision": (
            "freeze_design_and_authorize_corpus_and_runner_implementation_only"
            if passed else "reject_V82_design"
        ),
        "checks": checks,
        "access": {
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "original_user_language_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0
        }
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "82-local-clarification-surface-design-lock",
        "experiment": "v82_local_clarification_surface_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)),
        "design_sha256": file_sha256(design_path),
        "config_payload": design,
        "predecessor_outcome_lock": str(predecessor_path.relative_to(PROJECT_ROOT)),
        "predecessor_outcome_lock_sha256": file_sha256(predecessor_path),
        "parent_V79_outcome_lock": str(v79_lock_path.relative_to(PROJECT_ROOT)),
        "parent_V79_outcome_lock_sha256": file_sha256(v79_lock_path),
        "parent_V79_result": str(v79_result_path.relative_to(PROJECT_ROOT)),
        "parent_V79_result_sha256": file_sha256(v79_result_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_design_prompt_records_model_decoding_contract_or_gates": False,
            "construct_and_seal_corpus": True,
            "implement_and_audit_runner": True,
            "run_local_model": False,
            "run_API_model": False,
            "train_adapter": False,
            "collect_human_or_original_user_language": False,
            "perform_real_tool_call_or_external_side_effect": False
        }
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
