#!/usr/bin/env python3
"""Audit and freeze the V83 strict model-free interface design."""
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
    design_path = PROJECT_ROOT / "configs/v83-strict-clarification-interface-design.json"
    v79_path = PROJECT_ROOT / "configs/v79-terminal-utility-outcome-lock.json"
    v82_path = PROJECT_ROOT / "configs/v82-local-clarification-surface-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v83-strict-clarification-interface-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v83_interface_design.py"
    audit_path = PROJECT_ROOT / "outputs/v83-strict-clarification-interface/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v83-strict-clarification-interface-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V83 interface design is already frozen")
    if (PROJECT_ROOT / "outputs/v83-strict-clarification-interface/evaluation").exists():
        raise RuntimeError("V83 outcome exists before design lock")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    design = json.loads(design_path.read_text())
    v79 = json.loads(v79_path.read_text())
    v79_payload = {key: value for key, value in v79.items() if key != "lock_payload_sha256"}
    v82 = json.loads(v82_path.read_text())
    v82_payload = {key: value for key, value in v82.items() if key != "lock_payload_sha256"}
    result = json.loads((PROJECT_ROOT / v79["result"]).read_text())
    all_nodes = [
        node
        for fixture in result["fixtures"].values()
        for node in fixture["exact"]["policy_nodes"]
    ]
    clarification_nodes = [node for node in all_nodes if node["action"].startswith("ask_")]
    rendered_count = len(clarification_nodes) * (
        1 + len(design["enumeration"]["finiteGrammarStyles"])
    )
    checks = {
        "frozen_V79_parent_exact": bool(
            payload_hash(v79_payload) == v79["lock_payload_sha256"]
            and file_sha256(PROJECT_ROOT / v79["result"]) == v79["result_sha256"]
        ),
        "V82_failure_authorizes_only_model_free_integration": bool(
            payload_hash(v82_payload) == v82["lock_payload_sha256"]
            and not v82["outcome"]["passed"]
            and not v82["authorization"]["deploy_local_model_surface_renderer"]
            and v82["authorization"]["use_locked_canonical_renderer"]
            and v82["authorization"]["use_locked_finite_grammar_renderer"]
            and v82["authorization"]["integrate_and_verify_model_free_fail_closed_interface"]
        ),
        "complete_exact_enumeration_counts": bool(
            len(all_nodes) == design["enumeration"]["allV79PolicyNodes"]
            and len(clarification_nodes)
            == design["enumeration"]["reachableV79ClarificationNodes"]
            and rendered_count == design["enumeration"]["requiredRenderedReachableCases"]
        ),
        "strict_choice_fragments_cover_every_clarification_code": bool(
            set(design["strictChoiceFragments"]) == set(design["clarificationCodes"])
            and all(design["strictChoiceFragments"][code] for code in design["clarificationCodes"])
        ),
        "only_canonical_and_finite_grammar_authorized": bool(
            design["authorizedRendererSources"] == ["canonical", "finite_grammar"]
            and set(design["disabledRendererSources"])
            == {"local_model", "API_model", "adapter_model", "untrusted_passthrough"}
        ),
        "noncompensatory_identity_safety_and_zero_access_gates": bool(
            design["gates"]["minimumStrictSurfaceValidityRate"] == 1.0
            and design["gates"]["minimumActionCodePreservationRate"] == 1.0
            and design["gates"]["minimumUnsafeMutationRejectionRate"] == 1.0
            and design["gates"]["maximumPolicyValueAbsoluteError"] <= 1e-12
            and design["gates"]["maximumModelGenerationCount"] == 0
            and design["gates"]["maximumAPICallCount"] == 0
            and design["gates"]["maximumRealToolCallCount"] == 0
        ),
        "design_stage_has_no_model_training_human_tool_or_side_effect_authority": bool(
            design["stageAuthorization"]["auditAndFreezeDesign"]
            and not design["stageAuthorization"]["accessLocalOrAPIModel"]
            and not design["stageAuthorization"]["trainAdapter"]
            and not design["stageAuthorization"]["collectHumanOrOriginalUserLanguage"]
            and not design["stageAuthorization"]["performRealToolCall"]
            and not design["stageAuthorization"]["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "83-strict-clarification-interface-design-audit",
        "experiment": "v83_strict_clarification_interface_design_audit",
        "passed": passed,
        "decision": (
            "freeze_design_and_authorize_model_free_implementation_and_evaluation"
            if passed else "reject_V83_design"
        ),
        "checks": checks,
        "access": {
            "model_load_count": 0, "model_generation_count": 0, "API_call_count": 0,
            "adapter_training_run_count": 0, "human_record_access_count": 0,
            "original_user_language_access_count": 0, "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "83-strict-clarification-interface-design-lock",
        "experiment": "v83_strict_clarification_interface_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)),
        "design_sha256": file_sha256(design_path),
        "config_payload": design,
        "parent_V79_outcome_lock": str(v79_path.relative_to(PROJECT_ROOT)),
        "parent_V79_outcome_lock_sha256": file_sha256(v79_path),
        "parent_V79_result": v79["result"],
        "parent_V79_result_sha256": v79["result_sha256"],
        "surface_boundary_V82_outcome_lock": str(v82_path.relative_to(PROJECT_ROOT)),
        "surface_boundary_V82_outcome_lock_sha256": file_sha256(v82_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_design_sources_enumeration_contract_or_gates": False,
            "implement_and_test_interface": True,
            "evaluate_model_free_integration_once": True,
            "access_local_or_API_model": False,
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
