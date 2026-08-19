#!/usr/bin/env python3
"""Resolve, audit, and freeze the V79 terminal-utility design delta."""
from __future__ import annotations

import copy
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
    parent_lock_path = PROJECT_ROOT / "configs/v78-clarification-outcome-lock.json"
    delta_path = PROJECT_ROOT / "configs/v79-terminal-utility-design.json"
    plan_path = PROJECT_ROOT / "docs/v79-terminal-utility-benchmark-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v79_terminal_utility_design.py"
    audit_path = PROJECT_ROOT / "outputs/v79-structured-llm-interface/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v79-terminal-utility-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V79 terminal-utility design is already frozen")

    parent_lock = json.loads(parent_lock_path.read_text())
    parent_payload = {
        key: value for key, value in parent_lock.items() if key != "lock_payload_sha256"
    }
    parent_result = json.loads((PROJECT_ROOT / parent_lock["result"]).read_text())
    parent_implementation = json.loads(
        (PROJECT_ROOT / parent_lock["implementation_lock"]).read_text()
    )
    parent_config = parent_implementation["design_payload"]
    delta = json.loads(delta_path.read_text())
    inherited = delta["inheritExactlyFromV78"]
    resolved = {
        "schemaVersion": "79-terminal-utility-benchmark-design",
        "experiment": delta["experiment"],
        "purpose": delta["purpose"],
        **{key: copy.deepcopy(parent_config[key]) for key in inherited},
        "terminalUtility": copy.deepcopy(delta["terminalUtility"]),
        "additionalGates": copy.deepcopy(delta["additionalGates"]),
        "claimBoundary": copy.deepcopy(delta["claimBoundary"]),
        "decisionRule": copy.deepcopy(delta["decisionRule"]),
        "stageAuthorization": copy.deepcopy(delta["stageAuthorization"]),
    }
    inherited_exactly = all(resolved[key] == parent_config[key] for key in inherited)
    parent_valid = bool(
        payload_hash(parent_payload) == parent_lock["lock_payload_sha256"]
        and parent_lock["authorization"][
            "preregister_fresh_successor_with_explicit_terminal_unresolved_utility"
        ]
        and not parent_lock["authorization"]["modify_or_rerun_V78"]
        and not parent_lock["authorization"]["access_local_or_API_model"]
        and sum(not value for value in parent_result["gates"].values()) == 1
        and not parent_result["gates"]["unknown_has_safe_unknown_continuation"]
    )
    terminal = delta["terminalUtility"]
    positive = parent_config["positiveRewardProfile"]
    terminal_semantics = bool(
        terminal["activeUnresolvedAtHorizonExhaustion"] == -6.0
        and terminal["terminalStateAtHorizonExhaustion"] == 0.0
        and terminal["activeUnresolvedAtHorizonExhaustion"] < positive["abstain"]
        < positive["safePreview"]
        and terminal["appliesToPlannerAndEveryControl"]
    )
    additional = delta["additionalGates"]
    additional_gates = bool(
        additional["requiredV78PassedGateCount"] == 19
        and additional["requiredV78FailedGateSet"]
        == ["unknown_has_safe_unknown_continuation"]
        and additional["requiredActiveUnresolvedTerminalUtility"] == -6.0
        and additional["requiredTerminalStateTerminalUtility"] == 0.0
        and additional["requireAllInheritedV78Gates"]
        and additional["requireTerminalUtilityReplayAgreement"]
    )
    boundary = delta["claimBoundary"]
    decision = delta["decisionRule"]
    stage = delta["stageAuthorization"]
    firewall = bool(
        boundary["targetedSuccessorToFrozenNegativeResult"]
        and boundary["modelFreeStage"]
        and not boundary["localModelEvidence"]
        and not boundary["APIModelEvidence"]
        and not boundary["humanLanguageEvidence"]
        and not boundary["executionAuthority"]
        and not decision["passAuthorizesLocalModelForwardPassesImmediately"]
        and not decision["passAuthorizesAPIAccess"]
        and not decision["passAuthorizesLoRA"]
        and not decision["passAuthorizesLearnedLikelihoods"]
        and not decision["passAuthorizesExecutionAuthority"]
        and stage["auditAndFreezeDesign"]
        and not stage["implementTerminalUtilityPlanner"]
        and not stage["computePlannerOutcomes"]
        and not stage["accessLocalModel"]
        and not stage["accessAPIModel"]
        and not stage["trainAdapter"]
        and not stage["collectHumanLanguage"]
        and not stage["performRealToolCall"]
        and not stage["performExternalSideEffect"]
    )
    downstream = (
        "python/v79_terminal_utility_planning.py",
        "python/evaluate_v79_terminal_utility_benchmark.py",
        "python/test_v79_terminal_utility_benchmark.py",
        "configs/v79-terminal-utility-implementation-lock.json",
        "outputs/v79-structured-llm-interface/model-free-evaluation",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream)
    checks = {
        "V78_frozen_negative_outcome_authorizes_targeted_successor": parent_valid,
        "all_registered_task_fields_inherited_exactly": inherited_exactly,
        "terminal_unresolved_utility_has_required_semantic_ordering": terminal_semantics,
        "additional_noncompensatory_gates_are_fixed": additional_gates,
        "zero_model_API_human_tool_and_execution_authorization": firewall,
        "implementation_and_outcomes_absent": downstream_absent,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "79-terminal-utility-design-audit",
        "experiment": "v79_terminal_utility_design_audit",
        "passed": passed,
        "decision": (
            "freeze_resolved_design_and_authorize_model_free_implementation_only"
            if passed
            else "reject_v79_terminal_utility_design"
        ),
        "checks": checks,
        "inherited_field_count": len(inherited),
        "access": {
            "V79_policy_value_count": 0,
            "V79_optimal_action_count": 0,
            "model_forward_pass_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "79-terminal-utility-design-lock",
        "experiment": "v79_terminal_utility_design_lock",
        "parent_outcome_lock": str(parent_lock_path.relative_to(PROJECT_ROOT)),
        "parent_outcome_lock_sha256": file_sha256(parent_lock_path),
        "design_delta": str(delta_path.relative_to(PROJECT_ROOT)),
        "design_delta_sha256": file_sha256(delta_path),
        "resolved_config_payload": resolved,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_inherited_V78_fields_or_terminal_utility": False,
            "implement_and_structurally_audit_model_free_terminal_utility": True,
            "compute_V79_policy_values_or_optimal_actions": False,
            "access_local_or_API_model": False,
            "access_human_records_or_real_tools": False,
            "perform_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path), "sha256": file_sha256(lock_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
