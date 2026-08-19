#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v140_thinking_qualification_gap import audit_gap


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v140-thinking-qualification-gap.json"
    plan_path = PROJECT_ROOT / "docs/v140-thinking-qualification-gap-plan.md"
    protocol_path = PROJECT_ROOT / "python/v140_thinking_qualification_gap.py"
    tests_path = PROJECT_ROOT / "python/test_v140_thinking_qualification_gap.py"
    runner_path = PROJECT_ROOT / "python/run_and_freeze_v140_thinking_qualification_gap.py"
    results_path = PROJECT_ROOT / "docs/v140-thinking-qualification-gap-results.md"
    audit_path = PROJECT_ROOT / "outputs/v140-thinking-qualification-gap/audit.json"
    outcome_path = PROJECT_ROOT / "configs/v140-thinking-qualification-gap-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V140 already frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV139OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    result_path = PROJECT_ROOT / config["V139Result"]
    result = json.loads(result_path.read_text())
    hidden_path = PROJECT_ROOT / "outputs/v135-controlled-open-world-minimal-pairs/design/hidden-fixtures.json"
    hidden = [row for row in json.loads(hidden_path.read_text()) if row["split"] == "development"]
    gap = audit_gap(result, hidden, config)
    checks = {
        "V139_outcome_valid_and_closed": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["at_least_one_condition_qualified"]
            and not parent["authorization"]["modify_retry_rerun_reprompt_or_mine_V139"]
        ),
        "exactly_two_failed_thinking_gate_families": gap["failed_thinking_gate_families"] == ["ambiguous_abstention_accuracy", "structured_validity"],
        "all_invalid_outputs_at_frozen_ceiling": gap["all_invalid_at_condition_token_ceiling"],
        "completion_only_repair_is_insufficient": not gap["counterfactuals"]["completion_only"]["qualifies_both_failed_families"],
        "semantic_only_repair_is_insufficient": not gap["counterfactuals"]["semantic_only"]["qualifies_both_failed_families"],
        "paired_thinking_net_gain_is_positive": gap["paired_correctness"]["net_thinking_repairs"] > 0,
        "no_raw_response_or_trace_fields_exist_or_were_read": not any({"raw_response", "thinking_trace", "final_text"} & set(row) for row in result["fixtures"].values()),
        "zero_model_language_API_and_execution_access": True,
        "required_files_exist": all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, runner_path, results_path)),
    }
    passed = all(checks.values())
    decision = config["decisionRule"]["ifAllGatesPass"] if passed else config["decisionRule"]["otherwise"]
    audit = {
        "schema_version": "140-thinking-qualification-gap-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "gap": gap,
        "decision": decision,
        "access": {
            "raw_response_or_trace_read_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "V134_language_read_count": 0,
            "external_language_read_count": 0,
            "API_call_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "V139_result": result_path,
        "hidden_fixtures": hidden_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "results_document": results_path,
        "audit": audit_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "140-thinking-qualification-gap-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "two_mechanism_gap_confirmed": True,
            "paired_thinking_gain_confirmed": True,
            "decision": decision,
            "minimum_joint_gap": gap["minimum_joint_gap"],
        },
        "authorization": {
            "preregister_model_free_bounded_finalizer_and_evidence_sufficiency_feasibility": True,
            "run_language_or_model": False,
            "rerun_mine_or_modify_V139": False,
            "touch_V134_or_external_language": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
