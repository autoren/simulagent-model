#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v135_controlled_open_world_minimal_pairs import build_catalog
from v136_controlled_clarification_value import evaluate, evaluate_gates


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v136-controlled-clarification-value.json"
    plan_path = PROJECT_ROOT / "docs/v136-controlled-clarification-value-plan.md"
    protocol_path = PROJECT_ROOT / "python/v136_controlled_clarification_value.py"
    tests_path = PROJECT_ROOT / "python/test_v136_controlled_clarification_value.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v136_controlled_clarification_value.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v136_controlled_clarification_value_outcome.py"
    parent_path = PROJECT_ROOT / "configs/v135-controlled-open-world-minimal-pairs-outcome-lock.json"
    baseline_path = PROJECT_ROOT / "configs/v106-open-world-development-benchmark.json"
    result_path = PROJECT_ROOT / "outputs/v136-controlled-clarification-value/model-free/result.json"
    audit_path = PROJECT_ROOT / "outputs/v136-controlled-clarification-value/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v136-controlled-clarification-value-lock.json"
    if any(path.exists() for path in (result_path, audit_path, lock_path)):
        raise RuntimeError("V136 already frozen")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    v135_config = parent["outcome"] and json.loads((PROJECT_ROOT / parent["analysis_lock"]).read_text())["config_payload"]
    catalog = build_catalog(v135_config)
    result = evaluate(config, v135_config, catalog)
    gates = evaluate_gates(result, config)
    baseline = json.loads(baseline_path.read_text())
    parent_auth = parent["authorization"]
    checks = {
        "V135_exact_and_authorized_model_free_successor": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent_auth["run_model_free_sequential_value_audit"]
            and not parent_auth["run_local_or_API_model"]
        ),
        "historical_costs_exact": config["decisionCosts"] == baseline["decisionCosts"],
        "all_value_safety_and_access_gates_pass": all(gates.values()),
        "pass_authorizes_preregistration_not_model_run": bool(
            config["decisionRule"]["passAuthorizesDirectVsThinkingPreregistrationOnly"]
            and not config["decisionRule"]["passAuthorizesImmediateModelRun"]
            and not config["decisionRule"]["passAuthorizesV134LanguageAccess"]
            and not config["decisionRule"]["passAuthorizesAPIInductionTrainingAuthorityActionOrExecution"]
        ),
        "code_complete_before_freeze": all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, auditor_path, verifier_path)),
    }
    passed = all(checks.values())
    decision = config["decisionRule"]["ifEveryValueSafetyAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"]
    audit = {
        "schema_version": "136-controlled-clarification-value-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "gates": gates,
        "decision": decision,
        "summary": {key: value for key, value in result.items() if key not in {"rows", "clear_rows"}},
        "access": {
            "V134_language_read_count": 0,
            "external_language_read_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    write_json(result_path, result)
    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "baseline_costs": baseline_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
        "result": result_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "136-controlled-clarification-value-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "freeze_model_free_value_result": True,
            "preregister_direct_vs_thinking_successor": True,
            "modify_rerun_or_relax_V136": False,
            "run_local_or_API_model": False,
            "touch_V134_language": False,
            "run_induction_training_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
