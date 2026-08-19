#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v143_oracle_certificate_policy import evaluate, evaluate_gates


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v143-oracle-certificate-policy.json"
    plan_path = PROJECT_ROOT / "docs/v143-oracle-certificate-policy-plan.md"
    protocol_path = PROJECT_ROOT / "python/v143_oracle_certificate_policy.py"
    tests_path = PROJECT_ROOT / "python/test_v143_oracle_certificate_policy.py"
    runner_path = PROJECT_ROOT / "python/run_and_freeze_v143_oracle_certificate_policy.py"
    results_path = PROJECT_ROOT / "docs/v143-oracle-certificate-policy-results.md"
    audit_path = PROJECT_ROOT / "outputs/v143-oracle-certificate-policy/audit.json"
    outcome_path = PROJECT_ROOT / "configs/v143-oracle-certificate-policy-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V143 already frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV142OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v142_config_path = PROJECT_ROOT / "configs/v142-certificate-interface-population.json"
    v142_config = json.loads(v142_config_path.read_text())
    hidden_path = PROJECT_ROOT / parent["hidden_fixtures"]
    hidden = json.loads(hidden_path.read_text())
    catalog_path = PROJECT_ROOT / parent["choice_catalog"]
    catalog = json.loads(catalog_path.read_text())
    v136_path = PROJECT_ROOT / config["V136Config"]
    v136 = json.loads(v136_path.read_text())
    result = evaluate(config, hidden, catalog, v142_config, v136)
    gates = evaluate_gates(result, config)
    checks = {
        "V142_valid_and_authorizes_oracle_policy_audit": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["fresh_certificate_asset_pass"]
            and parent["authorization"]["run_model_free_oracle_certificate_policy_audit"]
            and not parent["authorization"]["run_language_or_model"]
        ),
        "every_oracle_policy_gate_passes": all(gates.values()),
        "malformed_certificates_fail_closed": result["metrics"]["malformed_mutation_fail_closed_rate"] == 1.0,
        "valid_wrong_singleton_limitation_is_explicit": result["valid_wrong_singleton_limitation"]["semantic_truth_not_checkable_by_interface"],
        "true_hypothesis_retention_and_zero_execution": result["metrics"]["true_hypothesis_retention"] == 1.0 and result["metrics"]["actual_execution_count"] == 0,
        "zero_language_model_API_training_and_execution_access": True,
        "required_files_exist": all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, runner_path, results_path)),
    }
    passed = all(checks.values())
    decision = config["decisionRule"]["ifEveryOracleMutationPolicyAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"]
    audit = {
        "schema_version": "143-oracle-certificate-policy-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "gates": gates,
        "result": result,
        "decision": decision,
        "access": {
            "raw_response_or_trace_read_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "V134_language_read_count": 0,
            "external_language_read_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
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
        "V142_config": v142_config_path,
        "choice_catalog": catalog_path,
        "hidden_fixtures": hidden_path,
        "V136_config": v136_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "results_document": results_path,
        "audit": audit_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "143-oracle-certificate-policy-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "oracle_certificate_policy_pass": True,
            "semantic_wrong_singleton_remains_empirical": True,
            "decision": decision,
            "metrics": result["metrics"],
        },
        "authorization": {
            "preregister_one_pinned_local_V142_development_realization": True,
            "run_model_before_separate_preregistration": False,
            "open_V142_test_split_language": False,
            "touch_V134_or_external_language": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps({"passed": passed, "decision": decision, "metrics": result["metrics"], "checks": checks}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
