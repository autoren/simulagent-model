#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v142_certificate_interface_population import audit_population, build_population


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v142-certificate-interface-population-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v142-certificate-interface-population-outcome-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v142-certificate-interface-population/outcome-audit.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v142_certificate_interface_population_outcome.py"
    if outcome_path.exists() or audit_path.exists():
        raise RuntimeError("V142 outcome already frozen")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependency_exact = all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
    lock_exact = payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
    expected = build_population(config)
    stored = {key: json.loads((PROJECT_ROOT / lock[key]).read_text()) for key in ("choice_catalog", "public_fixtures", "hidden_fixtures", "population_summary")}
    v135_public = json.loads((PROJECT_ROOT / lock["V135_public_fixtures"]).read_text())
    result = audit_population(stored, config, v135_public)
    checks = {
        "lock_and_dependencies_exact": lock_exact and dependency_exact,
        "stored_assets_equal_deterministic_rebuild": all(stored[key] == expected[key] for key in stored),
        "population_and_interface_audit_pass": result["passed"],
        "deterministic_finalizer_validity": result["deterministic_finalizer_validity"] == 1.0,
        "true_hypothesis_retention_and_zero_execution": result["true_hypothesis_retention"] == 1.0 and result["actual_execution_count"] == 0,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "142-certificate-interface-population-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "result": result,
        "decision": config["decisionRule"]["ifEveryInterfacePopulationAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    paths = {
        "analysis_lock": lock_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "choice_catalog": PROJECT_ROOT / lock["choice_catalog"],
        "public_fixtures": PROJECT_ROOT / lock["public_fixtures"],
        "hidden_fixtures": PROJECT_ROOT / lock["hidden_fixtures"],
        "population_summary": PROJECT_ROOT / lock["population_summary"],
        "results_document": PROJECT_ROOT / lock["results_document"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "142-certificate-interface-population-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "fresh_certificate_asset_pass": True,
            "deterministic_finalizer_validity": 1.0,
            "true_hypothesis_retention": 1.0,
            "actual_execution_count": 0,
            "decision": audit["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "run_model_free_oracle_certificate_policy_audit": True,
            "run_language_or_model": False,
            "modify_regenerate_relabel_or_open_hidden_population": False,
            "touch_V134_or_external_language": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
