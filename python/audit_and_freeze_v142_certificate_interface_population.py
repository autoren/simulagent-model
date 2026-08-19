#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v142_certificate_interface_population import audit_population, build_population


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v142-certificate-interface-population.json"
    plan_path = PROJECT_ROOT / "docs/v142-certificate-interface-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v142_certificate_interface_population.py"
    tests_path = PROJECT_ROOT / "python/test_v142_certificate_interface_population.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v142_certificate_interface_population.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v142_certificate_interface_population_outcome.py"
    results_path = PROJECT_ROOT / "docs/v142-certificate-interface-population-results.md"
    output_dir = PROJECT_ROOT / "outputs/v142-certificate-interface-population/design"
    audit_path = PROJECT_ROOT / "outputs/v142-certificate-interface-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v142-certificate-interface-population-lock.json"
    if output_dir.exists() or audit_path.exists() or lock_path.exists():
        raise RuntimeError("V142 already frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV141OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v135_public_path = PROJECT_ROOT / "outputs/v135-controlled-open-world-minimal-pairs/design/public-fixtures.json"
    v135_public = json.loads(v135_public_path.read_text())
    population = build_population(config)
    result = audit_population(population, config, v135_public)
    checks = {
        "V141_valid_and_authorizes_fresh_interface_population_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["robust_two_stage_envelope_feasible"]
            and parent["authorization"]["design_fresh_bounded_finalizer_evidence_certificate_interface_and_population"]
            and not parent["authorization"]["run_language_or_model"]
        ),
        "population_and_interface_audit_pass": result["passed"],
        "deterministic_finalizer_always_valid": result["deterministic_finalizer_validity"] == 1.0,
        "no_exact_V135_conversation_overlap": result["exact_conversation_overlap_with_V135_count"] == 0,
        "true_hypothesis_retention_and_zero_execution": result["true_hypothesis_retention"] == 1.0 and result["actual_execution_count"] == 0,
        "zero_external_model_or_execution_access": all(config["gates"][key] == 0 for key in ("maximumV134LanguageReadCount", "maximumExternalLanguageReadCount", "maximumModelLoadCount", "maximumModelGenerationCount", "maximumAPICallCount", "maximumTrainingRunCount", "maximumActualExecutionCount")),
        "required_files_exist": all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, auditor_path, verifier_path, results_path)),
    }
    passed = all(checks.values())
    decision = config["decisionRule"]["ifEveryInterfacePopulationAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"]
    output_dir.mkdir(parents=True, exist_ok=False)
    assets = {
        "choice_catalog": output_dir / "choice-catalog.json",
        "public_fixtures": output_dir / "public-fixtures.json",
        "hidden_fixtures": output_dir / "hidden-fixtures.json",
        "population_summary": output_dir / "population-summary.json",
    }
    for key, path in assets.items():
        write_json(path, population[key])
    audit = {
        "schema_version": "142-certificate-interface-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "result": result,
        "decision": decision,
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
    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "V135_public_fixtures": v135_public_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "verifier": verifier_path,
        "results_document": results_path,
        "design_audit": audit_path,
        **assets,
    }
    lock: dict[str, Any] = {
        "schema_version": "142-certificate-interface-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_model_free_oracle_policy_audit": True,
            "modify_regenerate_relabel_or_open_hidden_population": False,
            "run_language_or_model": False,
            "touch_V134_or_external_language": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps({"passed": passed, "decision": decision, "summary": result["summary"], "checks": checks}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
