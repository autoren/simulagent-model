#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v209_controlled_language_observation_pomdp import audit_oracle, evaluate_oracle
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v209-controlled-language-observation-pomdp-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v209-controlled-language-observation-pomdp/evaluation"
    audit_path = PROJECT_ROOT / "outputs/v209-controlled-language-observation-pomdp/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v209-controlled-language-observation-pomdp-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v209-controlled-language-observation-pomdp-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V209 outcome already frozen")

    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys
    )
    rebuilt = evaluate_oracle(lock["config_payload"])
    rebuilt_audit = audit_oracle(rebuilt, lock["config_payload"])
    summary_exact = json.loads((output_root / "summary.json").read_text()) == rebuilt
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    scientific_pass = rebuilt_audit["scientific_gates_passed"]
    expected_decision = lock["config_payload"]["decisionRule"][
        "ifEveryOracleIntegrityScientificAndAccessGatePasses" if scientific_pass else "otherwise"
    ]
    result_exact = bool(
        result["passed"] == rebuilt_audit["access_gates_passed"]
        and result["scientific_oracle_passed"] == scientific_pass
        and result["checks"] == rebuilt_audit["checks"]
        and result["access_checks"] == rebuilt_audit["access_checks"]
        and result["summary"] == rebuilt
        and result["decision"] == expected_decision
    )
    access = rebuilt["access"]
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "summary_reconstructs_exactly": summary_exact,
        "result_reconstructs_exactly": result_exact,
        "access_audit_passes": rebuilt_audit["access_gates_passed"],
        "results_document_exists": results_path.is_file(),
        "external_language_protected_model_API_training_authority_action_and_execution_remain_zero": all(
            access[key] == 0
            for key in (
                "external_language_record_read_count",
                "raw_model_response_read_count",
                "protected_access_count",
                "model_load_count",
                "model_generation_count",
                "API_call_count",
                "training_run_count",
                "ontology_registration_count",
                "trusted_state_mutation_count",
                "service_call_count",
                "external_side_effect_count",
                "actual_execution_count",
            )
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "209-controlled-language-observation-POMDP-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "scientific_oracle_passed": scientific_pass,
        "decision": "freeze_verified_V209_oracle_result" if passed else "freeze_failed_V209_verification",
        "checks": checks,
        "summary": rebuilt,
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "evaluation_lock": lock_path,
        "audit": audit_path,
        "summary": output_root / "summary.json",
        "result": result_path,
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "209-controlled-language-observation-POMDP-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "scientific_oracle_passed": scientific_pass,
            "decision": expected_decision,
            "summary": rebuilt,
        },
        "authorization": {
            "preregister_fresh_controlled_language_population_design_only": scientific_pass,
            "open_language_population_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
