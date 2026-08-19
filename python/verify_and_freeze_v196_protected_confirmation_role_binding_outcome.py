#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v196_protected_confirmation_role_binding import audit_binding, build_binding


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v196-protected-confirmation-role-binding-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v196-protected-confirmation-role-binding/binding"
    audit_path = PROJECT_ROOT / "outputs/v196-protected-confirmation-role-binding/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v196-protected-confirmation-role-binding-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v196-protected-confirmation-role-binding-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V196 outcome already verified or frozen")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys
    )
    config = lock["config_payload"]
    rebuilt = build_binding(
        json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text()),
        json.loads((PROJECT_ROOT / lock["contract_catalog"]).read_text()),
        json.loads((PROJECT_ROOT / lock["V183_hidden_identifiability"]).read_text()),
        json.loads((PROJECT_ROOT / lock["V191_hidden_targets"]).read_text()),
        config,
    )
    evaluated = audit_binding(rebuilt, config)
    expected = {
        "remaining-source-census.json": rebuilt["remaining_source_census"],
        "confirmation-identities.json": rebuilt["public_identities"],
        "hidden-targets.json": rebuilt["hidden_targets"],
        "binding-summary.json": rebuilt["summary"],
    }
    artifacts_exact = all(
        (output_root / name).is_file() and json.loads((output_root / name).read_text()) == value
        for name, value in expected.items()
    )
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    expected_decision = (
        config["decisionRule"]["ifEverySourceFreshnessPopulationAndAccessGatePasses"]
        if evaluated["passed"] else config["decisionRule"]["otherwise"]
    )
    result_exact = bool(
        result["passed"] == evaluated["passed"]
        and result["decision"] == expected_decision
        and result["checks"] == evaluated["checks"]
        and result["summary"] == evaluated["summary"]
    )
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "binding_artifacts_reconstruct_exactly": artifacts_exact,
        "result_reconstructs_exactly": result_exact,
        "source_freshness_population_and_access_gates_pass": evaluated["passed"],
        "results_document_exists": results_path.is_file(),
        "protected_language_model_API_authority_and_execution_access_remain_zero": bool(
            evaluated["summary"]["protected_utterance_read_or_emission_count"] == 0
            and evaluated["summary"]["model_load_count"] == 0
            and evaluated["summary"]["API_call_count"] == 0
            and evaluated["summary"]["ontology_registration_count"] == 0
            and evaluated["summary"]["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "196-protected-confirmation-role-binding-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_verified_V196_confirmation_binding" if passed else "freeze_failed_V196_verification",
        "checks": checks,
        "summary": evaluated["summary"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "binding_lock": lock_path,
        "audit": audit_path,
        "remaining_source_census": output_root / "remaining-source-census.json",
        "confirmation_identities": output_root / "confirmation-identities.json",
        "hidden_targets": output_root / "hidden-targets.json",
        "binding_summary": output_root / "binding-summary.json",
        "result": result_path,
        "results_document": results_path,
        "sealed_protected_language": PROJECT_ROOT / lock["sealed_protected_language"],
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "196-protected-confirmation-role-binding-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "decision": "freeze_V196_and_authorize_separate_unchanged_V195_policy_confirmation_preregistration_only",
            "summary": evaluated["summary"],
        },
        "authorization": {
            "preregister_unchanged_V195_policy_confirmation_only": True,
            "open_protected_language_or_run_model_without_separate_lock": False,
            "modify_prompt_model_budget_parser_cost_or_gate": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
