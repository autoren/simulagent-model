#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v197_protected_confirmation_language_projection import audit_projection, build_projection


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v197-protected-confirmation-language-projection-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v197-protected-confirmation-language-projection/projection"
    audit_path = PROJECT_ROOT / "outputs/v197-protected-confirmation-language-projection/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v197-protected-confirmation-language-projection-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v197-protected-confirmation-language-projection-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V197 outcome already frozen")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys
    )
    config = lock["config_payload"]
    rebuilt = build_projection(
        json.loads((PROJECT_ROOT / lock["sealed_protected_language"]).read_text()),
        json.loads((PROJECT_ROOT / lock["confirmation_identities"]).read_text()), config,
    )
    evaluated = audit_projection(rebuilt, config)
    language_path = output_root / "confirmation-language.json"
    summary_path = output_root / "projection-summary.json"
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    expected_decision = (
        config["decisionRule"]["ifEveryProjectionSeparationAndAccessGatePasses"]
        if evaluated["passed"] else config["decisionRule"]["otherwise"]
    )
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "language_projection_reconstructs_exactly": json.loads(language_path.read_text()) == rebuilt["language"],
        "projection_summary_reconstructs_exactly": json.loads(summary_path.read_text()) == rebuilt["summary"],
        "result_reconstructs_exactly": bool(
            result["passed"] == evaluated["passed"] and result["decision"] == expected_decision
            and result["checks"] == evaluated["checks"] and result["summary"] == evaluated["summary"]
        ),
        "projection_gates_pass": evaluated["passed"],
        "results_document_exists": results_path.is_file(),
        "model_API_authority_and_execution_access_remain_zero": bool(
            evaluated["summary"]["model_generation_count"] == 0
            and evaluated["summary"]["API_call_count"] == 0
            and evaluated["summary"]["ontology_registration_count"] == 0
            and evaluated["summary"]["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "197-protected-confirmation-language-projection-outcome-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_verified_V197_projection" if passed else "freeze_failed_V197_verification",
        "checks": checks, "summary": evaluated["summary"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "projection_lock": lock_path, "audit": audit_path, "confirmation_language": language_path,
        "projection_summary": summary_path, "result": result_path, "results_document": results_path,
        "hidden_targets": PROJECT_ROOT / lock["hidden_targets"], "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "197-protected-confirmation-language-projection-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {"passed": True, "decision": "freeze_V197_and_authorize_separate_unchanged_V195_policy_confirmation_preregistration_only", "summary": evaluated["summary"]},
        "authorization": {
            "preregister_unchanged_V195_policy_confirmation_only": True,
            "run_model_without_separate_lock": False,
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
