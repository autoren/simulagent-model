#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v194_deterministic_language_menu_rankers import audit_evaluation, evaluate_rankers
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v194-deterministic-language-menu-rankers-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v194-deterministic-language-menu-rankers/evaluation"
    audit_path = PROJECT_ROOT / "outputs/v194-deterministic-language-menu-rankers/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v194-deterministic-language-menu-rankers-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v194-deterministic-language-menu-rankers-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V194 outcome already verified or frozen")
    dependency_keys = (
        "config",
        "parent_V193_outcome",
        "source_V192_outcome",
        "source_V192_extraction_lock",
        "development_language",
        "hidden_targets",
        "visible_menu",
        "hidden_option_map",
        "primary_prior",
        "fixed_hierarchy_target_costs",
        "protocol",
        "tests",
        "runner",
        "verifier",
        "auditor",
        "design_audit",
    )
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys
    )
    rebuilt = evaluate_rankers(
        json.loads((PROJECT_ROOT / lock["development_language"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text()),
        json.loads((PROJECT_ROOT / lock["visible_menu"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_option_map"]).read_text()),
        json.loads((PROJECT_ROOT / lock["primary_prior"]).read_text()),
        json.loads((PROJECT_ROOT / lock["fixed_hierarchy_target_costs"]).read_text()),
        lock["config_payload"],
    )
    expected = {
        "ranker-results.json": rebuilt["ranker_results"],
        "shadow-predictions.json": rebuilt["predictions"],
        "evaluation-summary.json": rebuilt["summary"],
    }
    artifacts_exact = all(
        (output_root / name).is_file() and json.loads((output_root / name).read_text()) == value
        for name, value in expected.items()
    )
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    audit = audit_evaluation(rebuilt, lock["config_payload"])
    expected_decision = (
        lock["config_payload"]["decisionRule"]["ifEveryIntegritySafetyAndMinimumSignalGatePasses"]
        if audit["passed"]
        else lock["config_payload"]["decisionRule"]["otherwise"]
    )
    result_exact = bool(
        result["passed"] == audit["passed"]
        and result["checks"] == audit["checks"]
        and result["summary"] == audit["summary"]
        and result["decision"] == expected_decision
    )
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "evaluation_artifacts_reconstruct_exactly": artifacts_exact,
        "result_reconstructs_exactly": result_exact,
        "integrity_safety_and_minimum_signal_gates_pass": bool(audit["passed"] and result["passed"]),
        "results_document_exists": results_path.is_file(),
        "protected_model_authority_and_execution_access_remain_zero": bool(
            result["summary"]["protected_language_read_count"] == 0
            and result["summary"]["model_load_count"] == 0
            and result["summary"]["API_call_count"] == 0
            and result["summary"]["ontology_registration_count"] == 0
            and result["summary"]["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "194-deterministic-language-menu-rankers-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "deterministic_evaluation_gates_passed": bool(audit["passed"]),
        "decision": "freeze_verified_V194_controls" if passed else "freeze_failed_V194_verification",
        "checks": checks,
        "summary": result["summary"],
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "evaluation_lock": lock_path,
        "audit": audit_path,
        "ranker_results": output_root / "ranker-results.json",
        "shadow_predictions": output_root / "shadow-predictions.json",
        "evaluation_summary": output_root / "evaluation-summary.json",
        "result": result_path,
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "194-deterministic-language-menu-rankers-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "deterministic_evaluation_gates_passed": True,
            "decision": "freeze_V194_controls_and_allow_bounded_local_model_preregistration_only",
            "summary": result["summary"],
        },
        "authorization": {
            "preregister_one_bounded_local_model_shadow_comparator": True,
            "immediate_model_run_or_API_fallback": False,
            "read_protected_language_or_train": False,
            "register_prune_mutate_call_service_act_or_execute": False,
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
