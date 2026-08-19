#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v198_protected_language_menu_ranker_confirmation import (
    evaluate_char_last, evaluate_confirmation, evaluate_confirmation_access,
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v198-protected-language-menu-ranker-confirmation-lock.json"
    lock = json.loads(lock_path.read_text())
    output_dir = PROJECT_ROOT / "outputs/v198-protected-language-menu-ranker-confirmation/model-realization"
    audit_path = PROJECT_ROOT / "outputs/v198-protected-language-menu-ranker-confirmation/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v198-protected-language-menu-ranker-confirmation-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v198-protected-language-menu-ranker-confirmation-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V198 outcome already frozen")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys
    )
    config = lock["config_payload"]
    language = json.loads((PROJECT_ROOT / lock["confirmation_language"]).read_text())
    hidden = json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text())
    menu = json.loads((PROJECT_ROOT / lock["visible_menu"]).read_text())
    option_map = json.loads((PROJECT_ROOT / lock["hidden_option_map"]).read_text())
    prior = json.loads((PROJECT_ROOT / lock["primary_prior"]).read_text())
    fixed = json.loads((PROJECT_ROOT / lock["fixed_hierarchy_target_costs"]).read_text())
    development = json.loads((PROJECT_ROOT / lock["source_V195_outcome"]).read_text())["outcome"]["summary"]
    char = evaluate_char_last(language, hidden, menu, option_map, prior, fixed, config)
    access = json.loads((output_dir / "access.json").read_text())
    raw_paths = sorted((output_dir / "census/raw-fixtures").glob("*.json"))
    completed = {}
    for path in raw_paths:
        row = json.loads(path.read_text()); completed[row["record_id"]] = row
    rebuilt = evaluate_confirmation(completed, language, hidden, option_map, prior, fixed, char, access, development, config)
    access_checks = evaluate_confirmation_access(access, config); access_pass = all(access_checks.values())
    qualified = bool(rebuilt["summary"]["qualified"] and access_pass)
    decision = config["decisionRule"]["ifEveryQualificationAndAccessGatePasses"] if qualified else config["decisionRule"]["otherwise"]
    census = json.loads((output_dir / "census/result.json").read_text())
    expected_result = {
        "schema_version": "198-protected-language-menu-ranker-confirmation-result", "experiment": config["experiment"],
        "completed": True, "qualification_gates_passed": rebuilt["summary"]["qualified"],
        "access_gates_passed": access_pass, "qualified": qualified, "decision": decision,
        "qualification_gates": rebuilt["summary"]["qualification_gates"], "access_gates": access_checks,
        "summary": rebuilt["summary"], "census_passed": census["passed"], "claim_boundary": config["claimBoundary"],
    }
    result = json.loads((output_dir / "result.json").read_text())
    char_artifact = json.loads((output_dir / "CHAR_LAST-results.json").read_text())
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "all_and_only_113_observed_fixture_artifacts_exist": len(raw_paths) == 113 and len(completed) == 113,
        "CHAR_LAST_comparator_reconstructs_exactly": char_artifact == {"ranker_results": char["ranker_results"], "summary": char["summary"]},
        "evaluation_summary_reconstructs_exactly": json.loads((output_dir / "evaluation-summary.json").read_text()) == rebuilt["summary"],
        "scored_records_reconstruct_exactly": json.loads((output_dir / "scored-records.json").read_text()) == rebuilt["scored_records"],
        "result_reconstructs_exactly": result == expected_result,
        "access_gates_pass": access_pass,
        "raw_model_text_is_not_persisted": bool(
            access["persisted_raw_response_count"] == 0
            and all("raw_response" not in row and "reasoning_response" not in row for row in completed.values())
        ),
        "API_training_authority_action_and_execution_remain_zero": bool(
            access["API_call_count"] == 0 and access["training_run_count"] == 0
            and access["ontology_registration_count"] == 0 and access["trusted_state_mutation_count"] == 0
            and access["real_service_call_count"] == 0 and access["external_side_effect_count"] == 0
            and access["actual_execution_count"] == 0
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "198-protected-language-menu-ranker-confirmation-outcome-audit",
        "experiment": config["experiment"], "passed": passed,
        "scientific_confirmation_gates_passed": qualified, "access_gates_passed": access_pass,
        "decision": "freeze_verified_V198_confirmation" if passed else "freeze_failed_V198_verification",
        "checks": checks, "summary": rebuilt["summary"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {
        "confirmation_lock": lock_path, "audit": audit_path, "access": output_dir / "access.json",
        "census_result": output_dir / "census/result.json", "CHAR_LAST_results": output_dir / "CHAR_LAST-results.json",
        "CHAR_LAST_predictions": output_dir / "CHAR_LAST-predictions.json", "evaluation_summary": output_dir / "evaluation-summary.json",
        "scored_records": output_dir / "scored-records.json", "result": output_dir / "result.json",
        "results_document": results_path, "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "198-protected-language-menu-ranker-confirmation-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {"passed": True, "scientific_confirmation_gates_passed": qualified, "decision": decision, "summary": rebuilt["summary"]},
        "authorization": {
            "update_roadmap_and_preregister_separate_model_free_robustness_design": qualified,
            "run_API_additional_model_or_new_language_condition_without_separate_lock": False,
            "ontology_promotion_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
