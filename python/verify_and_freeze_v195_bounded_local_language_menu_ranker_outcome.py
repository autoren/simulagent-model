#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v195_bounded_local_language_menu_ranker import evaluate_access_gates, evaluate_model


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v195-bounded-local-language-menu-ranker-lock.json"
    lock = json.loads(lock_path.read_text())
    output_dir = PROJECT_ROOT / "outputs/v195-bounded-local-language-menu-ranker/model-realization"
    audit_path = PROJECT_ROOT / "outputs/v195-bounded-local-language-menu-ranker/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v195-bounded-local-language-menu-ranker-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v195-bounded-local-language-menu-ranker-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V195 outcome already verified or frozen")

    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(
        (PROJECT_ROOT / lock[key]).is_file()
        and file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
        for key in dependency_keys
    )
    config = lock["config_payload"]
    language = json.loads((PROJECT_ROOT / lock["development_language"]).read_text())
    hidden_targets = json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text())
    option_map = json.loads((PROJECT_ROOT / lock["hidden_option_map"]).read_text())
    prior = json.loads((PROJECT_ROOT / lock["primary_prior"]).read_text())
    fixed_costs = json.loads((PROJECT_ROOT / lock["fixed_hierarchy_target_costs"]).read_text())
    deterministic = json.loads((PROJECT_ROOT / lock["deterministic_ranker_results"]).read_text())
    access = json.loads((output_dir / "access.json").read_text())
    raw_paths = sorted((output_dir / "census/raw-fixtures").glob("*.json"))
    completed = {}
    for path in raw_paths:
        row = json.loads(path.read_text())
        completed[row["record_id"]] = row
    rebuilt = evaluate_model(
        completed, language, hidden_targets, option_map, prior, fixed_costs,
        deterministic, access, config,
    )
    summary_path = output_dir / "evaluation-summary.json"
    scored_path = output_dir / "scored-records.json"
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text())
    access_checks = evaluate_access_gates(access, config)
    access_pass = all(access_checks.values())
    qualified = bool(rebuilt["summary"]["qualified"] and access_pass)
    expected_decision = (
        config["decisionRule"]["ifEveryQualificationAndAccessGatePasses"]
        if qualified else config["decisionRule"]["otherwise"]
    )
    census_result = json.loads((output_dir / "census/result.json").read_text())
    expected_result = {
        "schema_version": "195-bounded-local-language-menu-ranker-result",
        "experiment": config["experiment"],
        "completed": True,
        "qualification_gates_passed": rebuilt["summary"]["qualified"],
        "access_gates_passed": access_pass,
        "qualified": qualified,
        "decision": expected_decision,
        "qualification_gates": rebuilt["summary"]["qualification_gates"],
        "access_gates": access_checks,
        "summary": rebuilt["summary"],
        "census_passed": census_result["passed"],
        "claim_boundary": config["claimBoundary"],
    }
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "all_and_only_84_observed_fixture_artifacts_exist": bool(
            len(raw_paths) == config["population"]["requiredObservedGenerationCount"]
            and len(completed) == len(raw_paths)
        ),
        "evaluation_summary_reconstructs_exactly": bool(
            summary_path.is_file() and json.loads(summary_path.read_text()) == rebuilt["summary"]
        ),
        "scored_records_reconstruct_exactly": bool(
            scored_path.is_file() and json.loads(scored_path.read_text()) == rebuilt["scored_records"]
        ),
        "result_reconstructs_exactly": result == expected_result,
        "access_gates_pass": access_pass,
        "raw_model_text_is_not_persisted": bool(
            access["persisted_raw_response_count"] == 0
            and all("raw_response" not in row and "reasoning_response" not in row for row in completed.values())
        ),
        "API_protected_training_authority_action_and_execution_remain_zero": bool(
            access["API_call_count"] == 0
            and access["protected_language_read_count"] == 0
            and access["training_run_count"] == 0
            and access["ontology_registration_count"] == 0
            and access["trusted_state_mutation_count"] == 0
            and access["real_service_call_count"] == 0
            and access["external_side_effect_count"] == 0
            and access["actual_execution_count"] == 0
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "195-bounded-local-language-menu-ranker-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_qualification_gates_passed": rebuilt["summary"]["qualified"],
        "access_gates_passed": access_pass,
        "decision": "freeze_verified_V195_result" if passed else "freeze_failed_V195_verification",
        "checks": checks,
        "summary": rebuilt["summary"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "experiment_lock": lock_path,
        "audit": audit_path,
        "access": output_dir / "access.json",
        "census_result": output_dir / "census/result.json",
        "evaluation_summary": summary_path,
        "scored_records": scored_path,
        "result": result_path,
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "195-bounded-local-language-menu-ranker-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_qualification_gates_passed": qualified,
            "decision": expected_decision,
            "summary": rebuilt["summary"],
        },
        "authorization": {
            "preregister_separate_confirmation_design_only": qualified,
            "run_confirmation_API_additional_model_or_protected_access_without_new_lock": False,
            "register_prune_mutate_call_service_act_or_execute": False,
            "update_roadmap_after_frozen_negative_or_positive_result": True,
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
