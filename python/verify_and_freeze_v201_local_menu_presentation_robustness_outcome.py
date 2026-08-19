#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v201_local_menu_presentation_robustness import evaluate_access_gates, evaluate_model
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v201-local-menu-presentation-robustness-lock.json"
    lock = json.loads(lock_path.read_text()); output_root = PROJECT_ROOT / "outputs/v201-local-menu-presentation-robustness/model-realization"
    audit_path = PROJECT_ROOT / "outputs/v201-local-menu-presentation-robustness/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v201-local-menu-presentation-robustness-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v201-local-menu-presentation-robustness-results.md"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V201 outcome already verified or frozen")
    dependency_keys = tuple(key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock)
    dependencies_exact = valid_lock(lock) and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
    census_path = output_root / "census/result.json"; census = json.loads(census_path.read_text()); completed = census["fixtures"]
    raw_paths = sorted((output_root / "census/raw-fixtures").glob("*.json"))
    normalized_fixture_artifacts_exact = bool(
        len(completed) == 168 and len(raw_paths) == 168
        and {json.loads(path.read_text())["name"] for path in raw_paths} == set(completed)
        and all(json.loads(path.read_text()) == completed[json.loads(path.read_text())["name"]] for path in raw_paths)
    )
    no_raw_text_persisted = all(
        not ({"raw_response", "reasoning_text", "reasoning_response"} & set(row)) and not row["raw_response_persisted"]
        for row in completed.values()
    )
    config = lock["config_payload"]; access = json.loads((output_root / "access.json").read_text())
    evaluation = evaluate_model(
        completed,
        *[json.loads((PROJECT_ROOT / lock[key]).read_text()) for key in (
            "development_language", "hidden_targets", "hidden_variant_maps", "canonical_hidden_option_map",
            "canonical_model_census", "transformed_CHAR_LAST_summary", "primary_prior", "fixed_hierarchy_target_costs",
        )], access, config,
    )
    summary_exact = json.loads((output_root / "evaluation-summary.json").read_text()) == evaluation["summary"]
    scored_exact = json.loads((output_root / "scored-records.json").read_text()) == evaluation["scored_records"]
    access_checks = evaluate_access_gates(access, config); access_pass = all(access_checks.values())
    qualified = bool(evaluation["summary"]["qualified"] and access_pass)
    expected_decision = config["decisionRule"]["ifEveryPerVariantQualificationAndAccessGatePasses" if qualified else "otherwise"]
    result_path = output_root / "result.json"; result = json.loads(result_path.read_text())
    result_exact = bool(
        result["qualified"] == qualified and result["qualification_gates_passed"] == evaluation["summary"]["qualified"]
        and result["access_gates_passed"] == access_pass and result["access_gates"] == access_checks
        and result["summary"] == evaluation["summary"] and result["decision"] == expected_decision
        and result["census_passed"] == census["passed"]
    )
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "all_and_only_168_normalized_fixture_artifacts_exist": normalized_fixture_artifacts_exact,
        "raw_model_text_is_not_persisted": no_raw_text_persisted,
        "evaluation_summary_reconstructs_exactly": summary_exact,
        "scored_records_reconstruct_exactly": scored_exact,
        "result_reconstructs_exactly": result_exact,
        "access_gates_pass": access_pass,
        "results_document_exists": results_path.is_file(),
        "protected_API_training_authority_action_and_execution_remain_zero": bool(
            access["protected_language_read_count"] == 0 and access["API_call_count"] == 0
            and access["training_run_count"] == 0 and access["ontology_registration_count"] == 0
            and access["trusted_state_mutation_count"] == 0 and access["real_service_call_count"] == 0
            and access["external_side_effect_count"] == 0 and access["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "201-local-menu-presentation-robustness-outcome-audit", "experiment": lock["experiment"],
        "passed": passed, "scientific_qualification_gates_passed": evaluation["summary"]["qualified"],
        "access_gates_passed": access_pass, "decision": "freeze_verified_V201_robustness_result" if passed else "freeze_failed_V201_verification",
        "checks": checks, "summary": evaluation["summary"],
    }
    write_json(audit_path, outcome_audit)
    if not passed: print(json.dumps(outcome_audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {
        "experiment_lock": lock_path, "audit": audit_path, "census_result": census_path,
        "evaluation_summary": output_root / "evaluation-summary.json", "scored_records": output_root / "scored-records.json",
        "access": output_root / "access.json", "result": result_path, "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "201-local-menu-presentation-robustness-outcome-lock", "experiment": lock["experiment"],
        "outcome": {"passed": True, "scientific_qualification_gates_passed": evaluation["summary"]["qualified"], "decision": expected_decision, "summary": evaluation["summary"]},
        "authorization": {
            "preregister_separate_paired_protected_robustness_design_only": bool(qualified),
            "run_protected_without_separate_lock_or_use_API_additional_model_synthetic_language": False,
            "registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
