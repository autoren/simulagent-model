#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v208_external_behavioral_abstention_source_census import audit_census, evaluate_census
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v208-external-behavioral-abstention-source-census-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v208-external-behavioral-abstention-source-census/evaluation"
    audit_path = PROJECT_ROOT / "outputs/v208-external-behavioral-abstention-source-census/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v208-external-behavioral-abstention-source-census-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v208-external-behavioral-abstention-source-census-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V208 outcome exists")

    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys
    )
    stored_summary = json.loads((output_root / "summary.json").read_text())
    revisions = {
        record["candidate_id"]: {"code": record["code"]["commit"], "dataset": record["dataset"]["commit"]}
        for record in stored_summary["records"]
    }
    rebuilt = evaluate_census(lock["config_payload"], pinned_revisions=revisions)
    rebuilt_audit = audit_census(rebuilt, lock["config_payload"])
    summary_exact = rebuilt == stored_summary
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    scientific_pass = rebuilt["scientific_feasibility_passed"]
    expected_decision = lock["config_payload"]["decisionRule"][
        "ifAtLeastOneCandidatePassesEveryGate" if scientific_pass else "otherwise"
    ]
    result_exact = bool(
        result["passed"] == rebuilt_audit["passed"]
        and result["scientific_feasibility_passed"] == scientific_pass
        and result["checks"] == rebuilt_audit["checks"]
        and result["summary"] == rebuilt
        and result["decision"] == expected_decision
    )
    access = rebuilt["access"]
    checks = {
        "design_lock_and_dependencies_exact": dependencies_exact,
        "pinned_metadata_reconstructs_exactly": summary_exact,
        "result_reconstructs_exactly": result_exact,
        "integrity_and_access_audit_passes": rebuilt_audit["passed"],
        "results_document_exists": results_path.is_file(),
        "blob_task_language_model_API_tool_service_authority_action_and_execution_zero": all(
            access[key] == 0
            for key in (
                "github_README_or_other_blob_body_read_count",
                "github_license_blob_body_read_count",
                "huggingFace_tree_or_payload_read_count",
                "task_instruction_example_dialogue_rationale_response_read_count",
                "model_or_policy_evaluation_count",
                "API_call_count",
                "training_run_count",
                "tool_or_service_call_count",
                "external_side_effect_count",
                "actual_execution_count",
            )
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "208-external-behavioral-abstention-source-census-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "scientific_feasibility_passed": scientific_pass,
        "decision": "freeze_verified_V208_source_census" if passed else "freeze_failed_V208_verification",
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
        "schema_version": "208-external-behavioral-abstention-source-census-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "scientific_feasibility_passed": scientific_pass,
            "decision": expected_decision,
            "summary": rebuilt,
        },
        "authorization": {
            "preregister_exact_identifier_selection_and_text_extraction_only": scientific_pass,
            "park_external_behavioral_abstention_if_negative": not scientific_pass,
            "task_text_model_API_training_tool_service_registration_authority_action_or_execution": False,
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
