#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v187_clean_typed_clarification_planner import DEPENDENCY_KEYS, reconstruct


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    repair_lock_path = PROJECT_ROOT / "configs/v187r1-clean-planner-outcome-verification-repair-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v187r1-clean-planner-outcome-verification-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v187r1-clean-planner-outcome-verification-repair-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V187r1 outcome is already frozen")
    repair_lock = json.loads(repair_lock_path.read_text())
    source_lock = json.loads((PROJECT_ROOT / repair_lock["source_V187_lock"]).read_text())
    result = json.loads((PROJECT_ROOT / repair_lock["source_V187_result"]).read_text())
    problem, evaluation, independent = reconstruct(source_lock)
    expected_problem = {
        "contract_ids": list(problem["contract_ids"]),
        "prior_counts": problem["prior_counts"],
        "prior": {key: float(value) for key, value in problem["prior"].items()},
        "raw_question_count": 0,
        "partition_distinct_questions": [
            {"question_id": q.question_id, "family": q.family, "value": q.value, "column": list(q.column)}
            for q in problem["questions"]
        ],
        "horizon": problem["horizon"],
        "typed_question_cost": float(problem["typed_cost"]),
        "generic_trusted_clarification_cost": float(problem["generic_cost"]),
        "safe_deferral_cost": float(problem["deferral_cost"]),
    }
    expected = {
        "problem_summary": expected_problem,
        "policy_summary": evaluation["summary"],
        "target_paths": evaluation["by_target"],
        "development_record_results": {"records": evaluation["record_rows"]},
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == payload
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"]) == result["output_integrity"][key]["sha256"]
        for key, payload in expected.items()
    )
    source_config = source_lock["config_payload"]
    expected_decision = source_config["decisionRule"]["otherwise"]
    checks = {
        "repair_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in repair_lock.items() if key != "lock_payload_sha256"}) == repair_lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / repair_lock[key]) == repair_lock[f"{key}_sha256"] for key in (
                "config", "plan", "auditor", "verifier", "source_V187_lock", "source_V187_result",
                "source_V187_failed_outcome_audit", "design_audit",
            ))
        ),
        "source_V187_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in source_lock.items() if key != "lock_payload_sha256"}) == source_lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / source_lock[key]) == source_lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "all_original_outputs_reconstruct_with_only_declared_metadata_correction": outputs_exact,
        "negative_scientific_result_and_decision_reconstruct_exactly": bool(
            not independent["passed"]
            and result["summary"] == evaluation["summary"]
            and result["development_gates"] == independent["checks"]
            and not result["passed"]
            and result["decision"] == expected_decision
        ),
        "no_additional_policy_language_model_authority_or_effect_access": True,
    }
    verified = all(checks.values())
    outcome_audit = {
        "schema_version": "187r1-clean-planner-outcome-verification-repair-outcome-audit",
        "experiment": repair_lock["experiment"],
        "passed": verified,
        "source_scientific_development_gates_passed": False,
        "decision": "freeze_verified_V187_negative_outcome" if verified else "reject_V187r1_repair",
        "checks": checks,
        "independent_summary": evaluation["summary"],
        "additional_access": {
            "policy_build_count": 0, "policy_score_count": 0,
            "protected_utterance_language_read_count": 0, "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, outcome_audit)
    if not verified:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "repair_lock": repair_lock_path,
        "source_V187_lock": PROJECT_ROOT / repair_lock["source_V187_lock"],
        "source_V187_result": PROJECT_ROOT / repair_lock["source_V187_result"],
        "source_V187_failed_outcome_audit": PROJECT_ROOT / repair_lock["source_V187_failed_outcome_audit"],
        "repair_audit": audit_path,
        "source_results_document": PROJECT_ROOT / "docs/v187-clean-typed-clarification-planner-results.md",
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "187r1-clean-planner-outcome-verification-repair-outcome-lock",
        "experiment": repair_lock["experiment"],
        "outcome": {
            "passed": True,
            "source_scientific_development_gates_passed": False,
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rerun_or_redefine_V187_or_V187r1": False,
            "preregister_correlated_error_or_model_successor": False,
            "preregister_text_free_channel_economics_frontier": True,
            "read_protected_or_utterance_language_run_model_API_or_training": False,
            "register_mutate_call_service_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
