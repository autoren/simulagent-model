#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v154_adaptive_local_question_order import evaluate_condition
from v154r1_outcome_verifier_repair import sole_json_key_type_mismatch


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v154r1-outcome-verifier-repair.json"
    plan_path = PROJECT_ROOT / "docs/v154r1-outcome-verifier-repair-plan.md"
    protocol_path = PROJECT_ROOT / "python/v154r1_outcome_verifier_repair.py"
    tests_path = PROJECT_ROOT / "python/test_v154r1_outcome_verifier_repair.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v154r1_outcome_verifier_repair.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v154r1_outcome_verifier_repair.py"
    audit_path = PROJECT_ROOT / "outputs/v154r1-outcome-verifier-repair/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v154r1-outcome-verifier-repair-lock.json"
    nominal_v154_outcome = PROJECT_ROOT / "configs/v154-adaptive-local-question-order-outcome-lock.json"
    repaired_outcome = PROJECT_ROOT / "configs/v154r1-outcome-verifier-repair-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or repaired_outcome.exists():
        raise RuntimeError("V154r1 already preregistered or frozen")
    if nominal_v154_outcome.exists():
        raise RuntimeError("nominal V154 outcome lock unexpectedly exists")

    config = _load(config_path)
    parent_path = PROJECT_ROOT / config["parentV154AnalysisLock"]
    failed_audit_path = PROJECT_ROOT / config["failedV154OutcomeAudit"]
    result_path = PROJECT_ROOT / config["v154Result"]
    access_path = PROJECT_ROOT / config["v154Access"]
    direct_path = PROJECT_ROOT / config["v154DirectResult"]
    low_path = PROJECT_ROOT / config["v154BoundedLowResult"]
    results_doc_path = PROJECT_ROOT / config["v154ResultsDocument"]
    parent = _load(parent_path)
    failed_audit = _load(failed_audit_path)
    result = _load(result_path)
    access = _load(access_path)
    direct = _load(direct_path)
    low = _load(low_path)
    original_verifier_path = PROJECT_ROOT / parent["verifier"]

    dependencies = [key for key in parent if not key.endswith("_sha256") and f"{key}_sha256" in parent]
    parent_dependencies_exact = all(
        file_sha256(PROJECT_ROOT / parent[key]) == parent[f"{key}_sha256"] for key in dependencies
    )
    v154_config = parent["config_payload"]
    hidden = _load(PROJECT_ROOT / parent["development_hidden_fixtures"])
    answers = _load(PROJECT_ROOT / parent["development_answer_metadata"])
    catalog = _load(PROJECT_ROOT / parent["interaction_catalog"])
    witness = _load(PROJECT_ROOT / parent["witness_config"])
    comparator = _load(PROJECT_ROOT / parent["comparator_config"])
    expected_direct = evaluate_condition(
        direct["fixtures"], hidden, answers, catalog, witness, comparator, v154_config
    )
    expected_low = evaluate_condition(
        low["fixtures"], hidden, answers, catalog, witness, comparator, v154_config
    )

    false_checks = sorted(key for key, value in failed_audit["checks"].items() if not value)
    prohibited_access = (
        "closed_answer_model_generation_count", "evaluation_fixture_model_generation_count",
        "retry_count", "manual_raw_response_inspection_count", "persisted_raw_response_count",
        "API_call_count", "training_run_count", "real_service_call_count",
        "external_side_effect_count", "actual_execution_count",
    )
    checks = {
        "parent_V154_analysis_lock_and_every_dependency_unchanged": bool(
            valid_lock(parent) and parent_dependencies_exact
        ),
        "original_failed_audit_preserved_and_only_summary_checks_failed": bool(
            not failed_audit["passed"]
            and false_checks == sorted(config["diagnosis"]["expectedFailedChecks"])
            and all(
                value for key, value in failed_audit["checks"].items()
                if key not in config["diagnosis"]["expectedFailedChecks"]
            )
        ),
        "direct_mismatch_is_only_JSON_rank_count_key_type": sole_json_key_type_mismatch(
            expected_direct, result["direct_summary"]
        ),
        "bounded_low_mismatch_is_only_JSON_rank_count_key_type": sole_json_key_type_mismatch(
            expected_low, result["bounded_low_reasoning_summary"]
        ),
        "scientific_decision_and_selection_remain_exact": bool(
            result["selected_condition"] == config["frozenOutcome"]["selectedCondition"]
            and result["decision"] == config["frozenOutcome"]["decision"]
            and not config["frozenOutcome"]["developmentQualified"]
            and not expected_direct["qualified"]
            and not expected_low["qualified"]
        ),
        "original_verifier_is_locked_and_unmodified": bool(
            original_verifier_path == PROJECT_ROOT / "python/verify_and_freeze_v154_adaptive_local_question_order_outcome.py"
            and file_sha256(original_verifier_path) == parent["verifier_sha256"]
        ),
        "original_model_access_record_is_complete_and_fail_closed": bool(
            access["model_load_count"] == 1
            and access["tokenizer_load_count"] == 1
            and access["total_generation_count"] == 288
            and all(access[key] == 0 for key in prohibited_access)
            and all(result["access_gates"].values())
        ),
        "repair_scope_has_zero_model_language_evaluation_or_external_access": bool(
            all(value == 0 for value in config["accessGates"].values())
            and not config["authorization"]["rerunModelOrTokenizer"]
            and not config["authorization"]["inspectRawLanguage"]
            and not config["authorization"]["openEvaluationSplit"]
            and not config["authorization"]["runAPITrainingServicesSideEffectsOrExecution"]
        ),
        "repair_cannot_change_metrics_gates_decision_or_original_records": bool(
            config["authorization"]["writeRepairedTechnicalOutcomeAuditAndLock"]
            and not config["authorization"]["modifyOriginalV154VerifierOrFailedAudit"]
            and not config["authorization"]["changeMetricsGatesDecisionOrClaims"]
            and not config["authorization"]["retryRerunRepromptTuneOrCalibrate"]
        ),
        "required_files_exist": all(
            path.is_file() for path in (
                config_path, plan_path, protocol_path, tests_path, auditor_path, verifier_path,
                parent_path, failed_audit_path, result_path, access_path, direct_path, low_path,
                results_doc_path, original_verifier_path,
            )
        ),
        "nominal_V154_outcome_lock_absent": not nominal_v154_outcome.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "154r1-outcome-verifier-repair-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "diagnosis": {
            "false_original_checks": false_checks,
            "direct_raw_equal": expected_direct == result["direct_summary"],
            "bounded_low_raw_equal": expected_low == result["bounded_low_reasoning_summary"],
            "direct_canonical_equal": True,
            "bounded_low_canonical_equal": True,
        },
        "repair_access": {key: 0 for key in config["accessGates"]},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    paths = {
        "config": config_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
        "parent_analysis_lock": parent_path,
        "failed_V154_outcome_audit": failed_audit_path,
        "V154_result": result_path,
        "V154_access": access_path,
        "V154_direct_result": direct_path,
        "V154_bounded_low_result": low_path,
        "V154_results_document": results_doc_path,
        "original_V154_verifier": original_verifier_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "154r1-outcome-verifier-repair-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": config["authorization"],
    }
    for key, path in paths.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
