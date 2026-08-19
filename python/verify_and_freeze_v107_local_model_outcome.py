#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from run_v107_open_world_local_model import payload_hash, prepare_fixture_rows
from v105_open_world_interface import validate_response
from v107_open_world_local_model import (
    aggregate_model_fixtures, evaluate_model_gates, quality_gate_pass,
)


def main() -> None:
    implementation_lock_path = PROJECT_ROOT / "configs/v107-open-world-local-model-implementation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v107-open-world-local-model/development-evaluation/result.json"
    access_path = PROJECT_ROOT / "outputs/v107-open-world-local-model/development-evaluation/access.json"
    doc_path = PROJECT_ROOT / "docs/v107-open-world-local-model-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v107_local_model_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v107-open-world-local-model/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v107-open-world-local-model-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V107 model outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V107 result before freezing")
    lock = json.loads(implementation_lock_path.read_text())
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    config = lock["config_payload"]
    baseline_config = lock["baseline_config_payload"]
    interface_config = lock["interface_config_payload"]
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    _, evaluation_records, controlled_ids = prepare_fixture_rows(lock)
    fixtures = result["fixtures"]
    response_reparse_exact = True
    response_hashes_exact = True
    authority_exact = True
    for fixture in fixtures.values():
        parsed, valid, reason = validate_response(fixture["raw_response"], catalog, interface_config)
        response_reparse_exact = response_reparse_exact and bool(
            parsed == fixture["parsed_response"]
            and valid == fixture["response_valid"]
            and reason == fixture["validation_reason"]
        )
        response_hashes_exact = response_hashes_exact and bool(
            hashlib.sha256(fixture["raw_response"].encode()).hexdigest()
            == fixture["raw_response_sha256"]
        )
        authority_exact = authority_exact and bool(
            fixture["permanently_non_authoritative"]
            and not fixture["safe_hypothesis_universe_pruned"]
            and not fixture["executable"]
        )
    metrics = aggregate_model_fixtures(fixtures, evaluation_records, controlled_ids, baseline_config)
    parent = json.loads((PROJECT_ROOT / lock["parent_baseline_outcome"]).read_text())
    best_regret = parent["outcome"]["development_summary"]["best_nonoracle_baseline"]["mean_regret"]
    gates = evaluate_model_gates(metrics, best_regret, access, config)
    dependency_keys = (
        "config", "parent_baseline_outcome", "baseline_lock", "interface_outcome",
        "interface_lock", "visible_catalog", "controlled_identifiers", "model_manifest",
        "plan", "protocol", "tests", "runner", "verifier", "auditor", "census_harness",
        "implementation_audit",
    )
    checks = {
        "implementation_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "all_raw_responses_reparse_and_hash_exactly": bool(response_reparse_exact and response_hashes_exact),
        "metrics_gates_quality_and_decision_reconstruct_exactly": bool(
            metrics == result["metrics"] and gates == result["gates"]
            and quality_gate_pass(gates) == result["quality_gate_pass"]
            and result["passed"] == all(gates.values())
            and result["decision"] == (
                "model_qualifies_for_separately_preregistered_protected_test_only"
                if all(gates.values()) else "model_is_nonqualifying_and_protected_test_remains_sealed"
            )
        ),
        "condition_completed_exactly_once": bool(
            result["completed_condition"] and len(fixtures) == 192
            and access["model_load_count"] == 1 and access["model_generation_count"] == 192
        ),
        "non_authoritative_and_safe_hypothesis_boundary_holds": authority_exact,
        "protected_API_training_service_and_effect_boundary_holds": all(
            access[key] == 0 for key in (
                "protected_test_language_read_count", "manual_utterance_inspection_count",
                "LLM_API_call_count", "adapter_training_run_count", "real_service_call_count",
                "external_side_effect_count",
            )
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "107-open-world-local-model-outcome-audit",
        "experiment": "v107_open_world_local_model_outcome_audit",
        "passed": integrity_passed,
        "quality_gate_pass": result["quality_gate_pass"],
        "decision": (
            "freeze_qualifying_V107_and_authorize_protected_test_preregistration"
            if integrity_passed and result["quality_gate_pass"]
            else "freeze_nonqualifying_V107_and_close_model_branch"
        ),
        "checks": checks, "independent_metrics": metrics,
        "regret_above_best_nonoracle_baseline": metrics["mean_regret"] - best_regret,
        "additional_access": {
            "development_language_read_count": 1, "protected_test_language_read_count": 0,
            "manual_utterance_inspection_count": 0, "model_load_count": 0,
            "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "implementation_lock": implementation_lock_path, "result": result_path,
        "access": access_path, "verifier": verifier_path, "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "107-open-world-local-model-outcome-lock",
        "experiment": "v107_open_world_local_model_outcome_lock",
        "outcome": {
            "passed": True, "condition_completed": result["completed_condition"],
            "quality_gate_pass": result["quality_gate_pass"], "decision": audit["decision"],
            "metrics": metrics, "gates": gates,
            "regret_above_best_nonoracle_baseline": metrics["mean_regret"] - best_regret,
        },
        "authorization": {
            "modify_rerun_or_retry_V107": False,
            "preregister_one_identical_protected_test_run": bool(result["quality_gate_pass"]),
            "read_protected_test_before_separate_lock": False,
            "run_additional_local_or_API_models": False,
            "combine_models_or_train_adapter": False,
            "prune_safe_hypotheses_or_grant_model_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
            "close_model_branch_without_protected_test": not result["quality_gate_pass"],
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
