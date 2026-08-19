#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from run_v109_open_world_typed_choice import decision_for, payload_hash, prepare_fixture_rows
from v109_open_world_typed_choice import evaluate_v109_gates, validate_and_expand_choice


def main() -> None:
    implementation_lock_path = PROJECT_ROOT / "configs/v109-open-world-typed-choice-implementation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v109-open-world-typed-choice/holdback-evaluation/result.json"
    access_path = PROJECT_ROOT / "outputs/v109-open-world-typed-choice/holdback-evaluation/access.json"
    doc_path = PROJECT_ROOT / "docs/v109-open-world-typed-choice-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v109_typed_choice_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v109-open-world-typed-choice/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v109-open-world-typed-choice-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V109 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V109 result before freezing")
    lock = json.loads(implementation_lock_path.read_text())
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    config = lock["config_payload"]
    baseline_config = lock["baseline_config_payload"]
    choice_catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    _, holdback_records, controlled_ids = prepare_fixture_rows(lock)
    fixtures = result["fixtures"]
    response_reparse_exact = True
    response_hashes_exact = True
    authority_exact = True
    for fixture in fixtures.values():
        parsed, valid, reason = validate_and_expand_choice(fixture["raw_response"], choice_catalog, config)
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
    metrics, interface_gates, semantic_gates, access_gates = evaluate_v109_gates(
        fixtures, holdback_records, controlled_ids, choice_catalog, access,
        baseline_config, config,
    )
    interface_pass = all(interface_gates.values())
    semantic_pass = all(semantic_gates.values())
    access_pass = all(access_gates.values())
    decision = decision_for(interface_pass, semantic_pass, access_pass)
    combined_gates = {
        **{f"interface::{key}": value for key, value in interface_gates.items()},
        **{f"semantic::{key}": value for key, value in semantic_gates.items()},
        **{f"access::{key}": value for key, value in access_gates.items()},
    }
    dependency_keys = (
        "config", "parent_forensics_outcome", "forensics_lock", "V107_outcome",
        "V107_implementation_lock", "V107_result", "baseline_outcome", "baseline_lock",
        "development_membership", "development_language", "visible_catalog",
        "controlled_identifiers", "model_manifest", "choice_catalog", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "census_harness", "implementation_audit",
    )
    checks = {
        "implementation_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "all_raw_responses_reparse_and_hash_exactly": bool(response_reparse_exact and response_hashes_exact),
        "metrics_gates_pass_flags_and_decision_reconstruct_exactly": bool(
            metrics == result["metrics"] and interface_gates == result["interface_gates"]
            and semantic_gates == result["semantic_gates"] and access_gates == result["access_gates"]
            and combined_gates == result["gates"]
            and interface_pass == result["interface_pass"]
            and semantic_pass == result["semantic_pass"]
            and access_pass == result["access_pass"] and decision == result["decision"]
            and result["passed"] == all(combined_gates.values())
        ),
        "condition_completed_exactly_once": bool(
            result["completed_condition"] and len(fixtures) == 192
            and access["model_load_count"] == 1 and access["model_generation_count"] == 192
        ),
        "non_authoritative_complete_hypothesis_boundary_holds": authority_exact,
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
        "schema_version": "109-open-world-typed-choice-outcome-audit",
        "experiment": "v109_open_world_typed_choice_outcome_audit",
        "passed": integrity_passed, "interface_pass": interface_pass,
        "semantic_pass": semantic_pass, "access_pass": access_pass,
        "decision": decision, "checks": checks, "independent_metrics": metrics,
        "independent_interface_gates": interface_gates,
        "independent_semantic_gates": semantic_gates,
        "independent_access_gates": access_gates,
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
        "schema_version": "109-open-world-typed-choice-outcome-lock",
        "experiment": "v109_open_world_typed_choice_outcome_lock",
        "outcome": {
            "passed": True, "condition_completed": result["completed_condition"],
            "interface_pass": interface_pass, "semantic_pass": semantic_pass,
            "access_pass": access_pass, "decision": decision,
            "metrics": metrics, "interface_gates": interface_gates,
            "semantic_gates": semantic_gates, "access_gates": access_gates,
            "regret_above_ask_always": metrics["mean_regret"] - 1.125,
        },
        "authorization": {
            "modify_rerun_retry_or_replace_V109": False,
            "preregister_sequential_clarification_benchmark": bool(interface_pass and access_pass),
            "read_protected_test_or_run_model_before_separate_lock": False,
            "run_API_model_or_train_adapter": False,
            "prune_safe_hypotheses_or_grant_model_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
            "close_model_serialization_branch": not interface_pass,
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
