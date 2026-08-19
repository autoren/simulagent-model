#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v105_open_world_interface import validate_response
from v164_local_residual_open_set_transfer import (
    aggregate_residual_fixtures,
    evaluate_quality_and_access_gates,
)
from run_v164_local_residual_open_set_transfer import (
    payload_hash,
    prepare_records,
)


def main() -> None:
    implementation_lock_path = (
        PROJECT_ROOT / "configs/v164-local-residual-open-set-transfer-lock.json"
    )
    result_path = (
        PROJECT_ROOT
        / "outputs/v164-local-residual-open-set-transfer/development/result.json"
    )
    access_path = (
        PROJECT_ROOT
        / "outputs/v164-local-residual-open-set-transfer/development/access.json"
    )
    doc_path = (
        PROJECT_ROOT / "docs/v164-local-residual-open-set-transfer-results.md"
    )
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v164_local_residual_open_set_transfer_outcome.py"
    )
    audit_path = (
        PROJECT_ROOT
        / "outputs/v164-local-residual-open-set-transfer/outcome-audit.json"
    )
    outcome_path = (
        PROJECT_ROOT
        / "configs/v164-local-residual-open-set-transfer-outcome-lock.json"
    )
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V164 model outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V164 result before freezing")

    lock = json.loads(implementation_lock_path.read_text())
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    residual_records, evaluation_records, consensus_predictions = prepare_records(lock)
    fixtures = result["fixtures"]
    response_reparse_exact = True
    response_hashes_exact = True
    authority_exact = True
    for fixture in fixtures.values():
        parsed, valid, reason = validate_response(
            fixture["raw_response"], catalog, lock["interface_config_payload"]
        )
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
            and not fixture["nonresidual_decision_overridden"]
            and not fixture["executable"]
        )
    aggregate = aggregate_residual_fixtures(
        fixtures,
        residual_records,
        evaluation_records,
        consensus_predictions,
        lock["baseline_config_payload"],
    )
    parent = json.loads(
        (PROJECT_ROOT / lock["parent_deterministic_outcome"]).read_text()
    )
    frozen_consensus_regret = parent["outcome"]["baseline_metrics"][
        "deterministic_consensus"
    ]["mean_regret"]
    gates = evaluate_quality_and_access_gates(
        aggregate, frozen_consensus_regret, access, lock["config_payload"]
    )
    dependency_keys = (
        "config",
        "parent_deterministic_outcome",
        "parent_baseline_lock",
        "historical_interface_outcome",
        "historical_interface_lock",
        "direct_decoding_evidence",
        "visible_catalog",
        "safe_hypothesis_universe",
        "residual_manifest",
        "baseline_predictions",
        "model_manifest",
        "plan",
        "protocol",
        "tests",
        "runner",
        "verifier",
        "auditor",
        "census_harness",
        "implementation_audit",
    )
    expected_decision = (
        "local_residual_hybrid_qualifies_for_separate_protected_protocol_preregistration_only"
        if all(gates.values())
        else "local_residual_hybrid_is_nonqualifying_and_protected_transfer_remains_sealed"
    )
    checks = {
        "implementation_lock_and_dependencies_are_exact": bool(
            payload_hash(
                {
                    key: value
                    for key, value in lock.items()
                    if key != "lock_payload_sha256"
                }
            )
            == lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "all_raw_responses_reparse_and_hash_exactly": bool(
            response_reparse_exact and response_hashes_exact
        ),
        "aggregate_gates_pass_and_decision_reconstruct_exactly": bool(
            aggregate == result["aggregate"]
            and gates == result["gates"]
            and result["passed"] == all(gates.values())
            and result["decision"] == expected_decision
        ),
        "condition_completed_exactly_once_on_residual": bool(
            result["completed_condition"]
            and len(fixtures) == 76
            and access["model_load_count"] == 1
            and access["model_generation_count"] == 76
            and access["retry_count"] == 0
            and aggregate["model_nonresidual_override_count"] == 0
        ),
        "non_authoritative_complete_hypothesis_boundary_holds": bool(
            authority_exact
            and aggregate["true_hypothesis_retention"] == 1.0
            and aggregate["controlled_missing_observation_abstention_accuracy"]
            == 1.0
        ),
        "protected_API_training_service_effect_and_execution_boundary_holds": all(
            access[key] == 0
            for key in (
                "protected_language_read_count",
                "manual_utterance_inspection_count",
                "manual_raw_response_inspection_count",
                "LLM_API_call_count",
                "adapter_training_run_count",
                "real_service_call_count",
                "external_side_effect_count",
                "actual_execution_count",
            )
        ),
    }
    integrity_passed = all(checks.values())
    quality_passed = all(gates.values())
    audit = {
        "schema_version": "164-local-residual-open-set-transfer-outcome-audit",
        "experiment": "v164_local_residual_open_set_transfer_outcome_audit",
        "passed": integrity_passed,
        "quality_gate_pass": quality_passed,
        "decision": (
            "freeze_qualifying_V164_and_authorize_protected_protocol_preregistration"
            if integrity_passed and quality_passed
            else (
                "freeze_nonqualifying_V164_and_close_residual_model_protocol"
                if integrity_passed
                else "reject_V164_development_outcome"
            )
        ),
        "checks": checks,
        "independent_aggregate": aggregate,
        "frozen_consensus_regret": frozen_consensus_regret,
        "additional_access": {
            "development_language_read_count": 1,
            "protected_language_read_count": 0,
            "manual_utterance_inspection_count": 0,
            "manual_raw_response_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "retry_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "implementation_lock": implementation_lock_path,
        "result": result_path,
        "access": access_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "164-local-residual-open-set-transfer-outcome-lock",
        "experiment": "v164_local_residual_open_set_transfer_outcome_lock",
        "outcome": {
            "passed": True,
            "condition_completed": result["completed_condition"],
            "quality_gate_pass": quality_passed,
            "decision": audit["decision"],
            "aggregate": aggregate,
            "gates": gates,
            "frozen_consensus_regret": frozen_consensus_regret,
        },
        "authorization": {
            "modify_rerun_retry_reprompt_or_retune_V164": False,
            "preregister_identical_protected_deterministic_plus_residual_model_protocol": bool(
                quality_passed
            ),
            "read_protected_transfer_before_separate_lock": False,
            "run_additional_local_or_API_models": False,
            "combine_models_or_train_adapter": False,
            "induce_register_or_execute_capability": False,
            "prune_hypotheses_or_grant_model_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
            "close_residual_model_protocol_without_protected_test": not quality_passed,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(outcome_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(outcome_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
