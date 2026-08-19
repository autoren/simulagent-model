#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v154_adaptive_local_question_order import evaluate_condition, parse_ranking
from v154r1_outcome_verifier_repair import canonical_json, sole_json_key_type_mismatch


def main() -> None:
    repair_lock_path = PROJECT_ROOT / "configs/v154r1-outcome-verifier-repair-lock.json"
    results_doc_path = PROJECT_ROOT / "docs/v154r1-outcome-verifier-repair-results.md"
    audit_path = PROJECT_ROOT / "outputs/v154r1-outcome-verifier-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v154r1-outcome-verifier-repair-outcome-lock.json"
    nominal_v154_outcome = PROJECT_ROOT / "configs/v154-adaptive-local-question-order-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v154r1_outcome_verifier_repair.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V154r1 outcome already frozen")
    if nominal_v154_outcome.exists():
        raise RuntimeError("nominal V154 outcome lock unexpectedly exists")
    if not results_doc_path.is_file():
        raise RuntimeError("write the V154r1 technical results document before freezing")

    repair_lock = json.loads(repair_lock_path.read_text())
    repair_config = repair_lock["config_payload"]
    parent = json.loads((PROJECT_ROOT / repair_lock["parent_analysis_lock"]).read_text())
    config = parent["config_payload"]
    failed_audit = json.loads((PROJECT_ROOT / repair_lock["failed_V154_outcome_audit"]).read_text())
    result = json.loads((PROJECT_ROOT / repair_lock["V154_result"]).read_text())
    access = json.loads((PROJECT_ROOT / repair_lock["V154_access"]).read_text())
    direct = json.loads((PROJECT_ROOT / repair_lock["V154_direct_result"]).read_text())
    low = json.loads((PROJECT_ROOT / repair_lock["V154_bounded_low_result"]).read_text())
    hidden = json.loads((PROJECT_ROOT / parent["development_hidden_fixtures"]).read_text())
    answers = json.loads((PROJECT_ROOT / parent["development_answer_metadata"]).read_text())
    catalog = json.loads((PROJECT_ROOT / parent["interaction_catalog"]).read_text())
    witness = json.loads((PROJECT_ROOT / parent["witness_config"]).read_text())
    comparator = json.loads((PROJECT_ROOT / parent["comparator_config"]).read_text())
    expected_direct = evaluate_condition(
        direct["fixtures"], hidden, answers, catalog, witness, comparator, config
    )
    expected_low = evaluate_condition(
        low["fixtures"], hidden, answers, catalog, witness, comparator, config
    )

    repair_dependencies = [
        key for key in repair_lock if not key.endswith("_sha256") and f"{key}_sha256" in repair_lock
    ]
    parent_dependencies = [
        key for key in parent if not key.endswith("_sha256") and f"{key}_sha256" in parent
    ]
    raw_fields = {
        "raw_response", "prompt", "payload", "conversation", "thinking_trace",
        "reasoning_text", "final_text", "reasoning_content",
    }

    def reconstruct(condition: dict[str, Any]) -> bool:
        checks = []
        for row in condition["fixtures"].values():
            if row["ranking_valid"]:
                reparsed = parse_ranking(json.dumps(row["normalized_ranking"]), catalog, config)
                checks.append(
                    reparsed["ranking_valid"]
                    and reparsed["validation_reason"] == row["validation_reason"]
                    and reparsed["query_ranking"] == row["query_ranking"]
                )
            else:
                checks.append(
                    row["normalized_ranking"] is None
                    and row["query_ranking"] == config["fallbackQueryRanking"]
                )
        return bool(checks and all(checks))

    all_rows = list(direct["fixtures"].values()) + list(low["fixtures"].values())
    low_rows = list(low["fixtures"].values())
    expected_selected = None
    expected_decision = config["decisionRule"]["otherwise"]
    prohibited_access = (
        "closed_answer_model_generation_count", "evaluation_fixture_model_generation_count",
        "retry_count", "manual_raw_response_inspection_count", "persisted_raw_response_count",
        "API_call_count", "training_run_count", "real_service_call_count",
        "external_side_effect_count", "actual_execution_count",
    )
    false_original_checks = sorted(key for key, value in failed_audit["checks"].items() if not value)
    checks = {
        "repair_lock_and_dependencies_exact": bool(
            valid_lock(repair_lock)
            and all(
                file_sha256(PROJECT_ROOT / repair_lock[key]) == repair_lock[f"{key}_sha256"]
                for key in repair_dependencies
            )
        ),
        "parent_lock_and_dependencies_exact": bool(
            valid_lock(parent)
            and all(
                file_sha256(PROJECT_ROOT / parent[key]) == parent[f"{key}_sha256"]
                for key in parent_dependencies
            )
        ),
        "failed_original_audit_preserved_with_exact_two_false_checks": bool(
            not failed_audit["passed"]
            and false_original_checks == sorted(repair_config["diagnosis"]["expectedFailedChecks"])
            and file_sha256(PROJECT_ROOT / repair_lock["failed_V154_outcome_audit"])
            == repair_lock["failed_V154_outcome_audit_sha256"]
        ),
        "direct_raw_mismatch_diagnosed_as_sole_JSON_key_type_difference": sole_json_key_type_mismatch(
            expected_direct, result["direct_summary"]
        ),
        "bounded_low_raw_mismatch_diagnosed_as_sole_JSON_key_type_difference": sole_json_key_type_mismatch(
            expected_low, result["bounded_low_reasoning_summary"]
        ),
        "direct_summary_exact_after_locked_JSON_canonicalization": bool(
            canonical_json(expected_direct) == result["direct_summary"]
        ),
        "bounded_low_summary_exact_after_locked_JSON_canonicalization": bool(
            canonical_json(expected_low) == result["bounded_low_reasoning_summary"]
        ),
        "completed_exact_adaptive_census": bool(
            result["completed_condition"]
            and result["bounded_low_reasoning_triggered"]
            and len(direct["fixtures"]) == 96
            and len(low["fixtures"]) == 96
        ),
        "adaptive_decision_and_selection_exact": bool(
            not expected_direct["qualified"]
            and not expected_low["qualified"]
            and result["selected_condition"] == expected_selected
            and result["decision"] == expected_decision
            and result["conditions_run"] == ["direct", "bounded-low-reasoning"]
            and repair_config["frozenOutcome"]["selectedCondition"] == expected_selected
            and repair_config["frozenOutcome"]["decision"] == expected_decision
        ),
        "conditional_generation_counts_exact": bool(
            access["direct_generation_count"] == 96
            and access["bounded_reasoning_phase_generation_count"] == 96
            and access["bounded_final_phase_generation_count"] == 96
            and access["total_generation_count"] == 288
        ),
        "single_original_model_and_tokenizer_load": bool(
            access["model_load_count"] == 1 and access["tokenizer_load_count"] == 1
        ),
        "all_rankings_reconstruct_or_use_frozen_fallback": reconstruct(direct) and reconstruct(low),
        "no_raw_prompt_payload_conversation_reasoning_or_final_text_persisted": all(
            not (raw_fields & set(row)) and not row["raw_response_persisted"] for row in all_rows
        ),
        "resource_diagnostics_finite_and_bounded": all(
            isinstance(row["prompt_token_count"], int)
            and 0 < row["prompt_token_count"] <= config["prompt"]["maximumPromptTokens"]
            and isinstance(row["generated_token_count"], int)
            and row["generated_token_count"] >= 0
            and isinstance(row["generation_seconds"], (int, float))
            and not isinstance(row["generation_seconds"], bool)
            and math.isfinite(row["generation_seconds"])
            and row["generation_seconds"] >= 0.0
            for row in all_rows
        ),
        "bounded_low_phase_budgets_exact": all(
            0 <= row["reasoning_phase_generated_token_count"]
            <= config["conditions"]["boundedLowReasoning"]["reasoningPhaseMaximumTokens"]
            and 0 <= row["final_phase_generated_token_count"]
            <= config["conditions"]["boundedLowReasoning"]["finalPhaseMaximumTokens"]
            and row["generated_token_count"]
            == row["reasoning_phase_generated_token_count"] + row["final_phase_generated_token_count"]
            for row in low_rows
        ),
        "all_outputs_non_authoritative_nonexecuting_without_candidate_fields": all(
            row["permanently_non_authoritative"]
            and not row["authoritative_hypothesis_universe_pruned"]
            and not row["capability_defined_or_registered"]
            and not row["executable"]
            and row["actual_execution_count"] == 0
            and not ({"candidate_state_ids", "state_ranking", "llm_proposal", "confidence"} & set(row))
            for row in all_rows
        ),
        "all_original_access_gates_pass": all(result["access_gates"].values()),
        "zero_original_closed_answer_evaluation_retry_raw_API_training_services_side_effects_execution": all(
            access[key] == 0 for key in prohibited_access
        ),
        "zero_repair_model_language_evaluation_or_external_access": all(
            value == 0 for value in repair_config["accessGates"].values()
        ),
        "nominal_V154_outcome_lock_remains_absent": not nominal_v154_outcome.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "154r1-outcome-verifier-repair-outcome-audit",
        "experiment": repair_config["experiment"],
        "passed": passed,
        "checks": checks,
        "repair": "recursive_JSON_canonicalization_of_recomputed_summary_before_exact_comparison",
        "selected_condition": expected_selected,
        "development_qualified": False,
        "decision": expected_decision,
        "direct_summary": canonical_json(expected_direct),
        "bounded_low_reasoning_summary": canonical_json(expected_low),
        "repair_access": {key: 0 for key in repair_config["accessGates"]},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    paths = {
        "repair_lock": repair_lock_path,
        "parent_analysis_lock": PROJECT_ROOT / repair_lock["parent_analysis_lock"],
        "failed_V154_outcome_audit": PROJECT_ROOT / repair_lock["failed_V154_outcome_audit"],
        "V154_result": PROJECT_ROOT / repair_lock["V154_result"],
        "V154_access": PROJECT_ROOT / repair_lock["V154_access"],
        "V154_direct_result": PROJECT_ROOT / repair_lock["V154_direct_result"],
        "V154_bounded_low_result": PROJECT_ROOT / repair_lock["V154_bounded_low_result"],
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": results_doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "154r1-outcome-verifier-repair-outcome-lock",
        "experiment": repair_config["experiment"],
        "outcome": {
            "passed": True,
            "technical_repair_only": True,
            "selected_condition": expected_selected,
            "development_qualified": False,
            "decision": expected_decision,
            "direct_summary": canonical_json(expected_direct),
            "bounded_low_reasoning_summary": canonical_json(expected_low),
        },
        "authorization": {
            "retain_V154_as_project_authored_synthetic_negative_development_evidence": True,
            "treat_repair_as_new_scientific_evidence": False,
            "create_nominal_V154_outcome_lock": False,
            "open_or_run_evaluation": False,
            "retry_rerun_reprompt_change_reasoning_threshold_fit_calibrate_or_tune_V154": False,
            "add_candidate_state_proposal_confidence_or_pruning": False,
            "run_model_API_training_services_authority_action_side_effect_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
