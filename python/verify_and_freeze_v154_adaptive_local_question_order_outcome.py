#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v154_adaptive_local_question_order import evaluate_condition, parse_ranking


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v154-adaptive-local-question-order-lock.json"
    realization_dir = PROJECT_ROOT / "outputs/v154-adaptive-local-question-order/model-realization"
    result_path = realization_dir / "result.json"
    access_path = realization_dir / "access.json"
    direct_path = realization_dir / "direct/result.json"
    low_path = realization_dir / "bounded-low-reasoning/result.json"
    doc_path = PROJECT_ROOT / "docs/v154-adaptive-local-question-order-results.md"
    audit_path = PROJECT_ROOT / "outputs/v154-adaptive-local-question-order/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v154-adaptive-local-question-order-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v154_adaptive_local_question_order_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V154 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V154 results document before freezing the outcome")

    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    hidden = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    answers = json.loads((PROJECT_ROOT / lock["development_answer_metadata"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["interaction_catalog"]).read_text())
    witness = json.loads((PROJECT_ROOT / lock["witness_config"]).read_text())
    comparator = json.loads((PROJECT_ROOT / lock["comparator_config"]).read_text())
    direct = json.loads(direct_path.read_text())
    low = json.loads(low_path.read_text()) if low_path.is_file() else None
    expected_direct = evaluate_condition(direct["fixtures"], hidden, answers, catalog, witness, comparator, config)
    expected_low = (
        evaluate_condition(low["fixtures"], hidden, answers, catalog, witness, comparator, config)
        if low is not None else None
    )

    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
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

    all_rows = list(direct["fixtures"].values()) + (list(low["fixtures"].values()) if low else [])
    low_rows = list(low["fixtures"].values()) if low else []
    direct_qualified = expected_direct["qualified"]
    low_expected_to_run = not direct_qualified
    conditional_counts_exact = bool(
        access["direct_generation_count"] == 96
        and (
            (
                direct_qualified
                and low is None
                and access["bounded_reasoning_phase_generation_count"] == 0
                and access["bounded_final_phase_generation_count"] == 0
                and access["total_generation_count"] == 96
            )
            or (
                not direct_qualified
                and low is not None
                and access["bounded_reasoning_phase_generation_count"] == 96
                and access["bounded_final_phase_generation_count"] == 96
                and access["total_generation_count"] == 288
            )
        )
    )
    if direct_qualified:
        expected_selected = "direct"
        expected_decision = config["decisionRule"]["ifDirectQualifies"]
    elif expected_low is not None and expected_low["qualified"]:
        expected_selected = "bounded-low-reasoning"
        expected_decision = config["decisionRule"]["ifDirectFailsAndBoundedLowReasoningQualifies"]
    else:
        expected_selected = None
        expected_decision = config["decisionRule"]["otherwise"]

    prohibited_access = (
        "closed_answer_model_generation_count", "evaluation_fixture_model_generation_count",
        "retry_count", "manual_raw_response_inspection_count", "persisted_raw_response_count",
        "API_call_count", "training_run_count", "real_service_call_count",
        "external_side_effect_count", "actual_execution_count",
    )
    checks = {
        "lock_and_dependencies_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "completed_exact_adaptive_census": bool(
            result["completed_condition"]
            and len(direct["fixtures"]) == 96
            and (low is None or len(low["fixtures"]) == 96)
            and result["bounded_low_reasoning_triggered"] == low_expected_to_run
        ),
        "direct_summary_exact": result["direct_summary"] == expected_direct,
        "bounded_low_summary_exact_if_triggered": result["bounded_low_reasoning_summary"] == expected_low,
        "adaptive_decision_and_selection_exact": bool(
            result["selected_condition"] == expected_selected
            and result["decision"] == expected_decision
            and result["conditions_run"]
            == (["direct"] if low is None else ["direct", "bounded-low-reasoning"])
        ),
        "conditional_generation_counts_exact": conditional_counts_exact,
        "single_model_and_tokenizer_load": access["model_load_count"] == 1 and access["tokenizer_load_count"] == 1,
        "all_rankings_reconstruct_or_use_frozen_fallback": reconstruct(direct) and (low is None or reconstruct(low)),
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
        "bounded_low_phase_budgets_exact_if_triggered": all(
            0 <= row["reasoning_phase_generated_token_count"] <= config["conditions"]["boundedLowReasoning"]["reasoningPhaseMaximumTokens"]
            and 0 <= row["final_phase_generated_token_count"] <= config["conditions"]["boundedLowReasoning"]["finalPhaseMaximumTokens"]
            and row["generated_token_count"]
            == row["reasoning_phase_generated_token_count"] + row["final_phase_generated_token_count"]
            for row in low_rows
        ),
        "all_outputs_permanently_non_authoritative_nonexecuting_without_candidate_fields": all(
            row["permanently_non_authoritative"]
            and not row["authoritative_hypothesis_universe_pruned"]
            and not row["capability_defined_or_registered"]
            and not row["executable"]
            and row["actual_execution_count"] == 0
            and not ({"candidate_state_ids", "state_ranking", "llm_proposal", "confidence"} & set(row))
            for row in all_rows
        ),
        "all_access_gates_pass": all(result["access_gates"].values()),
        "zero_closed_answer_evaluation_retry_raw_API_training_services_side_effects_execution": all(
            access[key] == 0 for key in prohibited_access
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "154-adaptive-local-question-order-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "selected_condition": expected_selected,
        "decision": expected_decision,
        "direct_summary": expected_direct,
        "bounded_low_reasoning_summary": expected_low,
        "access_gates": result["access_gates"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    paths = {
        "analysis_lock": lock_path,
        "result": result_path,
        "access": access_path,
        "direct_result": direct_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    if low is not None:
        paths["bounded_low_reasoning_result"] = low_path
    outcome: dict[str, Any] = {
        "schema_version": "154-adaptive-local-question-order-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "audit_pass": True,
            "selected_condition": expected_selected,
            "development_qualified": expected_selected is not None,
            "decision": expected_decision,
            "direct_summary": expected_direct,
            "bounded_low_reasoning_summary": expected_low,
        },
        "authorization": {
            "retain_as_project_authored_synthetic_development_evidence_only": True,
            "preregister_separate_V152_evaluation_realization": expected_selected is not None,
            "run_or_open_V152_evaluation_before_separate_preregistration": False,
            "retry_rerun_reprompt_change_reasoning_budget_tune_threshold_fit_or_mine_V154": False,
            "add_candidate_state_proposal_confidence_or_pruning": False,
            "run_API_training_induction_authority_action_or_execution": False,
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
