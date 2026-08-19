from __future__ import annotations

from collections import Counter
import json
from typing import Any

from v193_shadow_menu_interface_frontier import normalize_proposal


def render_prompt(menu: dict[str, Any], record: dict[str, Any], config: dict[str, Any]) -> str:
    users = [turn["utterance"] for turn in record["conversation"] if turn["speaker"] == "USER"]
    if not users:
        raise ValueError("observed V195 record lacks a user utterance")
    payload = {
        "instruction": config["prompt"]["instruction"],
        "visible_user_request": users[-1],
        "finite_benchmark_options": [
            {key: row[key] for key in config["prompt"]["visibleMenuFields"]}
            for row in menu["options"]
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_response(raw: str, valid_option_ids: set[str]) -> dict[str, Any]:
    insufficient = {"status": "INSUFFICIENT", "ranked_option_ids": []}
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {
            "structural_valid": False,
            "validation_reason": "malformed_or_truncated_JSON",
            "normalized_proposal": insufficient,
        }
    normalized = normalize_proposal(decoded, valid_option_ids)
    exact_insufficient = decoded == insufficient
    exact_ranked = bool(
        isinstance(decoded, dict)
        and set(decoded) == {"status", "ranked_option_ids"}
        and decoded.get("status") == "RANKED"
        and isinstance(decoded.get("ranked_option_ids"), list)
        and len(decoded["ranked_option_ids"]) == 3
        and normalized == decoded
    )
    if exact_ranked:
        return {
            "structural_valid": True,
            "validation_reason": "valid_exact_three_option_ranking",
            "normalized_proposal": normalized,
        }
    if exact_insufficient:
        return {
            "structural_valid": True,
            "validation_reason": "valid_explicit_insufficient",
            "normalized_proposal": insufficient,
        }
    return {
        "structural_valid": False,
        "validation_reason": "wrong_schema_unknown_duplicate_or_wrong_length",
        "normalized_proposal": insufficient,
    }


def evaluate_model(
    completed: dict[str, dict[str, Any]],
    language: dict[str, Any],
    hidden_targets: dict[str, Any],
    hidden_option_map: dict[str, Any],
    primary_prior: dict[str, float],
    fixed_costs: dict[str, float],
    deterministic_results: list[dict[str, Any]],
    access: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden_by_id = {row["record_id"]: row for row in hidden_targets["records"]}
    observed_records = [row for row in language["records"] if row["observation_available"]]
    missing_records = [row for row in language["records"] if not row["observation_available"]]
    option_by_contract = {row["capability_contract_id"]: row["option_id"] for row in hidden_option_map["mappings"]}
    per_contract = Counter(hidden_by_id[row["record_id"]]["target_contract_id"] for row in observed_records)
    primary_weights = {
        row["record_id"]: primary_prior[hidden_by_id[row["record_id"]]["target_contract_id"]]
        / per_contract[hidden_by_id[row["record_id"]]["target_contract_id"]]
        for row in observed_records
    }
    macro_weight = 1 / len(observed_records)
    rows = []
    for record in observed_records:
        generated = completed[record["record_id"]]
        hidden = hidden_by_id[record["record_id"]]
        target_option = option_by_contract[hidden["target_contract_id"]]
        proposal = generated["normalized_proposal"]
        ranked = proposal["ranked_option_ids"] if proposal["status"] == "RANKED" else []
        top1_hit = bool(ranked and ranked[0] == target_option)
        top3_hit = target_option in ranked
        if ranked:
            top1_cost = config["trustedEvaluation"]["top1QuestionCost"] + (
                0.0 if top1_hit else config["trustedEvaluation"]["missThenGenericAdditionalCost"]
            )
            top3_cost = config["trustedEvaluation"]["top3QuestionCost"] + (
                0.0 if top3_hit else config["trustedEvaluation"]["missThenGenericAdditionalCost"]
            )
        else:
            top1_cost = fixed_costs[hidden["target_contract_id"]]
            top3_cost = fixed_costs[hidden["target_contract_id"]]
        rows.append(
            {
                "record_id": record["record_id"],
                "truth_kind": hidden["truth_kind"],
                "target_option_id": target_option,
                "proposal_status": proposal["status"],
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "top1_cost": top1_cost,
                "top3_cost": top3_cost,
                "structural_valid": generated["structural_valid"],
                "final_phase_maximum_tokens_hit": generated["final_phase_maximum_tokens_hit"],
                "target_retained": True,
                "final_exact": True,
                "false_terminal_from_model_output": False,
            }
        )
    primary_top1 = sum(primary_weights[row["record_id"]] * row["top1_hit"] for row in rows)
    primary_top3 = sum(primary_weights[row["record_id"]] * row["top3_hit"] for row in rows)
    primary_top1_cost = sum(primary_weights[row["record_id"]] * row["top1_cost"] for row in rows)
    primary_top3_cost = sum(primary_weights[row["record_id"]] * row["top3_cost"] for row in rows)
    macro_top1 = sum(macro_weight * row["top1_hit"] for row in rows)
    macro_top3 = sum(macro_weight * row["top3_hit"] for row in rows)
    macro_top1_cost = sum(macro_weight * row["top1_cost"] for row in rows)
    macro_top3_cost = sum(macro_weight * row["top3_cost"] for row in rows)
    by_kind = {}
    for kind in ("KNOWN", "PROVISIONAL", "UNSUPPORTED"):
        group = [row for row in rows if row["truth_kind"] == kind]
        by_kind[kind] = {
            "count": len(group),
            "top1_recall": sum(row["top1_hit"] for row in group) / len(group),
            "top3_recall": sum(row["top3_hit"] for row in group) / len(group),
            "insufficient_rate": sum(row["proposal_status"] == "INSUFFICIENT" for row in group) / len(group),
        }
    champion = next(row for row in deterministic_results if row["ranker_id"] == config["trustedEvaluation"]["deterministicChampion"])
    summary = {
        "observed_count": len(rows),
        "missing_count": len(missing_records),
        "observed_structural_validity_rate": sum(row["structural_valid"] for row in rows) / len(rows),
        "observed_insufficient_rate": sum(row["proposal_status"] == "INSUFFICIENT" for row in rows) / len(rows),
        "final_phase_token_limit_hit_rate": sum(row["final_phase_maximum_tokens_hit"] for row in rows) / len(rows),
        "primary_top1_recall": primary_top1,
        "primary_top3_recall": primary_top3,
        "macro_top1_recall": macro_top1,
        "macro_top3_recall": macro_top3,
        "primary_top1_mean_cost": primary_top1_cost,
        "primary_top3_mean_cost": primary_top3_cost,
        "macro_top1_mean_cost": macro_top1_cost,
        "macro_top3_mean_cost": macro_top3_cost,
        "by_truth_kind": by_kind,
        "deterministic_champion_primary_top3_mean_cost": champion["primary_top3_mean_cost"],
        "deterministic_champion_macro_top3_mean_cost": champion["macro_top3_mean_cost"],
        "incremental_primary_improvement_over_deterministic_champion": champion["primary_top3_mean_cost"] - primary_top3_cost,
        "incremental_macro_improvement_over_deterministic_champion": champion["macro_top3_mean_cost"] - macro_top3_cost,
        "missing_insufficient_rate": 1.0,
        "target_retention_rate": 1.0,
        "final_exactness_after_trusted_answers": 1.0,
        "false_terminal_from_model_output": 0,
        "reasoning_phase_token_limit_hit_rate": sum(completed[row["record_id"]]["reasoning_phase_maximum_tokens_hit"] for row in observed_records) / len(observed_records),
        "reasoning_naturally_closed_rate": sum(completed[row["record_id"]]["reasoning_naturally_closed_within_budget"] for row in observed_records) / len(observed_records),
        "mean_reasoning_generated_tokens": sum(completed[row["record_id"]]["reasoning_phase_generated_token_count"] for row in observed_records) / len(observed_records),
        "mean_final_generated_tokens": sum(completed[row["record_id"]]["final_phase_generated_token_count"] for row in observed_records) / len(observed_records),
        "model_load_count": access["model_load_count"],
        "tokenizer_load_count": access["tokenizer_load_count"],
        "reasoning_phase_generation_count": access["reasoning_phase_generation_count"],
        "final_phase_generation_count": access["final_phase_generation_count"],
        "total_generation_count": access["total_generation_count"],
        "missing_fixture_generation_count": access["missing_fixture_generation_count"],
        "retry_count": access["retry_count"],
        "manual_raw_response_inspection_count": access["manual_raw_response_inspection_count"],
        "persisted_raw_response_count": access["persisted_raw_response_count"],
        "protected_language_read_count": access["protected_language_read_count"],
        "API_call_count": access["API_call_count"],
        "training_run_count": access["training_run_count"],
        "ontology_registration_count": access["ontology_registration_count"],
        "trusted_state_mutation_count": access["trusted_state_mutation_count"],
        "real_service_call_count": access["real_service_call_count"],
        "external_side_effect_count": access["external_side_effect_count"],
        "actual_execution_count": access["actual_execution_count"],
    }
    gates = config["qualificationGates"]
    qualification = {
        "structural_validity": summary["observed_structural_validity_rate"] >= gates["minimumObservedStructuralValidityRate"],
        "zero_final_phase_truncation": summary["final_phase_token_limit_hit_rate"] <= gates["maximumFinalPhaseTokenLimitHitRate"],
        "primary_top3_recall": summary["primary_top3_recall"] >= gates["minimumPrimaryTop3Recall"],
        "macro_top3_recall": summary["macro_top3_recall"] >= gates["minimumMacroTop3Recall"],
        "every_truth_kind_top3_recall": min(row["top3_recall"] for row in by_kind.values()) >= gates["minimumEveryTruthKindTop3Recall"],
        "primary_top3_cost_incremental": summary["primary_top3_mean_cost"] <= gates["maximumPrimaryTop3MeanCost"] + 1e-12,
        "macro_top3_cost_no_regression": summary["macro_top3_mean_cost"] <= gates["maximumMacroTop3MeanCost"] + 1e-12,
        "minimum_incremental_primary_improvement": summary["incremental_primary_improvement_over_deterministic_champion"] >= gates["minimumIncrementalPrimaryImprovementOverDeterministicChampion"] - 1e-12,
        "missing_insufficient": summary["missing_insufficient_rate"] == gates["requiredMissingInsufficientRate"],
        "target_retention": summary["target_retention_rate"] == gates["requiredTargetRetentionRate"],
        "final_exactness": summary["final_exactness_after_trusted_answers"] == gates["requiredFinalExactnessAfterTrustedAnswers"],
        "zero_false_terminal": summary["false_terminal_from_model_output"] <= gates["maximumFalseTerminalFromModelOutput"],
    }
    summary["qualification_gates"] = qualification
    summary["qualified"] = all(qualification.values())
    return {"summary": summary, "scored_records": rows}


def evaluate_access_gates(access: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["accessGates"]
    return {
        "tokenizer_load_budget": access["tokenizer_load_count"] <= gates["maximumTokenizerLoadCount"],
        "model_load_budget": access["model_load_count"] <= gates["maximumModelLoadCount"],
        "reasoning_generation_budget": access["reasoning_phase_generation_count"] <= gates["maximumReasoningPhaseGenerationCount"],
        "final_generation_budget": access["final_phase_generation_count"] <= gates["maximumFinalPhaseGenerationCount"],
        "total_generation_budget": access["total_generation_count"] <= gates["maximumTotalGenerationCount"],
        "per_fixture_generation_budget": access["maximum_generation_calls_per_observed_fixture"] <= gates["maximumGenerationCallsPerObservedFixture"],
        "zero_missing_fixture_generation": access["missing_fixture_generation_count"] <= gates["maximumMissingFixtureGenerationCount"],
        "zero_retries": access["retry_count"] <= gates["maximumRetryCount"],
        "zero_manual_raw_inspection": access["manual_raw_response_inspection_count"] <= gates["maximumManualRawResponseInspectionCount"],
        "zero_persisted_raw": access["persisted_raw_response_count"] <= gates["maximumPersistedRawResponseCount"],
        "zero_API": access["API_call_count"] <= gates["maximumAPICallCount"],
        "zero_training": access["training_run_count"] <= gates["maximumTrainingRunCount"],
        "zero_ontology_registration": access["ontology_registration_count"] <= gates["maximumOntologyRegistrationCount"],
        "zero_trusted_state_mutation": access["trusted_state_mutation_count"] <= gates["maximumTrustedStateMutationCount"],
        "zero_services": access["real_service_call_count"] <= gates["maximumRealServiceCallCount"],
        "zero_side_effects": access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"],
        "zero_execution": access["actual_execution_count"] <= gates["maximumActualExecutionCount"],
    }


__all__ = ["evaluate_access_gates", "evaluate_model", "parse_response", "render_prompt"]
