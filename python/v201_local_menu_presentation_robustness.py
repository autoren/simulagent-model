from __future__ import annotations

from collections import Counter
from typing import Any


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def evaluate_model(
    completed: dict[str, dict[str, Any]],
    language: dict[str, Any],
    hidden_targets: dict[str, Any],
    hidden_variant_maps: dict[str, Any],
    canonical_hidden_map: dict[str, Any],
    canonical_census: dict[str, Any],
    transformed_char_summary: dict[str, Any],
    primary_prior: dict[str, float],
    fixed_costs: dict[str, float],
    access: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden_by_id = {row["record_id"]: row for row in hidden_targets["records"]}
    maps_by_id = {row["record_id"]: row for row in hidden_variant_maps["records"]}
    canonical_contract_by_option = {
        row["option_id"]: row["capability_contract_id"] for row in canonical_hidden_map["mappings"]
    }
    canonical_outputs = canonical_census["fixtures"]
    baseline_by_variant = {row["variant_id"]: row for row in transformed_char_summary["variants"]}
    observed = [row for row in language["records"] if row["observation_available"]]
    missing = [row for row in language["records"] if not row["observation_available"]]
    per_contract = Counter(hidden_by_id[row["record_id"]]["target_contract_id"] for row in observed)
    primary_weights = {
        row["record_id"]: primary_prior[hidden_by_id[row["record_id"]]["target_contract_id"]]
        / per_contract[hidden_by_id[row["record_id"]]["target_contract_id"]]
        for row in observed
    }
    rows = []
    for record in observed:
        record_id = record["record_id"]
        canonical_proposal = canonical_outputs[record_id]["normalized_proposal"]
        canonical_ids = canonical_proposal["ranked_option_ids"] if canonical_proposal["status"] == "RANKED" else []
        canonical_contracts = [canonical_contract_by_option[option_id] for option_id in canonical_ids]
        target_contract = hidden_by_id[record_id]["target_contract_id"]
        canonical_hit = target_contract in canonical_contracts
        for variant_id in config["trustedEvaluation"]["variantIds"]:
            name = f"{record_id}@@{variant_id}"
            output = completed[name]
            mapping = next(row for row in maps_by_id[record_id]["variants"] if row["variant_id"] == variant_id)
            contract_by_option = {row["option_id"]: row["capability_contract_id"] for row in mapping["mappings"]}
            proposal = output["normalized_proposal"]
            ranked_ids = proposal["ranked_option_ids"] if proposal["status"] == "RANKED" else []
            ranked_contracts = [contract_by_option[option_id] for option_id in ranked_ids]
            top1_hit = bool(ranked_contracts and ranked_contracts[0] == target_contract)
            top3_hit = target_contract in ranked_contracts
            if ranked_contracts:
                top1_cost = config["trustedEvaluation"]["top1QuestionCost"] + (
                    0.0 if top1_hit else config["trustedEvaluation"]["missThenGenericAdditionalCost"]
                )
                top3_cost = config["trustedEvaluation"]["top3QuestionCost"] + (
                    0.0 if top3_hit else config["trustedEvaluation"]["missThenGenericAdditionalCost"]
                )
            else:
                top1_cost = fixed_costs[target_contract]
                top3_cost = fixed_costs[target_contract]
            rows.append(
                {
                    "name": name, "record_id": record_id, "variant_id": variant_id,
                    "truth_kind": hidden_by_id[record_id]["truth_kind"],
                    "proposal_status": proposal["status"], "ranked_contract_ids": ranked_contracts,
                    "target_contract_id": target_contract, "top1_hit": top1_hit, "top3_hit": top3_hit,
                    "top1_cost": top1_cost, "top3_cost": top3_cost,
                    "top1_contract_agrees_with_canonical": bool(
                        (not ranked_contracts and not canonical_contracts)
                        or (ranked_contracts and canonical_contracts and ranked_contracts[0] == canonical_contracts[0])
                    ),
                    "top3_contract_set_jaccard_with_canonical": _jaccard(set(ranked_contracts), set(canonical_contracts)),
                    "target_top3_hit_disagrees_with_canonical": top3_hit != canonical_hit,
                    "structural_valid": output["structural_valid"],
                    "final_phase_maximum_tokens_hit": output["final_phase_maximum_tokens_hit"],
                    "target_retained": True, "final_exact": True, "false_terminal_from_model_output": False,
                }
            )
    gates = config["qualificationGates"]
    variants = []
    for variant_id in config["trustedEvaluation"]["variantIds"]:
        group = [row for row in rows if row["variant_id"] == variant_id]
        baseline = baseline_by_variant[variant_id]
        metrics = {
            "variant_id": variant_id, "count": len(group),
            "structural_validity_rate": sum(row["structural_valid"] for row in group) / len(group),
            "insufficient_rate": sum(row["proposal_status"] == "INSUFFICIENT" for row in group) / len(group),
            "final_phase_token_limit_hit_rate": sum(row["final_phase_maximum_tokens_hit"] for row in group) / len(group),
            "primary_top1_recall": sum(primary_weights[row["record_id"]] * row["top1_hit"] for row in group),
            "primary_top3_recall": sum(primary_weights[row["record_id"]] * row["top3_hit"] for row in group),
            "macro_top1_recall": sum(row["top1_hit"] for row in group) / len(group),
            "macro_top3_recall": sum(row["top3_hit"] for row in group) / len(group),
            "primary_top1_mean_cost": sum(primary_weights[row["record_id"]] * row["top1_cost"] for row in group),
            "primary_top3_mean_cost": sum(primary_weights[row["record_id"]] * row["top3_cost"] for row in group),
            "macro_top1_mean_cost": sum(row["top1_cost"] for row in group) / len(group),
            "macro_top3_mean_cost": sum(row["top3_cost"] for row in group) / len(group),
            "top1_contract_agreement_with_canonical": sum(row["top1_contract_agrees_with_canonical"] for row in group) / len(group),
            "mean_top3_contract_set_jaccard_with_canonical": sum(row["top3_contract_set_jaccard_with_canonical"] for row in group) / len(group),
            "target_top3_hit_disagreement_rate": sum(row["target_top3_hit_disagrees_with_canonical"] for row in group) / len(group),
            "transformed_CHAR_LAST_primary_top3_mean_cost": baseline["primary_top3_mean_cost"],
        }
        metrics["incremental_primary_improvement_over_transformed_CHAR_LAST"] = baseline["primary_top3_mean_cost"] - metrics["primary_top3_mean_cost"]
        metrics["primary_top3_recall_drop_from_canonical"] = gates["canonicalPrimaryTop3Recall"] - metrics["primary_top3_recall"]
        metrics["macro_top3_recall_drop_from_canonical"] = gates["canonicalMacroTop3Recall"] - metrics["macro_top3_recall"]
        metrics["primary_top3_cost_increase_from_canonical"] = metrics["primary_top3_mean_cost"] - gates["canonicalPrimaryTop3MeanCost"]
        qualification = {
            "primary_top3_recall_drop": metrics["primary_top3_recall_drop_from_canonical"] <= gates["maximumPerVariantPrimaryTop3RecallDrop"] + 1e-12,
            "macro_top3_recall_drop": metrics["macro_top3_recall_drop_from_canonical"] <= gates["maximumPerVariantMacroTop3RecallDrop"] + 1e-12,
            "primary_top3_cost_increase": metrics["primary_top3_cost_increase_from_canonical"] <= gates["maximumPerVariantPrimaryTop3CostIncrease"] + 1e-12,
            "top1_contract_agreement": metrics["top1_contract_agreement_with_canonical"] >= gates["minimumPerVariantTop1ContractAgreementWithCanonical"],
            "top3_contract_set_jaccard": metrics["mean_top3_contract_set_jaccard_with_canonical"] >= gates["minimumPerVariantMeanTop3ContractSetJaccardWithCanonical"],
            "target_hit_disagreement": metrics["target_top3_hit_disagreement_rate"] <= gates["maximumPerVariantTargetTop3HitDisagreementRate"],
            "incremental_over_transformed_CHAR_LAST": metrics["incremental_primary_improvement_over_transformed_CHAR_LAST"] >= gates["minimumPerVariantIncrementalPrimaryImprovementOverTransformedCHAR_LAST"] - 1e-12,
            "structural_validity": metrics["structural_validity_rate"] >= gates["minimumPerVariantObservedStructuralValidityRate"],
            "zero_final_truncation": metrics["final_phase_token_limit_hit_rate"] <= gates["maximumPerVariantFinalPhaseTokenLimitHitRate"],
            "target_retention": all(row["target_retained"] for row in group),
            "final_exactness": all(row["final_exact"] for row in group),
            "zero_false_terminal": sum(row["false_terminal_from_model_output"] for row in group) <= gates["maximumFalseTerminalFromModelOutput"],
        }
        metrics["qualification_gates"] = qualification
        metrics["qualified"] = all(qualification.values())
        variants.append(metrics)
    summary = {
        "observed_record_count": len(observed), "missing_record_count": len(missing),
        "variant_count": len(variants), "observed_record_variant_count": len(rows), "variants": variants,
        "all_variants_qualified": all(row["qualified"] for row in variants),
        "missing_insufficient_rate": 1.0,
        "target_retention_rate": 1.0, "final_exactness_after_trusted_answers": 1.0,
        "false_terminal_from_model_output": 0,
        "reasoning_phase_token_limit_hit_rate": sum(completed[row["name"]]["reasoning_phase_maximum_tokens_hit"] for row in rows) / len(rows),
        "reasoning_naturally_closed_rate": sum(completed[row["name"]]["reasoning_naturally_closed_within_budget"] for row in rows) / len(rows),
        "mean_reasoning_generated_tokens": sum(completed[row["name"]]["reasoning_phase_generated_token_count"] for row in rows) / len(rows),
        "mean_final_generated_tokens": sum(completed[row["name"]]["final_phase_generated_token_count"] for row in rows) / len(rows),
        **access,
    }
    summary["qualified"] = bool(
        summary["all_variants_qualified"]
        and summary["missing_insufficient_rate"] == 1.0
        and summary["target_retention_rate"] == 1.0 and summary["final_exactness_after_trusted_answers"] == 1.0
        and summary["false_terminal_from_model_output"] == 0
    )
    return {"summary": summary, "scored_records": rows}


def evaluate_access_gates(access: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["accessGates"]
    return {
        "development_language_read_count": access["development_language_read_count"] == gates["requiredDevelopmentLanguageReadCount"],
        "tokenizer_load_budget": access["tokenizer_load_count"] <= gates["maximumTokenizerLoadCount"],
        "model_load_budget": access["model_load_count"] <= gates["maximumModelLoadCount"],
        "reasoning_generation_budget": access["reasoning_phase_generation_count"] <= gates["maximumReasoningPhaseGenerationCount"],
        "final_generation_budget": access["final_phase_generation_count"] <= gates["maximumFinalPhaseGenerationCount"],
        "total_generation_budget": access["total_generation_count"] <= gates["maximumTotalGenerationCount"],
        "per_record_variant_generation_budget": access["maximum_generation_calls_per_observed_record_variant"] <= gates["maximumGenerationCallsPerObservedRecordVariant"],
        "zero_missing_generation": access["missing_record_variant_generation_count"] <= gates["maximumMissingRecordVariantGenerationCount"],
        "zero_retries": access["retry_count"] <= gates["maximumRetryCount"],
        "zero_manual_raw_inspection": access["manual_raw_response_inspection_count"] <= gates["maximumManualRawResponseInspectionCount"],
        "zero_persisted_raw": access["persisted_raw_response_count"] <= gates["maximumPersistedRawResponseCount"],
        "zero_protected": access["protected_language_read_count"] <= gates["maximumProtectedLanguageReadCount"],
        "zero_API": access["API_call_count"] <= gates["maximumAPICallCount"],
        "zero_training": access["training_run_count"] <= gates["maximumTrainingRunCount"],
        "zero_ontology_registration": access["ontology_registration_count"] <= gates["maximumOntologyRegistrationCount"],
        "zero_trusted_mutation": access["trusted_state_mutation_count"] <= gates["maximumTrustedStateMutationCount"],
        "zero_services": access["real_service_call_count"] <= gates["maximumRealServiceCallCount"],
        "zero_side_effects": access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"],
        "zero_execution": access["actual_execution_count"] <= gates["maximumActualExecutionCount"],
    }


__all__ = ["evaluate_access_gates", "evaluate_model"]
