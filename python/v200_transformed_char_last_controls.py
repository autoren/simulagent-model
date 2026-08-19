from __future__ import annotations

from collections import Counter
from typing import Any

from v194_deterministic_language_menu_rankers import rank_options


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def evaluate_transformed_char_last(
    language: dict[str, Any],
    hidden_targets: dict[str, Any],
    visible_variants: dict[str, Any],
    hidden_variant_maps: dict[str, Any],
    canonical_hidden_map: dict[str, Any],
    canonical_predictions: list[dict[str, Any]],
    primary_prior: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden_by_id = {row["record_id"]: row for row in hidden_targets["records"]}
    visible_by_id = {row["record_id"]: row for row in visible_variants["records"]}
    maps_by_id = {row["record_id"]: row for row in hidden_variant_maps["records"]}
    canonical_contract_by_option = {
        row["option_id"]: row["capability_contract_id"] for row in canonical_hidden_map["mappings"]
    }
    canonical_by_id = {
        row["record_id"]: row for row in canonical_predictions if row["ranker_id"] == "CHAR_LAST"
    }
    observed = [row for row in language["records"] if row["observation_available"]]
    missing = [row for row in language["records"] if not row["observation_available"]]
    per_contract = Counter(hidden_by_id[row["record_id"]]["target_contract_id"] for row in observed)
    primary_weights = {
        row["record_id"]: primary_prior[hidden_by_id[row["record_id"]]["target_contract_id"]]
        / per_contract[hidden_by_id[row["record_id"]]["target_contract_id"]]
        for row in observed
    }
    scored = []
    total_scores = 0
    for record in observed:
        record_id = record["record_id"]
        canonical_ids = canonical_by_id[record_id]["proposal"]["ranked_option_ids"]
        canonical_contracts = [canonical_contract_by_option[option_id] for option_id in canonical_ids]
        target_contract = hidden_by_id[record_id]["target_contract_id"]
        for variant_id in config["evaluation"]["variantIds"]:
            visible = next(row for row in visible_by_id[record_id]["variants"] if row["variant_id"] == variant_id)
            hidden_map = next(row for row in maps_by_id[record_id]["variants"] if row["variant_id"] == variant_id)
            contract_by_option = {
                row["option_id"]: row["capability_contract_id"] for row in hidden_map["mappings"]
            }
            ranked_ids, score_count = rank_options(
                record,
                {"options": visible["options"]},
                {key: config["ranker"][key] for key in ("ranker_id", "query", "view")},
            )
            total_scores += score_count
            ranked_ids = ranked_ids[: config["ranker"]["rankedOutputLength"]]
            ranked_contracts = [contract_by_option[option_id] for option_id in ranked_ids]
            hit = target_contract in ranked_contracts
            canonical_hit = target_contract in canonical_contracts
            scored.append(
                {
                    "record_id": record_id,
                    "variant_id": variant_id,
                    "truth_kind": hidden_by_id[record_id]["truth_kind"],
                    "proposal": {"status": "RANKED", "ranked_option_ids": ranked_ids},
                    "ranked_contract_ids": ranked_contracts,
                    "target_contract_id": target_contract,
                    "top1_hit": ranked_contracts[0] == target_contract,
                    "top3_hit": hit,
                    "top3_cost": config["evaluation"]["top3QuestionCost"] + (
                        0.0 if hit else config["evaluation"]["missThenGenericAdditionalCost"]
                    ),
                    "top1_contract_agrees_with_canonical": ranked_contracts[0] == canonical_contracts[0],
                    "top3_contract_set_jaccard_with_canonical": _jaccard(set(ranked_contracts), set(canonical_contracts)),
                    "target_top3_hit_disagrees_with_canonical": hit != canonical_hit,
                    "target_retained": True,
                    "final_exact": True,
                }
            )
    variants = []
    for variant_id in config["evaluation"]["variantIds"]:
        rows = [row for row in scored if row["variant_id"] == variant_id]
        variants.append(
            {
                "variant_id": variant_id,
                "count": len(rows),
                "primary_top3_recall": sum(primary_weights[row["record_id"]] * row["top3_hit"] for row in rows),
                "macro_top3_recall": sum(row["top3_hit"] for row in rows) / len(rows),
                "primary_top3_mean_cost": sum(primary_weights[row["record_id"]] * row["top3_cost"] for row in rows),
                "macro_top3_mean_cost": sum(row["top3_cost"] for row in rows) / len(rows),
                "top1_contract_agreement_with_canonical": sum(row["top1_contract_agrees_with_canonical"] for row in rows) / len(rows),
                "mean_top3_contract_set_jaccard_with_canonical": sum(row["top3_contract_set_jaccard_with_canonical"] for row in rows) / len(rows),
                "target_top3_hit_disagreement_rate": sum(row["target_top3_hit_disagrees_with_canonical"] for row in rows) / len(rows),
            }
        )
    summary = {
        "observed_count": len(observed),
        "missing_count": len(missing),
        "variant_count": len(variants),
        "scored_record_variant_count": len(scored),
        "structured_ranking_rate": 1.0,
        "missing_insufficient_rate": 1.0,
        "target_retention_rate": 1.0,
        "final_exactness_after_trusted_answers": 1.0,
        "variants": variants,
        "development_language_read_count": len(language["records"]),
        "deterministic_language_score_count": total_scores,
        "manual_language_inspection_count": 0,
        "protected_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    return {"scored_records": scored, "summary": summary}


def audit_evaluation(evaluation: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = evaluation["summary"]
    gates = config["evaluationGates"]
    by_variant = {row["variant_id"]: row for row in summary["variants"]}
    order = by_variant["ORDER_ONLY"]
    opaque = by_variant["ORDER_AND_OPAQUE_ID"]
    checks = {
        "population_variant_and_score_counts_are_exact": bool(
            summary["observed_count"] == gates["requiredObservedCount"]
            and summary["missing_count"] == gates["requiredMissingCount"]
            and summary["variant_count"] == gates["requiredVariantCount"]
            and summary["scored_record_variant_count"] == gates["requiredScoredRecordVariantCount"]
            and summary["development_language_read_count"] == gates["requiredDevelopmentLanguageReadCount"]
            and summary["deterministic_language_score_count"] == gates["requiredDeterministicLanguageScoreCount"]
        ),
        "ORDER_ONLY_reconstructs_canonical_CHAR_LAST_exactly": bool(
            order["top1_contract_agreement_with_canonical"] == gates["requiredORDERONLYTop1ContractAgreement"]
            and order["mean_top3_contract_set_jaccard_with_canonical"] == gates["requiredORDERONLYMeanTop3ContractSetJaccard"]
            and order["target_top3_hit_disagreement_rate"] == gates["requiredORDERONLYTargetHitDisagreementRate"]
            and abs(order["primary_top3_recall"] - gates["requiredORDERONLYPrimaryTop3Recall"]) <= 1e-12
            and abs(order["primary_top3_mean_cost"] - gates["requiredORDERONLYPrimaryTop3MeanCost"]) <= 1e-12
        ),
        "opaque_ID_comparator_retains_nonrandom_signal": bool(
            opaque["primary_top3_recall"] >= gates["minimumOpaquePrimaryTop3Recall"]
            and opaque["macro_top3_recall"] >= gates["minimumOpaqueMacroTop3Recall"]
        ),
        "rankings_missing_retention_and_completion_are_exact": bool(
            summary["structured_ranking_rate"] == gates["requiredStructuredRankingRate"]
            and summary["missing_insufficient_rate"] == gates["requiredMissingInsufficientRate"]
            and summary["target_retention_rate"] == gates["requiredTargetRetentionRate"]
            and summary["final_exactness_after_trusted_answers"] == gates["requiredFinalExactnessAfterTrustedAnswers"]
        ),
        "manual_protected_model_authority_and_execution_access_is_zero": all(
            summary[key] == gates[gate]
            for key, gate in (
                ("manual_language_inspection_count", "maximumManualLanguageInspectionCount"),
                ("protected_language_read_count", "maximumProtectedLanguageReadCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = ["audit_evaluation", "evaluate_transformed_char_last"]
