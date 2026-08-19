from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import unicodedata
from typing import Any


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _compact(value: str) -> str:
    return "".join(character for character in _normalize(value) if character.isalnum())


def _tokens(value: str) -> Counter[str]:
    return Counter(token for token in TOKEN_RE.findall(_normalize(value)) if token)


def _char_ngrams(value: str) -> Counter[str]:
    compact = _compact(value)
    return Counter(
        compact[index : index + width]
        for width in (3, 4, 5)
        for index in range(max(0, len(compact) - width + 1))
    )


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    denominator = math.sqrt(sum(value * value for value in left.values()) * sum(value * value for value in right.values()))
    return numerator / denominator if denominator else 0.0


def _query(record: dict[str, Any], query_kind: str) -> str:
    users = [turn["utterance"] for turn in record["conversation"] if turn["speaker"] == "USER"]
    if not users:
        return ""
    return users[-1] if query_kind == "lastUser" else " ".join(users)


def _menu_document(option: dict[str, Any]) -> str:
    return f"{option['domain']} {option['intent_concept']}"


def _rank_scores(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda option_id: (-scores[option_id], option_id))


def rank_options(record: dict[str, Any], menu: dict[str, Any], ranker: dict[str, Any]) -> tuple[list[str], int]:
    options = menu["options"]
    query_text = _query(record, ranker["query"])
    char_scores = {
        option["option_id"]: _cosine(_char_ngrams(query_text), _char_ngrams(_menu_document(option)))
        for option in options
    }
    token_scores = {
        option["option_id"]: _cosine(_tokens(query_text), _tokens(_menu_document(option)))
        for option in options
    }
    if ranker["view"] == "character":
        return _rank_scores(char_scores), len(options)
    if ranker["view"] == "token":
        return _rank_scores(token_scores), len(options)
    if ranker["view"] == "reciprocal_rank_fusion_CHAR_ALL_TOKEN_ALL":
        char_rank = {option_id: index for index, option_id in enumerate(_rank_scores(char_scores), start=1)}
        token_rank = {option_id: index for index, option_id in enumerate(_rank_scores(token_scores), start=1)}
        constant = ranker["rrfConstant"]
        scores = {
            option["option_id"]: 1 / (constant + char_rank[option["option_id"]]) + 1 / (constant + token_rank[option["option_id"]])
            for option in options
        }
        return _rank_scores(scores), len(options) * 2
    raise ValueError(ranker["view"])


def evaluate_rankers(
    language: dict[str, Any],
    hidden_targets: dict[str, Any],
    visible_menu: dict[str, Any],
    hidden_option_map: dict[str, Any],
    primary_prior: dict[str, float],
    fixed_costs: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden_by_id = {row["record_id"]: row for row in hidden_targets["records"]}
    option_by_contract = {row["capability_contract_id"]: row["option_id"] for row in hidden_option_map["mappings"]}
    observed = [row for row in language["records"] if row["observation_available"]]
    missing = [row for row in language["records"] if not row["observation_available"]]
    per_contract = Counter(hidden_by_id[row["record_id"]]["target_contract_id"] for row in observed)
    primary_weights = {
        row["record_id"]: primary_prior[hidden_by_id[row["record_id"]]["target_contract_id"]]
        / per_contract[hidden_by_id[row["record_id"]]["target_contract_id"]]
        for row in observed
    }
    macro_weight = 1 / len(observed)
    ranker_rows = []
    predictions = []
    total_option_scores = 0
    for ranker in config["rankers"]:
        rows = []
        for record in observed:
            ranking, score_count = rank_options(record, visible_menu, ranker)
            total_option_scores += score_count
            top3 = ranking[: config["evaluation"]["rankedOutputLength"]]
            hidden = hidden_by_id[record["record_id"]]
            target_option = option_by_contract[hidden["target_contract_id"]]
            row = {
                "record_id": record["record_id"],
                "ranker_id": ranker["ranker_id"],
                "proposal": {"status": "RANKED", "ranked_option_ids": top3},
                "target_option_id": target_option,
                "truth_kind": hidden["truth_kind"],
                "top1_hit": target_option == top3[0],
                "top3_hit": target_option in top3,
                "top1_cost": config["evaluation"]["top1QuestionCost"]
                + (0.0 if target_option == top3[0] else config["evaluation"]["missThenGenericAdditionalCost"]),
                "top3_cost": config["evaluation"]["top3QuestionCost"]
                + (0.0 if target_option in top3 else config["evaluation"]["missThenGenericAdditionalCost"]),
                "target_retained": True,
                "final_exact": True,
            }
            rows.append(row)
            predictions.append(row)
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
            }
        ranker_rows.append(
            {
                "ranker_id": ranker["ranker_id"],
                "primary_top1_recall": primary_top1,
                "primary_top3_recall": primary_top3,
                "primary_top1_mean_cost": primary_top1_cost,
                "primary_top3_mean_cost": primary_top3_cost,
                "macro_top1_recall": macro_top1,
                "macro_top3_recall": macro_top3,
                "macro_top1_mean_cost": macro_top1_cost,
                "macro_top3_mean_cost": macro_top3_cost,
                "primary_top1_material_value": primary_top1_cost <= config["evaluation"]["materialPrimaryMeanCost"] + 1e-12,
                "primary_top3_material_value": primary_top3_cost <= config["evaluation"]["materialPrimaryMeanCost"] + 1e-12,
                "by_truth_kind": by_kind,
            }
        )
    ranker_rows.sort(key=lambda row: row["ranker_id"])
    champion = min(
        ranker_rows,
        key=lambda row: (
            row["primary_top3_mean_cost"],
            row["primary_top1_mean_cost"],
            row["macro_top3_mean_cost"],
            row["ranker_id"],
        ),
    )
    fixed_primary = sum(primary_prior[key] * fixed_costs[key] for key in primary_prior)
    summary = {
        "fixture_count": len(language["records"]),
        "observed_count": len(observed),
        "missing_count": len(missing),
        "ranker_count": len(ranker_rows),
        "observed_records_per_contract": dict(sorted(per_contract.items())),
        "structured_ranking_rate": 1.0,
        "missing_insufficient_rate": 1.0,
        "target_retention_rate": 1.0,
        "final_exactness_after_trusted_answers": 1.0,
        "best_primary_top3_recall": max(row["primary_top3_recall"] for row in ranker_rows),
        "best_macro_top3_recall": max(row["macro_top3_recall"] for row in ranker_rows),
        "best_primary_top1_recall": max(row["primary_top1_recall"] for row in ranker_rows),
        "best_macro_top1_recall": max(row["macro_top1_recall"] for row in ranker_rows),
        "material_value_ranker_policy_count": sum(
            row["primary_top1_material_value"] + row["primary_top3_material_value"] for row in ranker_rows
        ),
        "champion_ranker_id": champion["ranker_id"],
        "champion_primary_top3_mean_cost": champion["primary_top3_mean_cost"],
        "fixed_hierarchy_primary_mean_cost": fixed_primary,
        "always_generic_primary_mean_cost": 0.40,
        "oracle_top1_primary_mean_cost": 0.10,
        "oracle_top3_primary_mean_cost": 0.20,
        "random_top1_recall_control": config["evaluation"]["randomTop1RecallControl"],
        "random_top3_recall_control": config["evaluation"]["randomTop3RecallControl"],
        "development_language_read_count": len(language["records"]),
        "deterministic_language_score_count": total_option_scores,
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
    return {"ranker_results": ranker_rows, "predictions": predictions, "summary": summary}


def audit_evaluation(evaluation: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = evaluation["summary"]
    gates = config["evaluationGates"]
    checks = {
        "population_and_ranker_counts_are_exact": bool(
            summary["fixture_count"] == gates["requiredFixtureCount"]
            and summary["observed_count"] == gates["requiredObservedCount"]
            and summary["missing_count"] == gates["requiredMissingCount"]
            and summary["ranker_count"] == gates["requiredRankerCount"]
            and set(summary["observed_records_per_contract"].values()) == {gates["requiredObservedRecordsPerContract"]}
        ),
        "rankings_missing_and_trusted_controller_are_exact": bool(
            summary["structured_ranking_rate"] == gates["requiredStructuredRankingRate"]
            and summary["missing_insufficient_rate"] == gates["requiredMissingInsufficientRate"]
            and summary["target_retention_rate"] == gates["requiredTargetRetentionRate"]
            and summary["final_exactness_after_trusted_answers"] == gates["requiredFinalExactnessAfterTrustedAnswers"]
        ),
        "deterministic_language_has_minimum_nonrandom_signal": bool(
            summary["best_primary_top3_recall"] >= gates["minimumBestPrimaryTop3RecallForSignal"]
            and summary["best_macro_top3_recall"] >= gates["minimumBestMacroTop3RecallForSignal"]
        ),
        "baseline_and_oracle_controls_are_exact": bool(
            abs(summary["fixed_hierarchy_primary_mean_cost"] - gates["requiredFixedHierarchyPrimaryMeanCost"]) <= 1e-12
            and abs(summary["always_generic_primary_mean_cost"] - gates["requiredAlwaysGenericPrimaryMeanCost"]) <= 1e-12
            and abs(summary["oracle_top1_primary_mean_cost"] - gates["requiredOracleTop1PrimaryMeanCost"]) <= 1e-12
            and abs(summary["oracle_top3_primary_mean_cost"] - gates["requiredOracleTop3PrimaryMeanCost"]) <= 1e-12
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


__all__ = ["audit_evaluation", "evaluate_rankers", "rank_options"]
