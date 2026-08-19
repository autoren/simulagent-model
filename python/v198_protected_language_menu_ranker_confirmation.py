from __future__ import annotations

from typing import Any

from v194_deterministic_language_menu_rankers import evaluate_rankers
from v195_bounded_local_language_menu_ranker import evaluate_access_gates, evaluate_model


def evaluate_char_last(
    language: dict[str, Any], hidden_targets: dict[str, Any], visible_menu: dict[str, Any],
    hidden_option_map: dict[str, Any], primary_prior: dict[str, float], fixed_costs: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    comparator_config = {
        "rankers": [config["deterministicComparator"]],
        "evaluation": {
            "rankedOutputLength": config["deterministicComparator"]["rankedOutputLength"],
            "top1QuestionCost": config["trustedEvaluation"]["top1QuestionCost"],
            "top3QuestionCost": config["trustedEvaluation"]["top3QuestionCost"],
            "missThenGenericAdditionalCost": config["trustedEvaluation"]["missThenGenericAdditionalCost"],
            "materialPrimaryMeanCost": 0.36,
            "randomTop1RecallControl": 1 / 14,
            "randomTop3RecallControl": 3 / 14,
        },
    }
    return evaluate_rankers(
        language, hidden_targets, visible_menu, hidden_option_map, primary_prior, fixed_costs, comparator_config
    )


def evaluate_confirmation(
    completed: dict[str, dict[str, Any]], language: dict[str, Any], hidden_targets: dict[str, Any],
    hidden_option_map: dict[str, Any], primary_prior: dict[str, float], fixed_costs: dict[str, float],
    char_evaluation: dict[str, Any], access: dict[str, Any], development_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluated = evaluate_model(
        completed, language, hidden_targets, hidden_option_map, primary_prior, fixed_costs,
        char_evaluation["ranker_results"], access, config,
    )
    summary = evaluated["summary"]
    summary["development_primary_top3_recall"] = development_summary["primary_top3_recall"]
    summary["development_macro_top3_recall"] = development_summary["macro_top3_recall"]
    summary["development_primary_top3_mean_cost"] = development_summary["primary_top3_mean_cost"]
    summary["development_macro_top3_mean_cost"] = development_summary["macro_top3_mean_cost"]
    summary["confirmation_minus_development_primary_top3_recall"] = summary["primary_top3_recall"] - development_summary["primary_top3_recall"]
    summary["confirmation_minus_development_macro_top3_recall"] = summary["macro_top3_recall"] - development_summary["macro_top3_recall"]
    summary["confirmation_minus_development_primary_top3_cost"] = summary["primary_top3_mean_cost"] - development_summary["primary_top3_mean_cost"]
    summary["confirmation_minus_development_macro_top3_cost"] = summary["macro_top3_mean_cost"] - development_summary["macro_top3_mean_cost"]
    summary["confirmation_CHAR_LAST_primary_top3_recall"] = char_evaluation["ranker_results"][0]["primary_top3_recall"]
    summary["confirmation_CHAR_LAST_primary_top3_mean_cost"] = char_evaluation["ranker_results"][0]["primary_top3_mean_cost"]
    return evaluated


def evaluate_confirmation_access(access: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    checks = evaluate_access_gates(access, config)
    gates = config["accessGates"]
    checks.update({
        "exact_confirmation_language_read_count": access["confirmation_language_read_count"] == gates["requiredConfirmationLanguageReadCount"],
        "zero_unselected_protected_language_read_or_score": access["unselected_protected_language_read_or_score_count"] <= gates["maximumUnselectedProtectedLanguageReadOrScoreCount"],
    })
    return checks


__all__ = ["evaluate_char_last", "evaluate_confirmation", "evaluate_confirmation_access"]
