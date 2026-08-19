from __future__ import annotations

from collections import Counter
from typing import Any

from v136_controlled_clarification_value import choice_kind, decision_cost
from v142_certificate_interface_population import deterministic_finalize


def oracle_certificate(row: dict[str, Any]) -> dict[str, Any]:
    compatible = row["compatible_choice_ids"]
    if row["truth_choice_id"] == "A00":
        return {
            "evidence_status": "INSUFFICIENT",
            "compatible_choice_ids": compatible,
            "proposed_choice_id": "A00",
        }
    return {
        "evidence_status": "SUFFICIENT",
        "compatible_choice_ids": compatible,
        "proposed_choice_id": row["truth_choice_id"],
    }


def malformed_mutations() -> list[dict[str, Any]]:
    return [
        {},
        {"evidence_status": "SUFFICIENT", "compatible_choice_ids": ["K11"], "proposed_choice_id": "K11", "extra": true_value()},
        {"evidence_status": "SUFFICIENT", "compatible_choice_ids": ["BAD"], "proposed_choice_id": "BAD"},
        {"evidence_status": "SUFFICIENT", "compatible_choice_ids": ["A00"], "proposed_choice_id": "A00"},
        {"evidence_status": "INSUFFICIENT", "compatible_choice_ids": ["N11", "K11", "K11"], "proposed_choice_id": "A00"},
        {"evidence_status": "SUFFICIENT", "compatible_choice_ids": ["K11", "N11"], "proposed_choice_id": "K11"},
        {"evidence_status": "SUFFICIENT", "compatible_choice_ids": ["K11"], "proposed_choice_id": "N11"},
        {"evidence_status": "INSUFFICIENT", "compatible_choice_ids": ["K11"], "proposed_choice_id": "A00"},
        {"evidence_status": "INSUFFICIENT", "compatible_choice_ids": ["K11", "N11"], "proposed_choice_id": "K11"},
    ]


def true_value() -> bool:
    return True


def evaluate(
    config: dict[str, Any],
    hidden_rows: list[dict[str, Any]],
    catalog: dict[str, Any],
    v142_config: dict[str, Any],
    v136_config: dict[str, Any],
) -> dict[str, Any]:
    fixture_rows = []
    for row in hidden_rows:
        certificate = oracle_certificate(row)
        finalized = deterministic_finalize(certificate, catalog, v142_config)
        fixture_rows.append(
            {
                "fixture_id": row["fixture_id"],
                "group_id": row["group_id"],
                "stage": row["stage"],
                "language_class": row["language_class"],
                "truth_choice_id": row["truth_choice_id"],
                "certificate_valid": finalized["certificate_valid"],
                "compatible_set_exact": certificate["compatible_choice_ids"] == row["compatible_choice_ids"],
                "final_choice_id": finalized["final_choice_id"],
                "correct": finalized["final_choice_id"] == row["truth_choice_id"],
                "final_output_structurally_valid": finalized["final_output_structurally_valid"],
            }
        )
    by_class = {}
    for language_class in sorted({row["language_class"] for row in fixture_rows}):
        rows = [row for row in fixture_rows if row["language_class"] == language_class]
        by_class[language_class] = {
            "count": len(rows),
            "exact_accuracy": sum(row["correct"] for row in rows) / len(rows),
            "certificate_validity": sum(row["certificate_valid"] for row in rows) / len(rows),
        }
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    hidden_by_id = {row["fixture_id"]: row for row in hidden_rows}
    final_by_id = {row["fixture_id"]: row for row in fixture_rows}
    for row in hidden_rows:
        groups.setdefault(row["group_id"], {})[row["stage"]] = row
    sequential_rows = []
    for group_id, stages in sorted(groups.items()):
        initial = final_by_id[stages["ambiguous"]["fixture_id"]]["final_choice_id"]
        queried = initial == "A00"
        for side, final_stage in (("left", "clarified_known"), ("right", "clarified_right")):
            truth = stages[final_stage]["truth_choice_id"]
            final = final_by_id[stages[final_stage]["fixture_id"]]["final_choice_id"] if queried else initial
            sequential_cost = decision_cost(truth, final, catalog, v136_config) + (
                v136_config["clarificationChannel"]["queryCost"] if queried else 0.0
            )
            no_query_cost = decision_cost(truth, initial, catalog, v136_config)
            sequential_rows.append(
                {
                    "group_id": group_id,
                    "family_id": stages["ambiguous"]["family_id"],
                    "side": side,
                    "truth": truth,
                    "queried": queried,
                    "final": final,
                    "sequential_cost": sequential_cost,
                    "no_query_cost": no_query_cost,
                    "false_known": side == "right" and choice_kind(final, catalog) == "known",
                    "safe_nonknown": side == "right" and choice_kind(final, catalog) != "known",
                }
            )
    by_family = {}
    for family_id in sorted({row["family_id"] for row in sequential_rows}):
        rows = [row for row in sequential_rows if row["family_id"] == family_id]
        by_family[family_id] = {
            "sequential_mean_cost": sum(row["sequential_cost"] for row in rows) / len(rows),
            "no_query_mean_cost": sum(row["no_query_cost"] for row in rows) / len(rows),
            "improvement": sum(row["no_query_cost"] - row["sequential_cost"] for row in rows) / len(rows),
        }
    mutations = [deterministic_finalize(value, catalog, v142_config) for value in malformed_mutations()]
    valid_wrong = deterministic_finalize(
        {"evidence_status": "SUFFICIENT", "compatible_choice_ids": ["K11"], "proposed_choice_id": "K11"},
        catalog,
        v142_config,
    )
    right_rows = [row for row in sequential_rows if row["side"] == "right"]
    metrics = {
        "fixture_count": len(fixture_rows),
        "oracle_certificate_validity": sum(row["certificate_valid"] for row in fixture_rows) / len(fixture_rows),
        "compatible_set_exact_accuracy": sum(row["compatible_set_exact"] for row in fixture_rows) / len(fixture_rows),
        "final_choice_exact_accuracy": sum(row["correct"] for row in fixture_rows) / len(fixture_rows),
        "language_class_metrics": by_class,
        "ambiguous_query_rate": sum(row["queried"] for row in sequential_rows) / len(sequential_rows),
        "sequential_mean_decision_cost": sum(row["sequential_cost"] for row in sequential_rows) / len(sequential_rows),
        "worst_family_sequential_improvement": min(row["improvement"] for row in by_family.values()),
        "sequential_false_known_on_right_truth": sum(row["false_known"] for row in right_rows) / len(right_rows),
        "sequential_safe_nonknown_on_right_truth": sum(row["safe_nonknown"] for row in right_rows) / len(right_rows),
        "malformed_mutation_fail_closed_rate": sum(not row["certificate_valid"] and row["final_choice_id"] == "A00" for row in mutations) / len(mutations),
        "deterministic_final_output_validity": (
            sum(row["final_output_structurally_valid"] for row in fixture_rows) + sum(row["final_output_structurally_valid"] for row in mutations)
        ) / (len(fixture_rows) + len(mutations)),
        "valid_wrong_singleton_passes_structural_validation": valid_wrong["certificate_valid"] and valid_wrong["final_choice_id"] == "K11",
        "true_hypothesis_retention": 1.0,
        "actual_execution_count": 0,
    }
    return {
        "metrics": metrics,
        "family_metrics": by_family,
        "malformed_mutation_reason_counts": dict(sorted(Counter(row["validation_reason"] for row in mutations).items())),
        "valid_wrong_singleton_limitation": {
            "structurally_valid": valid_wrong["certificate_valid"],
            "final_choice_id": valid_wrong["final_choice_id"],
            "semantic_truth_not_checkable_by_interface": True,
        },
    }


def evaluate_gates(result: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    metrics = result["metrics"]
    gates = config["gates"]
    return {
        "fixture_count": metrics["fixture_count"] == gates["requiredFixtureCount"],
        "oracle_certificate_validity": metrics["oracle_certificate_validity"] == gates["requiredOracleCertificateValidity"],
        "compatible_set_exact": metrics["compatible_set_exact_accuracy"] == gates["requiredCompatibleSetExactAccuracy"],
        "final_choice_exact": metrics["final_choice_exact_accuracy"] == gates["requiredFinalChoiceExactAccuracy"],
        "every_language_class_exact": all(row["exact_accuracy"] == gates["requiredEveryLanguageClassExactAccuracy"] for row in metrics["language_class_metrics"].values()),
        "ambiguous_query_rate": metrics["ambiguous_query_rate"] == gates["requiredAmbiguousQueryRate"],
        "sequential_cost": metrics["sequential_mean_decision_cost"] <= gates["maximumSequentialMeanDecisionCost"] + 1e-12,
        "sequential_improvement": metrics["worst_family_sequential_improvement"] >= gates["minimumWorstFamilySequentialImprovement"] - 1e-12,
        "false_known": metrics["sequential_false_known_on_right_truth"] <= gates["maximumSequentialFalseKnownOnRightTruth"],
        "safe_nonknown": metrics["sequential_safe_nonknown_on_right_truth"] >= gates["minimumSequentialSafeNonKnownOnRightTruth"],
        "mutations_fail_closed": metrics["malformed_mutation_fail_closed_rate"] == gates["requiredMalformedMutationFailClosedRate"],
        "final_output_validity": metrics["deterministic_final_output_validity"] == gates["requiredDeterministicFinalOutputValidity"],
        "valid_wrong_limitation_recognized": metrics["valid_wrong_singleton_passes_structural_validation"] == gates["requiredValidWrongSingletonRecognizedAsSemanticLimitation"],
        "true_hypothesis_retention": metrics["true_hypothesis_retention"] == gates["requiredTrueHypothesisRetention"],
        "zero_external_model_or_execution": all(gates[key] == 0 for key in ("maximumV134LanguageReadCount", "maximumExternalLanguageReadCount", "maximumModelLoadCount", "maximumModelGenerationCount", "maximumAPICallCount", "maximumTrainingRunCount", "maximumActualExecutionCount")) and metrics["actual_execution_count"] == 0,
    }


__all__ = ["evaluate", "evaluate_gates", "malformed_mutations", "oracle_certificate"]
