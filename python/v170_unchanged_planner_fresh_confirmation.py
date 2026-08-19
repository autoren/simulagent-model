from __future__ import annotations

from fractions import Fraction
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe
from v167_exact_evidence_gathering_planner import NONORACLE_POLICIES, evaluate_case, fraction_payload
from v167r1_history_action_metric_repair import corrected_case_is_history_dependent


def evaluate_fresh_population(
    states_artifact: dict[str, Any],
    eligible_artifact: dict[str, Any],
    v167_config: dict[str, Any],
) -> dict[str, Any]:
    state_by_id = {row["state_id"]: row for row in states_artifact["states"]}
    eligible_ids = list(eligible_artifact["state_ids"])
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    cases = []
    for state_id in eligible_ids:
        state = state_by_id[state_id]
        case = evaluate_case(
            state_id,
            state["candidate_ids"],
            state["candidate_ids"][0],
            state_id,
            universe,
            v167_config,
        )
        case["history_dependent_second_action"] = corrected_case_is_history_dependent(case)
        for key in ("target_candidate_id", "target_retained", "logical_target_group"):
            case.pop(key)
        cases.append(case)
    policies = (*NONORACLE_POLICIES, "oracle_class")
    mean_risk = {}
    for policy in policies:
        total = sum(
            Fraction(case["policy_expected_risk"][policy]["numerator"], case["policy_expected_risk"][policy]["denominator"])
            for case in cases
        )
        mean_risk[policy] = fraction_payload(total / len(cases))
    summary = {
        "case_count": len(cases),
        "candidate_count_values": sorted({case["candidate_count"] for case in cases}),
        "class_coverage_values": sorted({len(case["candidate_class_counts"]) for case in cases}),
        "eligible_membership_coverage": len(cases) / len(eligible_ids),
        "mean_expected_risk": mean_risk,
        "positive_value_of_information_case_count": sum(case["positive_value_of_information"] for case in cases),
        "history_dependent_second_action_case_count": sum(case["history_dependent_second_action"] for case in cases),
        "strict_improvement_over_optimal_open_loop_case_count": sum(case["strict_improvement_over_optimal_open_loop"] for case in cases),
        "unique_exact_bayes_root_queries": sorted({case["exact_bayes_root_query"] for case in cases}),
        "bayes_no_worse_than_every_nonoracle_baseline_case_rate": sum(case["bayes_no_worse_than_every_nonoracle_baseline"] for case in cases) / len(cases),
    }
    return {"cases": cases, "summary": summary}


def evaluate_integrity_gates(evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, bool]:
    summary, gates = evaluation["summary"], config["integrityGates"]
    return {
        "case_count": summary["case_count"] == gates["requiredCaseCount"],
        "candidate_count": summary["candidate_count_values"] == [gates["requiredCandidateCount"]],
        "class_coverage": summary["class_coverage_values"] == [gates["requiredClassCoverage"]],
        "eligible_membership_coverage": summary["eligible_membership_coverage"] == gates["requiredEligibleMembershipCoverage"],
        "pointwise_bayes_dominance": summary["bayes_no_worse_than_every_nonoracle_baseline_case_rate"] == gates["requiredBayesNoWorseThanEveryNonOracleBaselineCaseRate"],
        "exact_risk_reconstruction": gates["requiredExactRiskReconstruction"] == 1.0,
        "zero_disallowed_access": all(access[key] <= gates[maximum] for key, maximum in {
            "evaluation_record_count": "maximumEvaluationRecordCount", "manual_judgment_count": "maximumManualJudgmentCount",
            "model_load_count": "maximumModelLoadCount", "model_generation_count": "maximumModelGenerationCount",
            "API_call_count": "maximumAPICallCount", "training_run_count": "maximumTrainingRunCount",
            "ontology_registration_count": "maximumOntologyRegistrationCount", "trusted_state_mutation_count": "maximumTrustedStateMutationCount",
            "real_service_call_count": "maximumRealServiceCallCount", "external_side_effect_count": "maximumExternalSideEffectCount",
            "actual_execution_count": "maximumActualExecutionCount",
        }.items()),
    }


def evaluate_strong_thresholds(evaluation: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    summary, thresholds = evaluation["summary"], config["strongConfirmationThresholds"]
    return {
        "positive_value_prevalence": summary["positive_value_of_information_case_count"] >= thresholds["minimumPositiveValueOfInformationCaseCount"],
        "history_dependence_prevalence": summary["history_dependent_second_action_case_count"] >= thresholds["minimumHistoryDependentSecondActionCaseCount"],
        "adaptive_open_loop_advantage": summary["strict_improvement_over_optimal_open_loop_case_count"] >= thresholds["minimumStrictImprovementOverOptimalOpenLoopCaseCount"],
        "root_query_variation": len(summary["unique_exact_bayes_root_queries"]) >= thresholds["minimumUniqueExactBayesRootQueryCount"],
    }


__all__ = ["evaluate_fresh_population", "evaluate_integrity_gates", "evaluate_strong_thresholds"]
