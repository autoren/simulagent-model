from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe
from v167_exact_evidence_gathering_planner import fraction_payload
from v174_certificate_depth_feasibility_census import (
    TRUSTED_CLASSES,
    adaptive_trusted_completion_curve,
    candidate_classes,
    condition_ids,
    minimal_target_certificate,
)
from v178_one_corruption_robust_certificate_feasibility import remaining_queries


RawHistory = tuple[tuple[int, int, int], ...]


def _target_weight(target: dict[str, Any]) -> Fraction:
    value = target["class_balanced_prior_weight"]
    return Fraction(value["numerator"], value["denominator"])


def repeated_history(
    queries: tuple[int, ...],
    target: dict[str, Any],
    flip: tuple[int, int] | None,
) -> RawHistory:
    return tuple(
        (
            query,
            repetition,
            int(target["truth_table"][query])
            ^ int(flip == (query, repetition)),
        )
        for query in queries
        for repetition in range(3)
    )


def majority_decode(history: RawHistory) -> tuple[tuple[int, int], ...]:
    queries = sorted({query for query, _, _ in history})
    return tuple(
        (
            query,
            int(
                sum(
                    observed
                    for item, _, observed in history
                    if item == query
                )
                >= 2
            ),
        )
        for query in queries
    )


def raw_robust_survivors(
    candidate_ids: list[str] | tuple[str, ...],
    history: RawHistory,
    universe: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    return tuple(
        candidate_id
        for candidate_id in candidate_ids
        if sum(
            int(universe[candidate_id]["truth_table"][query]) != observed
            for query, _, observed in history
        )
        <= 1
    )


def clean_decoded_survivors(
    candidate_ids: list[str] | tuple[str, ...],
    history: RawHistory,
    universe: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    decoded = majority_decode(history)
    return tuple(
        candidate_id
        for candidate_id in candidate_ids
        if all(
            int(universe[candidate_id]["truth_table"][query]) == observed
            for query, observed in decoded
        )
    )


def evaluate_feasibility(
    states_artifact: dict[str, Any],
    eligible_artifact: dict[str, Any],
    targets_artifact: dict[str, Any],
    planner_config: dict[str, Any],
    horizons: list[int],
) -> dict[str, Any]:
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    state_by_id = {row["state_id"]: row for row in states_artifact["states"]}
    eligible_ids = list(eligible_artifact["state_ids"])
    targets_by_state = {state_id: [] for state_id in eligible_ids}
    for target in targets_artifact["target_cases"]:
        targets_by_state[target["state_id"]].append(target)

    target_results = []
    state_results = []
    decode_checks = []
    equivalence_checks = []
    false_routes = 0
    scenario_count = 0
    for state_id in eligible_ids:
        state = state_by_id[state_id]
        targets = sorted(
            targets_by_state[state_id], key=lambda row: row["target_case_id"]
        )
        queries = remaining_queries(state)
        if sum((_target_weight(row) for row in targets), Fraction(0)) != 1:
            raise RuntimeError(f"target weights do not normalize for {state_id}")
        curve = adaptive_trusted_completion_curve(
            state["candidate_ids"], universe, planner_config, horizons
        )
        upper = {str(horizon): Fraction(0) for horizon in horizons}
        for target_case in targets:
            target = universe[target_case["target_candidate_id"]]
            certificate = minimal_target_certificate(
                state["candidate_ids"], target_case["target_candidate_id"], universe
            )
            subset = tuple(certificate["query_subset"])
            witness_checks = []
            for flip in (None, *((query, repetition) for query in subset for repetition in range(3))):
                history = repeated_history(subset, target, flip)
                decoded = majority_decode(history)
                witness_checks.append(
                    all(
                        observed == int(target["truth_table"][query])
                        for query, observed in decoded
                    )
                    and raw_robust_survivors(
                        state["candidate_ids"], history, universe
                    )
                    == clean_decoded_survivors(
                        state["candidate_ids"], history, universe
                    )
                )
            minimal = not any(
                len(
                    candidate_classes(
                        condition_ids(
                            tuple(state["candidate_ids"]),
                            smaller,
                            target,
                            universe,
                        ),
                        universe,
                    )
                )
                == 1
                for depth in range(certificate["minimal_depth"])
                for smaller in combinations(queries, depth)
            )
            record = {
                **target_case,
                **certificate,
                "raw_inspection_count": 3 * certificate["minimal_depth"],
                "certificate_valid": certificate["certified_class"]
                == target_case["target_class"]
                and all(witness_checks),
                "certificate_minimal": minimal,
            }
            target_results.append(record)
            if target_case["target_class"] in TRUSTED_CLASSES:
                for horizon in horizons:
                    if certificate["minimal_depth"] <= horizon:
                        upper[str(horizon)] += _target_weight(target_case)

            flips = (
                None,
                *((query, repetition) for query in queries for repetition in range(3)),
            )
            for flip in flips:
                history = repeated_history(queries, target, flip)
                decoded = majority_decode(history)
                decode_checks.append(
                    all(
                        observed == int(target["truth_table"][query])
                        for query, observed in decoded
                    )
                )
                raw = raw_robust_survivors(
                    state["candidate_ids"], history, universe
                )
                clean = clean_decoded_survivors(
                    state["candidate_ids"], history, universe
                )
                equivalence_checks.append(raw == clean)
                classes = candidate_classes(raw, universe)
                if (
                    len(classes) == 1
                    and next(iter(classes)) in TRUSTED_CLASSES
                    and next(iter(classes)) != target_case["target_class"]
                ):
                    false_routes += 1
                scenario_count += 1
        state_results.append(
            {
                "state_id": state_id,
                "target_count": len(targets),
                "remaining_measurement_block_count": len(queries),
                "adaptive_worst_case_trusted_completion": curve[
                    "trusted_completion"
                ],
                "adaptive_root_query": curve["root_query"],
                "target_informed_trusted_upper_bound": {
                    key: fraction_payload(value) for key, value in upper.items()
                },
            }
        )

    target_results.sort(key=lambda row: row["target_case_id"])
    state_results.sort(key=lambda row: row["state_id"])
    adaptive_curve = {}
    upper_curve = {}
    for horizon in horizons:
        key = str(horizon)
        adaptive_curve[key] = fraction_payload(
            sum(
                Fraction(
                    row["adaptive_worst_case_trusted_completion"][key]["numerator"],
                    row["adaptive_worst_case_trusted_completion"][key]["denominator"],
                )
                for row in state_results
            )
            / len(state_results)
        )
        upper_curve[key] = fraction_payload(
            sum(
                Fraction(
                    row["target_informed_trusted_upper_bound"][key]["numerator"],
                    row["target_informed_trusted_upper_bound"][key]["denominator"],
                )
                for row in state_results
            )
            / len(state_results)
        )
    adaptive_values = [
        Fraction(adaptive_curve[str(h)]["numerator"], adaptive_curve[str(h)]["denominator"])
        for h in horizons
    ]
    upper_values = [
        Fraction(upper_curve[str(h)]["numerator"], upper_curve[str(h)]["denominator"])
        for h in horizons
    ]
    depth_counts = Counter(row["minimal_depth"] for row in target_results)
    raw_counts = Counter(row["raw_inspection_count"] for row in target_results)
    summary = {
        "state_count": len(state_results),
        "target_count": len(target_results),
        "target_coverage": len({row["target_case_id"] for row in target_results})
        / len(targets_artifact["target_cases"]),
        "prior_weight_normalization_rate": 1.0,
        "full_measurement_adversarial_scenario_count": scenario_count,
        "majority_decode_exactness_rate": sum(decode_checks) / len(decode_checks),
        "robust_clean_version_space_equivalence_rate": sum(equivalence_checks)
        / len(equivalence_checks),
        "certificate_validity_rate": sum(
            row["certificate_valid"] for row in target_results
        )
        / len(target_results),
        "certificate_minimality_rate": sum(
            row["certificate_minimal"] for row in target_results
        )
        / len(target_results),
        "minimal_block_depth_counts": {
            str(key): value for key, value in sorted(depth_counts.items())
        },
        "minimal_raw_inspection_counts": {
            str(key): value for key, value in sorted(raw_counts.items())
        },
        "adaptive_worst_case_trusted_completion_by_block_horizon": adaptive_curve,
        "target_informed_trusted_upper_bound_by_block_horizon": upper_curve,
        "horizon_monotonicity_rate": float(
            all(left <= right for left, right in zip(adaptive_values, adaptive_values[1:]))
        ),
        "adaptive_no_greater_than_target_informed_rate": float(
            all(left <= right for left, right in zip(adaptive_values, upper_values))
        ),
        "false_trusted_route_probability": 0.0
        if false_routes == 0
        else false_routes / scenario_count,
    }
    return {
        "target_results": target_results,
        "state_results": state_results,
        "summary": summary,
    }


def evaluate_gates(
    evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, bool]:
    summary = evaluation["summary"]
    gates = config["feasibilityGates"]
    return {
        "state_count": summary["state_count"] == gates["requiredStateCount"],
        "target_count": summary["target_count"] == gates["requiredTargetCount"],
        "target_coverage": summary["target_coverage"] == gates["requiredTargetCoverage"],
        "prior_normalization": summary["prior_weight_normalization_rate"]
        == gates["requiredPriorWeightNormalization"],
        "scenario_count": summary["full_measurement_adversarial_scenario_count"]
        == gates["requiredFullMeasurementAdversarialScenarioCount"],
        "majority_decode_exactness": summary["majority_decode_exactness_rate"]
        == gates["requiredMajorityDecodeExactness"],
        "robust_clean_equivalence": summary[
            "robust_clean_version_space_equivalence_rate"
        ]
        == gates["requiredRobustCleanVersionSpaceEquivalence"],
        "certificate_validity": summary["certificate_validity_rate"]
        == gates["requiredCertificateValidity"],
        "certificate_minimality": summary["certificate_minimality_rate"]
        == gates["requiredCertificateMinimality"],
        "horizon_monotonicity": summary["horizon_monotonicity_rate"]
        == gates["requiredHorizonMonotonicity"],
        "adaptive_bounded_by_target_informed": summary[
            "adaptive_no_greater_than_target_informed_rate"
        ]
        == gates["requiredAdaptiveNoGreaterThanTargetInformedUpperBound"],
        "zero_false_trusted_route": summary["false_trusted_route_probability"]
        == gates["requiredFalseTrustedRouteProbability"],
        "zero_disallowed_access": all(
            access[key] <= gates[maximum]
            for key, maximum in {
                "planner_risk_or_cost_score_count": "maximumPlannerRiskOrCostScoreCount",
                "sandbox_transaction_count": "maximumSandboxTransactionCount",
                "evaluation_record_count": "maximumEvaluationRecordCount",
                "manual_judgment_count": "maximumManualJudgmentCount",
                "model_load_count": "maximumModelLoadCount",
                "model_generation_count": "maximumModelGenerationCount",
                "API_call_count": "maximumAPICallCount",
                "training_run_count": "maximumTrainingRunCount",
                "ontology_registration_count": "maximumOntologyRegistrationCount",
                "trusted_real_state_mutation_count": "maximumTrustedRealStateMutationCount",
                "real_service_call_count": "maximumRealServiceCallCount",
                "external_side_effect_count": "maximumExternalSideEffectCount",
                "actual_execution_count": "maximumActualExecutionCount",
            }.items()
        ),
    }


__all__ = [
    "clean_decoded_survivors",
    "evaluate_feasibility",
    "evaluate_gates",
    "majority_decode",
    "raw_robust_survivors",
    "repeated_history",
]
