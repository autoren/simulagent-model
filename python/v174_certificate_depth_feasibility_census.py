from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe
from v167_exact_evidence_gathering_planner import (
    available_queries,
    branch_probability,
    condition,
    fraction_payload,
    initial_belief,
)


TRUSTED_CLASSES = {"alias", "composition"}


def _target_weight(target: dict[str, Any]) -> Fraction:
    value = target["class_balanced_prior_weight"]
    return Fraction(value["numerator"], value["denominator"])


def candidate_classes(
    candidate_ids: tuple[str, ...] | list[str], universe: dict[str, dict[str, Any]]
) -> set[str]:
    return {universe[candidate_id]["expressibility_class"] for candidate_id in candidate_ids}


def condition_ids(
    candidate_ids: tuple[str, ...],
    queries: tuple[int, ...],
    target: dict[str, Any],
    universe: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    return tuple(
        candidate_id
        for candidate_id in candidate_ids
        if all(
            int(universe[candidate_id]["truth_table"][query])
            == int(target["truth_table"][query])
            for query in queries
        )
    )


def minimal_target_certificate(
    candidate_ids: list[str],
    target_candidate_id: str,
    universe: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target = universe[target_candidate_id]
    belief_stub = tuple((candidate_id, Fraction(1, len(candidate_ids))) for candidate_id in candidate_ids)
    queries = available_queries(belief_stub, universe)
    source = tuple(candidate_ids)
    for depth in range(len(queries) + 1):
        for subset in combinations(queries, depth):
            survivors = condition_ids(source, subset, target, universe)
            classes = candidate_classes(survivors, universe)
            if len(classes) == 1:
                return {
                    "minimal_depth": depth,
                    "query_subset": list(subset),
                    "certified_class": next(iter(classes)),
                    "surviving_candidate_count": len(survivors),
                }
    raise RuntimeError("full truth table must identify a class")


def adaptive_trusted_completion_curve(
    candidate_ids: list[str],
    universe: dict[str, dict[str, Any]],
    planner_config: dict[str, Any],
    horizons: list[int],
) -> dict[str, Any]:
    initial = initial_belief(candidate_ids, universe, planner_config)
    initial_queries = available_queries(initial, universe)

    @lru_cache(maxsize=None)
    def solve(
        belief: tuple[tuple[str, Fraction], ...],
        queries: tuple[int, ...],
        remaining: int,
    ) -> tuple[Fraction, tuple[Any, ...]]:
        classes = candidate_classes([candidate_id for candidate_id, _ in belief], universe)
        if len(classes) == 1 and next(iter(classes)) in TRUSTED_CLASSES:
            return Fraction(1), ("STOP_TRUSTED", next(iter(classes)))
        if remaining == 0 or not queries:
            return Fraction(0), ("STOP_DEFER",)
        best_value = Fraction(0)
        best_tree: tuple[Any, ...] = ("STOP_DEFER",)
        for query in queries:
            value = Fraction(0)
            children = []
            for outcome in (0, 1):
                probability = branch_probability(belief, query, outcome, universe)
                if probability:
                    child = condition(belief, query, outcome, universe)
                    child_value, child_tree = solve(
                        child,
                        tuple(item for item in queries if item != query),
                        remaining - 1,
                    )
                    value += probability * child_value
                    children.append((outcome, probability, child_tree))
            if value > best_value:
                best_value = value
                best_tree = ("QUERY", query, tuple(children))
        return best_value, best_tree

    values = {}
    roots = {}
    for horizon in horizons:
        value, tree = solve(initial, initial_queries, horizon)
        values[str(horizon)] = fraction_payload(value)
        roots[str(horizon)] = tree[1] if tree[0] == "QUERY" else None
    return {"trusted_completion": values, "root_query": roots}


def evaluate_feasibility(
    states_artifact: dict[str, Any],
    eligible_artifact: dict[str, Any],
    target_artifact: dict[str, Any],
    planner_config: dict[str, Any],
    horizons: list[int],
) -> dict[str, Any]:
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    state_by_id = {row["state_id"]: row for row in states_artifact["states"]}
    eligible_ids = list(eligible_artifact["state_ids"])
    targets_by_state: dict[str, list[dict[str, Any]]] = {state_id: [] for state_id in eligible_ids}
    for target in target_artifact["target_cases"]:
        targets_by_state[target["state_id"]].append(target)

    target_results = []
    state_results = []
    for state_id in eligible_ids:
        state = state_by_id[state_id]
        targets = sorted(targets_by_state[state_id], key=lambda row: row["target_case_id"])
        if sum((_target_weight(target) for target in targets), Fraction(0)) != 1:
            raise RuntimeError(f"weights do not normalize for {state_id}")
        curve = adaptive_trusted_completion_curve(
            state["candidate_ids"], universe, planner_config, horizons
        )
        upper = {str(horizon): Fraction(0) for horizon in horizons}
        provisional = {str(horizon): Fraction(0) for horizon in horizons}
        for target in targets:
            certificate = minimal_target_certificate(
                state["candidate_ids"], target["target_candidate_id"], universe
            )
            record = {
                **target,
                **certificate,
                "certificate_valid": certificate["certified_class"] == target["target_class"],
            }
            target_results.append(record)
            weight = _target_weight(target)
            for horizon in horizons:
                if certificate["minimal_depth"] <= horizon:
                    if target["target_class"] in TRUSTED_CLASSES:
                        upper[str(horizon)] += weight
                    elif target["target_class"] == "provisional_primitive":
                        provisional[str(horizon)] += weight
        state_results.append(
            {
                "state_id": state_id,
                "target_count": len(targets),
                "adaptive_trusted_completion": curve["trusted_completion"],
                "adaptive_root_query": curve["root_query"],
                "target_informed_trusted_upper_bound": {
                    key: fraction_payload(value) for key, value in upper.items()
                },
                "target_informed_provisional_certificate_probability": {
                    key: fraction_payload(value) for key, value in provisional.items()
                },
            }
        )

    target_results.sort(key=lambda row: row["target_case_id"])
    state_results.sort(key=lambda row: row["state_id"])
    target_by_id = {row["target_case_id"]: row for row in target_artifact["target_cases"]}
    class_depth: dict[str, dict[str, Any]] = {}
    for target_class in ("alias", "composition", "provisional_primitive"):
        rows = [row for row in target_results if row["target_class"] == target_class]
        total_weight = Fraction(len(eligible_ids), 3)
        distribution = {}
        for depth in range(max(horizons) + 1):
            mass = sum(
                _target_weight(target_by_id[row["target_case_id"]])
                for row in rows
                if row["minimal_depth"] == depth
            ) / total_weight
            distribution[str(depth)] = fraction_payload(mass)
        mean_depth = sum(
            _target_weight(target_by_id[row["target_case_id"]]) * row["minimal_depth"]
            for row in rows
        ) / total_weight
        class_depth[target_class] = {
            "conditional_depth_distribution": distribution,
            "conditional_mean_depth": fraction_payload(mean_depth),
            "raw_target_count": len(rows),
        }

    aggregate_curve = {}
    aggregate_upper = {}
    aggregate_provisional = {}
    for horizon in horizons:
        key = str(horizon)
        aggregate_curve[key] = fraction_payload(
            sum(
                Fraction(
                    row["adaptive_trusted_completion"][key]["numerator"],
                    row["adaptive_trusted_completion"][key]["denominator"],
                )
                for row in state_results
            )
            / len(state_results)
        )
        aggregate_upper[key] = fraction_payload(
            sum(
                Fraction(
                    row["target_informed_trusted_upper_bound"][key]["numerator"],
                    row["target_informed_trusted_upper_bound"][key]["denominator"],
                )
                for row in state_results
            )
            / len(state_results)
        )
        aggregate_provisional[key] = fraction_payload(
            sum(
                Fraction(
                    row["target_informed_provisional_certificate_probability"][key]["numerator"],
                    row["target_informed_provisional_certificate_probability"][key]["denominator"],
                )
                for row in state_results
            )
            / len(state_results)
        )

    depth_counts = Counter(row["minimal_depth"] for row in target_results)
    depths = [row["minimal_depth"] for row in target_results]
    validity = [row["certificate_valid"] for row in target_results]
    minimality = []
    for row in target_results:
        state = state_by_id[row["state_id"]]
        target = universe[row["target_candidate_id"]]
        smaller_exists = False
        available = tuple(index for index in range(8) if index not in {constraint["valuation_index"] for constraint in state["constraints"]})
        for depth in range(row["minimal_depth"]):
            if any(
                len(candidate_classes(condition_ids(tuple(state["candidate_ids"]), subset, target, universe), universe)) == 1
                for subset in combinations(available, depth)
            ):
                smaller_exists = True
                break
        minimality.append(not smaller_exists)

    adaptive_values = [
        Fraction(aggregate_curve[str(horizon)]["numerator"], aggregate_curve[str(horizon)]["denominator"])
        for horizon in horizons
    ]
    upper_values = [
        Fraction(aggregate_upper[str(horizon)]["numerator"], aggregate_upper[str(horizon)]["denominator"])
        for horizon in horizons
    ]
    summary = {
        "state_count": len(state_results),
        "target_count": len(target_results),
        "target_coverage": len({row["target_case_id"] for row in target_results}) / len(target_artifact["target_cases"]),
        "prior_weight_normalization_rate": 1.0,
        "certificate_validity_rate": sum(validity) / len(validity),
        "certificate_minimality_rate": sum(minimality) / len(minimality),
        "full_depth_certifiability_rate": sum(depth <= max(horizons) for depth in depths) / len(depths),
        "minimal_depth_raw_counts": {str(key): value for key, value in sorted(depth_counts.items())},
        "class_certificate_depth": class_depth,
        "adaptive_trusted_completion_by_horizon": aggregate_curve,
        "target_informed_trusted_upper_bound_by_horizon": aggregate_upper,
        "target_informed_provisional_certificate_probability_by_horizon": aggregate_provisional,
        "horizon_monotonicity_rate": float(all(left <= right for left, right in zip(adaptive_values, adaptive_values[1:]))),
        "adaptive_no_greater_than_target_informed_rate": float(all(left <= right for left, right in zip(adaptive_values, upper_values))),
        "zero_horizon_trusted_completion": fraction_payload(adaptive_values[0]),
    }
    return {"target_results": target_results, "state_results": state_results, "summary": summary}


def evaluate_gates(
    evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, bool]:
    summary = evaluation["summary"]
    gates = config["feasibilityGates"]
    zero = summary["zero_horizon_trusted_completion"]
    return {
        "state_count": summary["state_count"] == gates["requiredStateCount"],
        "target_count": summary["target_count"] == gates["requiredTargetCount"],
        "target_coverage": summary["target_coverage"] == gates["requiredTargetCoverage"],
        "prior_weight_normalization": summary["prior_weight_normalization_rate"] == gates["requiredPriorWeightNormalization"],
        "certificate_validity": summary["certificate_validity_rate"] == gates["requiredCertificateValidity"],
        "certificate_minimality": summary["certificate_minimality_rate"] == gates["requiredCertificateMinimality"],
        "full_depth_certifiability": summary["full_depth_certifiability_rate"] == gates["requiredFullDepthCertifiability"],
        "horizon_monotonicity": summary["horizon_monotonicity_rate"] == gates["requiredHorizonMonotonicity"],
        "adaptive_bounded_by_target_informed": summary["adaptive_no_greater_than_target_informed_rate"] == gates["requiredAdaptiveNoGreaterThanTargetInformedUpperBound"],
        "zero_horizon_trusted_completion": Fraction(zero["numerator"], zero["denominator"]) == gates["requiredZeroHorizonTrustedCompletion"],
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
                "trusted_state_mutation_count": "maximumTrustedStateMutationCount",
                "real_service_call_count": "maximumRealServiceCallCount",
                "external_side_effect_count": "maximumExternalSideEffectCount",
                "actual_execution_count": "maximumActualExecutionCount",
            }.items()
        ),
    }


__all__ = [
    "TRUSTED_CLASSES",
    "adaptive_trusted_completion_curve",
    "candidate_classes",
    "condition_ids",
    "evaluate_feasibility",
    "evaluate_gates",
    "minimal_target_certificate",
]
