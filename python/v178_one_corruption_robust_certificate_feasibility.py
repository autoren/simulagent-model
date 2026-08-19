from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations, product
import hashlib
import json
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe
from v167_exact_evidence_gathering_planner import fraction_payload


TRUSTED_CLASSES = {"alias", "composition"}
PolicyTree = tuple[Any, ...]
History = tuple[tuple[int, int], ...]


def _target_weight(target: dict[str, Any]) -> Fraction:
    value = target["class_balanced_prior_weight"]
    return Fraction(value["numerator"], value["denominator"])


def remaining_queries(state: dict[str, Any]) -> tuple[int, ...]:
    observed = {row["valuation_index"] for row in state["constraints"]}
    return tuple(index for index in range(8) if index not in observed)


def robust_survivors(
    candidate_ids: tuple[str, ...] | list[str],
    history: History,
    universe: dict[str, dict[str, Any]],
    corruption_budget: int = 1,
) -> tuple[str, ...]:
    return tuple(
        candidate_id
        for candidate_id in candidate_ids
        if sum(
            int(universe[candidate_id]["truth_table"][query]) != observed
            for query, observed in history
        )
        <= corruption_budget
    )


def robust_route_class(
    candidate_ids: tuple[str, ...] | list[str],
    history: History,
    universe: dict[str, dict[str, Any]],
) -> str | None:
    survivors = robust_survivors(candidate_ids, history, universe)
    classes = {
        universe[candidate_id]["expressibility_class"]
        for candidate_id in survivors
    }
    if len(classes) == 1:
        return next(iter(classes))
    return None


def corrupted_history(
    subset: tuple[int, ...],
    target: dict[str, Any],
    flip_query: int | None,
) -> History:
    return tuple(
        (
            query,
            int(target["truth_table"][query]) ^ int(query == flip_query),
        )
        for query in subset
    )


def subset_is_robust_class_certificate(
    candidate_ids: list[str],
    subset: tuple[int, ...],
    target: dict[str, Any],
    universe: dict[str, dict[str, Any]],
) -> bool:
    target_class = target["expressibility_class"]
    for flip_query in (None, *subset):
        history = corrupted_history(subset, target, flip_query)
        survivors = robust_survivors(candidate_ids, history, universe)
        classes = {
            universe[candidate_id]["expressibility_class"]
            for candidate_id in survivors
        }
        if classes != {target_class}:
            return False
    return True


def minimal_robust_certificate(
    candidate_ids: list[str],
    queries: tuple[int, ...],
    target_candidate_id: str,
    universe: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target = universe[target_candidate_id]
    for depth in range(len(queries) + 1):
        for subset in combinations(queries, depth):
            if subset_is_robust_class_certificate(
                candidate_ids, subset, target, universe
            ):
                no_flip = corrupted_history(subset, target, None)
                survivors = robust_survivors(
                    candidate_ids, no_flip, universe
                )
                return {
                    "certifiable": True,
                    "minimal_depth": depth,
                    "query_subset": list(subset),
                    "certified_class": target["expressibility_class"],
                    "no_flip_surviving_candidate_count": len(survivors),
                }
    return {
        "certifiable": False,
        "minimal_depth": None,
        "query_subset": None,
        "certified_class": None,
        "no_flip_surviving_candidate_count": None,
    }


def enumerate_policy_trees(
    candidate_ids: list[str],
    queries: tuple[int, ...],
    history: History,
    horizon: int,
    universe: dict[str, dict[str, Any]],
) -> tuple[PolicyTree, ...]:
    route_class = robust_route_class(candidate_ids, history, universe)
    if route_class is not None:
        disposition = (
            f"route_{route_class}"
            if route_class in TRUSTED_CLASSES
            else "defer_provisional"
        )
        return (("STOP", disposition),)
    if horizon == 0 or not queries:
        return (("STOP", "defer_mixed"),)
    trees: list[PolicyTree] = []
    for query in queries:
        remaining = tuple(item for item in queries if item != query)
        children = []
        for observed in (0, 1):
            child_history = history + ((query, observed),)
            children.append(
                enumerate_policy_trees(
                    candidate_ids,
                    remaining,
                    child_history,
                    horizon - 1,
                    universe,
                )
            )
        for child_zero, child_one in product(*children):
            trees.append(("QUERY", query, child_zero, child_one))
    return tuple(trees)


def trace_policy(
    tree: PolicyTree,
    target: dict[str, Any],
    flip_query: int | None,
) -> tuple[History, str]:
    history: History = ()
    node = tree
    while node[0] == "QUERY":
        query = int(node[1])
        observed = int(target["truth_table"][query]) ^ int(
            query == flip_query
        )
        history = history + ((query, observed),)
        node = node[2 + observed]
    return history, str(node[1])


def best_target_blind_policy(
    state: dict[str, Any],
    targets: list[dict[str, Any]],
    horizon: int,
    universe: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    queries = remaining_queries(state)
    trees = enumerate_policy_trees(
        state["candidate_ids"], queries, (), horizon, universe
    )
    best_mass = Fraction(-1)
    best_tree: PolicyTree | None = None
    best_success_ids: tuple[str, ...] = ()
    best_false_route_count = 0
    for tree in trees:
        success_ids = []
        false_route_count = 0
        for target_case in targets:
            target = universe[target_case["target_candidate_id"]]
            scenario_success = []
            for flip_query in (None, *queries):
                history, _ = trace_policy(tree, target, flip_query)
                route_class = robust_route_class(
                    state["candidate_ids"], history, universe
                )
                if (
                    route_class in TRUSTED_CLASSES
                    and route_class != target_case["target_class"]
                ):
                    false_route_count += 1
                scenario_success.append(
                    route_class == target_case["target_class"]
                    and route_class in TRUSTED_CLASSES
                )
            if all(scenario_success):
                success_ids.append(target_case["target_case_id"])
        success_set = set(success_ids)
        mass = sum(
            (
                _target_weight(target_case)
                for target_case in targets
                if target_case["target_case_id"] in success_set
            ),
            Fraction(0),
        )
        tree_key = repr(tree)
        best_key = repr(best_tree) if best_tree is not None else ""
        if mass > best_mass or (mass == best_mass and tree_key < best_key):
            best_mass = mass
            best_tree = tree
            best_success_ids = tuple(sorted(success_ids))
            best_false_route_count = false_route_count
    if best_tree is None:
        raise RuntimeError("at least one policy tree must exist")
    return {
        "trusted_completion": best_mass,
        "policy_tree": best_tree,
        "root_query": best_tree[1] if best_tree[0] == "QUERY" else None,
        "robust_success_target_ids": best_success_ids,
        "policy_tree_count": len(trees),
        "false_trusted_route_count": best_false_route_count,
    }


def evaluate_feasibility(
    states_artifact: dict[str, Any],
    eligible_artifact: dict[str, Any],
    targets_artifact: dict[str, Any],
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
    containment_checks = []
    false_route_count = 0
    adversarial_scenario_count = 0
    for state_id in eligible_ids:
        state = state_by_id[state_id]
        targets = sorted(
            targets_by_state[state_id], key=lambda row: row["target_case_id"]
        )
        queries = remaining_queries(state)
        if sum((_target_weight(row) for row in targets), Fraction(0)) != 1:
            raise RuntimeError(f"target weights do not normalize for {state_id}")
        upper = {str(horizon): Fraction(0) for horizon in horizons}
        for target_case in targets:
            target = universe[target_case["target_candidate_id"]]
            certificate = minimal_robust_certificate(
                state["candidate_ids"],
                queries,
                target_case["target_candidate_id"],
                universe,
            )
            record = {**target_case, **certificate}
            if certificate["certifiable"]:
                subset = tuple(certificate["query_subset"])
                record["certificate_valid"] = (
                    certificate["certified_class"]
                    == target_case["target_class"]
                    and subset_is_robust_class_certificate(
                        state["candidate_ids"], subset, target, universe
                    )
                )
                record["certificate_minimal"] = not any(
                    subset_is_robust_class_certificate(
                        state["candidate_ids"], smaller, target, universe
                    )
                    for depth in range(certificate["minimal_depth"])
                    for smaller in combinations(queries, depth)
                )
                if target_case["target_class"] in TRUSTED_CLASSES:
                    for horizon in horizons:
                        if certificate["minimal_depth"] <= horizon:
                            upper[str(horizon)] += _target_weight(target_case)
            else:
                record["certificate_valid"] = None
                record["certificate_minimal"] = None
            for flip_query in (None, *queries):
                full_history = corrupted_history(queries, target, flip_query)
                containment_checks.append(
                    target_case["target_candidate_id"]
                    in robust_survivors(
                        state["candidate_ids"], full_history, universe
                    )
                )
                adversarial_scenario_count += 1
            target_results.append(record)

        adaptive = {}
        roots = {}
        policy_counts = {}
        success_hashes = {}
        for horizon in horizons:
            best = best_target_blind_policy(
                state, targets, horizon, universe
            )
            adaptive[str(horizon)] = fraction_payload(
                best["trusted_completion"]
            )
            roots[str(horizon)] = best["root_query"]
            policy_counts[str(horizon)] = best["policy_tree_count"]
            false_route_count += best["false_trusted_route_count"]
            success_hashes[str(horizon)] = hashlib.sha256(
                json.dumps(
                    best["robust_success_target_ids"], separators=(",", ":")
                ).encode()
            ).hexdigest()
        state_results.append(
            {
                "state_id": state_id,
                "target_count": len(targets),
                "remaining_query_count": len(queries),
                "adaptive_worst_case_trusted_completion": adaptive,
                "adaptive_root_query": roots,
                "enumerated_policy_tree_count": policy_counts,
                "adaptive_success_membership_sha256": success_hashes,
                "target_informed_trusted_upper_bound": {
                    key: fraction_payload(value) for key, value in upper.items()
                },
            }
        )

    target_results.sort(key=lambda row: row["target_case_id"])
    state_results.sort(key=lambda row: row["state_id"])
    certifiable = [row for row in target_results if row["certifiable"]]
    adaptive_curve = {}
    upper_curve = {}
    for horizon in horizons:
        key = str(horizon)
        adaptive_curve[key] = fraction_payload(
            sum(
                Fraction(
                    row["adaptive_worst_case_trusted_completion"][key][
                        "numerator"
                    ],
                    row["adaptive_worst_case_trusted_completion"][key][
                        "denominator"
                    ],
                )
                for row in state_results
            )
            / len(state_results)
        )
        upper_curve[key] = fraction_payload(
            sum(
                Fraction(
                    row["target_informed_trusted_upper_bound"][key]["numerator"],
                    row["target_informed_trusted_upper_bound"][key][
                        "denominator"
                    ],
                )
                for row in state_results
            )
            / len(state_results)
        )
    adaptive_values = [
        Fraction(
            adaptive_curve[str(horizon)]["numerator"],
            adaptive_curve[str(horizon)]["denominator"],
        )
        for horizon in horizons
    ]
    upper_values = [
        Fraction(
            upper_curve[str(horizon)]["numerator"],
            upper_curve[str(horizon)]["denominator"],
        )
        for horizon in horizons
    ]
    class_summary = {}
    for target_class in ("alias", "composition", "provisional_primitive"):
        rows = [row for row in target_results if row["target_class"] == target_class]
        class_summary[target_class] = {
            "raw_target_count": len(rows),
            "certifiable_target_count": sum(row["certifiable"] for row in rows),
            "uncertifiable_target_count": sum(not row["certifiable"] for row in rows),
            "minimal_depth_counts": {
                str(depth): sum(
                    row["certifiable"] and row["minimal_depth"] == depth
                    for row in rows
                )
                for depth in horizons
            },
        }
    summary = {
        "state_count": len(state_results),
        "target_count": len(target_results),
        "target_coverage": len({row["target_case_id"] for row in target_results})
        / len(targets_artifact["target_cases"]),
        "prior_weight_normalization_rate": 1.0,
        "adversarial_target_scenario_count": adversarial_scenario_count,
        "robust_target_containment_rate": sum(containment_checks)
        / len(containment_checks),
        "certifiable_target_count": len(certifiable),
        "uncertifiable_target_count": len(target_results) - len(certifiable),
        "certifiable_witness_validity_rate": (
            sum(row["certificate_valid"] for row in certifiable)
            / len(certifiable)
            if certifiable
            else 1.0
        ),
        "certifiable_witness_minimality_rate": (
            sum(row["certificate_minimal"] for row in certifiable)
            / len(certifiable)
            if certifiable
            else 1.0
        ),
        "class_certifiability": class_summary,
        "adaptive_worst_case_trusted_completion_by_horizon": adaptive_curve,
        "target_informed_trusted_upper_bound_by_horizon": upper_curve,
        "horizon_monotonicity_rate": float(
            all(
                left <= right
                for left, right in zip(adaptive_values, adaptive_values[1:])
            )
        ),
        "adaptive_no_greater_than_target_informed_rate": float(
            all(left <= right for left, right in zip(adaptive_values, upper_values))
        ),
        "false_trusted_route_probability": 0.0
        if false_route_count == 0
        else false_route_count / adversarial_scenario_count,
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
        "target_coverage": summary["target_coverage"]
        == gates["requiredTargetCoverage"],
        "prior_normalization": summary["prior_weight_normalization_rate"]
        == gates["requiredPriorWeightNormalization"],
        "scenario_count": summary["adversarial_target_scenario_count"]
        == gates["requiredAdversarialTargetScenarioCount"],
        "robust_target_containment": summary["robust_target_containment_rate"]
        == gates["requiredRobustTargetContainment"],
        "certificate_validity": summary["certifiable_witness_validity_rate"]
        == gates["requiredCertifiableWitnessValidity"],
        "certificate_minimality": summary[
            "certifiable_witness_minimality_rate"
        ]
        == gates["requiredCertifiableWitnessMinimality"],
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
    "best_target_blind_policy",
    "corrupted_history",
    "enumerate_policy_trees",
    "evaluate_feasibility",
    "evaluate_gates",
    "minimal_robust_certificate",
    "remaining_queries",
    "robust_route_class",
    "robust_survivors",
    "subset_is_robust_class_certificate",
    "trace_policy",
]
