from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import math
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe


NONORACLE_POLICIES = (
    "no_query_bayes_terminal",
    "forced_map_no_query",
    "random_open_loop_pair",
    "greedy_class_information_gain",
    "optimal_open_loop_pair",
    "exact_bayes_adaptive",
)


def parse_fraction(value: str | int) -> Fraction:
    if isinstance(value, int):
        return Fraction(value)
    numerator, denominator = value.split("/")
    return Fraction(int(numerator), int(denominator))


def fraction_payload(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def initial_belief(
    candidate_ids: list[str], universe_by_id: dict[str, dict[str, Any]], config: dict[str, Any]
) -> tuple[tuple[str, Fraction], ...]:
    classes = config["prior"]["expressibilityClasses"]
    counts = Counter(universe_by_id[item]["expressibility_class"] for item in candidate_ids)
    if set(counts) != set(classes):
        raise ValueError("candidate set does not cover every prior class")
    class_mass = {name: parse_fraction(config["prior"]["classMass"][name]) for name in classes}
    belief = tuple(
        sorted(
            (
                item,
                class_mass[universe_by_id[item]["expressibility_class"]]
                / counts[universe_by_id[item]["expressibility_class"]],
            )
            for item in candidate_ids
        )
    )
    if sum(weight for _, weight in belief) != 1:
        raise ValueError("initial belief is not normalized")
    return belief


def available_queries(
    belief: tuple[tuple[str, Fraction], ...], universe_by_id: dict[str, dict[str, Any]]
) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(8)
        if len({universe_by_id[item]["truth_table"][index] for item, _ in belief}) == 2
    )


def branch_probability(
    belief: tuple[tuple[str, Fraction], ...],
    query: int,
    outcome: int,
    universe_by_id: dict[str, dict[str, Any]],
) -> Fraction:
    return sum(
        weight
        for item, weight in belief
        if int(universe_by_id[item]["truth_table"][query]) == outcome
    )


def condition(
    belief: tuple[tuple[str, Fraction], ...],
    query: int,
    outcome: int,
    universe_by_id: dict[str, dict[str, Any]],
) -> tuple[tuple[str, Fraction], ...]:
    probability = branch_probability(belief, query, outcome, universe_by_id)
    if not probability:
        return ()
    return tuple(
        (item, weight / probability)
        for item, weight in belief
        if int(universe_by_id[item]["truth_table"][query]) == outcome
    )


def terminal_risk(
    belief: tuple[tuple[str, Fraction], ...],
    universe_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[Fraction, str]:
    decisions = config["tieBreaks"]["terminalDecisionOrder"]
    losses = config["terminalLoss"]
    risks = {
        decision: sum(
            weight
            * Fraction(losses[universe_by_id[item]["expressibility_class"]][decision])
            for item, weight in belief
        )
        for decision in decisions
    }
    best = min(decisions, key=lambda decision: (risks[decision], decisions.index(decision)))
    return risks[best], best


def forced_map_risk(
    belief: tuple[tuple[str, Fraction], ...],
    universe_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[Fraction, str]:
    order = [item for item in config["tieBreaks"]["terminalDecisionOrder"] if item != "defer"]
    mass = Counter()
    for item, weight in belief:
        mass[universe_by_id[item]["expressibility_class"]] += weight
    decision = max(order, key=lambda item: (mass[item], -order.index(item)))
    risk = sum(
        weight * Fraction(config["terminalLoss"][universe_by_id[item]["expressibility_class"]][decision])
        for item, weight in belief
    )
    return risk, decision


def _class_entropy(
    belief: tuple[tuple[str, Fraction], ...], universe_by_id: dict[str, dict[str, Any]]
) -> float:
    mass = Counter()
    for item, weight in belief:
        mass[universe_by_id[item]["expressibility_class"]] += float(weight)
    return -sum(value * math.log2(value) for value in mass.values() if value)


def exact_bayes_policy(
    belief: tuple[tuple[str, Fraction], ...],
    queries: tuple[int, ...],
    horizon: int,
    universe_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    query_cost = parse_fraction(config["queryModel"]["queryCost"])

    @lru_cache(maxsize=None)
    def solve(
        state: tuple[tuple[str, Fraction], ...], actions: tuple[int, ...], remaining: int
    ) -> tuple[Fraction, tuple[Any, ...]]:
        stop_risk, decision = terminal_risk(state, universe_by_id, config)
        best_risk = stop_risk
        best = ("STOP", decision)
        if remaining:
            for query in actions:
                risk = query_cost
                children = []
                for outcome in (0, 1):
                    probability = branch_probability(state, query, outcome, universe_by_id)
                    if probability:
                        child_state = condition(state, query, outcome, universe_by_id)
                        child_risk, child_policy = solve(
                            child_state,
                            tuple(item for item in actions if item != query),
                            remaining - 1,
                        )
                        risk += probability * child_risk
                        children.append((outcome, probability, child_policy))
                if risk < best_risk:
                    best_risk = risk
                    best = ("QUERY", query, tuple(children))
        return best_risk, best

    risk, tree = solve(belief, queries, horizon)
    return {"risk": risk, "tree": tree}


def _serialize_tree(tree: tuple[Any, ...]) -> dict[str, Any]:
    if tree[0] == "STOP":
        return {"action": "STOP", "terminal_decision": tree[1]}
    return {
        "action": f"QUERY_V{tree[1]}",
        "valuation_index": tree[1],
        "children": {
            str(outcome): {
                "probability": fraction_payload(probability),
                "policy": _serialize_tree(child),
            }
            for outcome, probability, child in tree[2]
        },
    }


def _tree_statistics(tree: tuple[Any, ...]) -> dict[str, Fraction]:
    if tree[0] == "STOP":
        return {
            "expected_queries": Fraction(0),
            "defer_probability": Fraction(tree[1] == "defer"),
        }
    expected_queries = Fraction(1)
    defer = Fraction(0)
    for _, probability, child in tree[2]:
        stats = _tree_statistics(child)
        expected_queries += probability * stats["expected_queries"]
        defer += probability * stats["defer_probability"]
    return {"expected_queries": expected_queries, "defer_probability": defer}


def _greedy_tree(
    belief: tuple[tuple[str, Fraction], ...],
    queries: tuple[int, ...],
    remaining: int,
    universe_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[Any, ...]:
    if not remaining or not queries:
        return ("STOP", terminal_risk(belief, universe_by_id, config)[1])
    base_entropy = _class_entropy(belief, universe_by_id)
    gains = {}
    for query in queries:
        expected = 0.0
        for outcome in (0, 1):
            probability = branch_probability(belief, query, outcome, universe_by_id)
            if probability:
                expected += float(probability) * _class_entropy(
                    condition(belief, query, outcome, universe_by_id), universe_by_id
                )
        gains[query] = base_entropy - expected
    query = max(queries, key=lambda item: (gains[item], -item))
    children = []
    for outcome in (0, 1):
        probability = branch_probability(belief, query, outcome, universe_by_id)
        if probability:
            child = condition(belief, query, outcome, universe_by_id)
            children.append(
                (
                    outcome,
                    probability,
                    _greedy_tree(
                        child,
                        tuple(item for item in queries if item != query),
                        remaining - 1,
                        universe_by_id,
                        config,
                    ),
                )
            )
    return ("QUERY", query, tuple(children))


def _evaluate_tree(
    belief: tuple[tuple[str, Fraction], ...],
    tree: tuple[Any, ...],
    universe_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> Fraction:
    if tree[0] == "STOP":
        decision = tree[1]
        return sum(
            weight * Fraction(config["terminalLoss"][universe_by_id[item]["expressibility_class"]][decision])
            for item, weight in belief
        )
    risk = parse_fraction(config["queryModel"]["queryCost"])
    query = tree[1]
    for outcome, probability, child_tree in tree[2]:
        risk += probability * _evaluate_tree(
            condition(belief, query, outcome, universe_by_id),
            child_tree,
            universe_by_id,
            config,
        )
    return risk


def _open_loop_pair_risk(
    belief: tuple[tuple[str, Fraction], ...],
    pair: tuple[int, int],
    universe_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> Fraction:
    risk = 2 * parse_fraction(config["queryModel"]["queryCost"])
    for outcomes in product((0, 1), repeat=2):
        state = belief
        probability = Fraction(1)
        for query, outcome in zip(pair, outcomes):
            branch = branch_probability(state, query, outcome, universe_by_id)
            probability *= branch
            if not branch:
                break
            state = condition(state, query, outcome, universe_by_id)
        if probability:
            risk += probability * terminal_risk(state, universe_by_id, config)[0]
    return risk


def evaluate_case(
    record_id: str,
    candidate_ids: list[str],
    target_candidate_id: str,
    logical_target_group: str,
    universe_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    belief = initial_belief(candidate_ids, universe_by_id, config)
    queries = available_queries(belief, universe_by_id)
    horizon = config["queryModel"]["maximumQueries"]
    no_query_risk, no_query_decision = terminal_risk(belief, universe_by_id, config)
    map_risk, map_decision = forced_map_risk(belief, universe_by_id, config)
    pairs = list(combinations(queries, 2))
    pair_risks = {pair: _open_loop_pair_risk(belief, pair, universe_by_id, config) for pair in pairs}
    open_pair = min(pairs, key=lambda pair: (pair_risks[pair], pair))
    open_risk = pair_risks[open_pair]
    random_risk = sum(pair_risks.values(), Fraction(0)) / len(pair_risks)
    greedy_tree = _greedy_tree(belief, queries, horizon, universe_by_id, config)
    greedy_risk = _evaluate_tree(belief, greedy_tree, universe_by_id, config)
    bayes = exact_bayes_policy(belief, queries, horizon, universe_by_id, config)
    bayes_stats = _tree_statistics(bayes["tree"])
    root_children = bayes["tree"][2] if bayes["tree"][0] == "QUERY" else ()
    child_actions = tuple(child[2][0:2] for child in root_children)
    risks = {
        "no_query_bayes_terminal": no_query_risk,
        "forced_map_no_query": map_risk,
        "random_open_loop_pair": random_risk,
        "greedy_class_information_gain": greedy_risk,
        "optimal_open_loop_pair": open_risk,
        "exact_bayes_adaptive": bayes["risk"],
        "oracle_class": Fraction(0),
    }
    return {
        "record_id": record_id,
        "logical_target_group": logical_target_group,
        "candidate_count": len(candidate_ids),
        "candidate_class_counts": dict(sorted(Counter(universe_by_id[item]["expressibility_class"] for item in candidate_ids).items())),
        "target_candidate_id": target_candidate_id,
        "target_retained": target_candidate_id in candidate_ids,
        "available_queries": list(queries),
        "policy_expected_risk": {name: fraction_payload(value) for name, value in risks.items()},
        "policy_regret_vs_exact_bayes": {name: fraction_payload(value - bayes["risk"]) for name, value in risks.items() if name != "oracle_class"},
        "no_query_decision": no_query_decision,
        "forced_map_decision": map_decision,
        "optimal_open_loop_pair": list(open_pair),
        "exact_bayes_policy": _serialize_tree(bayes["tree"]),
        "exact_bayes_root_query": bayes["tree"][1] if bayes["tree"][0] == "QUERY" else None,
        "history_dependent_second_action": len(set(child_actions)) > 1,
        "positive_value_of_information": bayes["risk"] < no_query_risk,
        "strict_improvement_over_optimal_open_loop": bayes["risk"] < open_risk,
        "bayes_no_worse_than_every_nonoracle_baseline": all(bayes["risk"] <= risks[name] for name in NONORACLE_POLICIES),
        "exact_bayes_expected_query_count": fraction_payload(bayes_stats["expected_queries"]),
        "exact_bayes_defer_probability": fraction_payload(bayes_stats["defer_probability"]),
    }


def build_planner_evaluation(
    frozen_predictions: dict[str, Any],
    hidden_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline = config["caseSelection"]["sourceBaseline"]
    selected = {
        row["record_id"]: row["predictions"][baseline]
        for row in frozen_predictions["predictions"]
        if row["predictions"][baseline]["evidence_status"] == config["caseSelection"]["requiredEvidenceStatus"]
        and row["predictions"][baseline]["candidate_count"] == config["caseSelection"]["requiredCandidateCount"]
    }
    hidden_by_id = {row["record_id"]: row for row in hidden_records}
    universe_by_id = {row["candidate_id"]: row for row in candidate_universe()}
    cases = [
        evaluate_case(
            record_id,
            selected[record_id]["candidate_ids"],
            hidden_by_id[record_id]["target_candidate_id"],
            hidden_by_id[record_id]["logical_target_group"],
            universe_by_id,
            config,
        )
        for record_id in sorted(selected)
    ]
    mean_risks = {
        policy: sum(Fraction(case["policy_expected_risk"][policy]["numerator"], case["policy_expected_risk"][policy]["denominator"]) for case in cases) / len(cases)
        for policy in (*NONORACLE_POLICIES, "oracle_class")
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[case["logical_target_group"]].append(case)
    renaming_invariant = [
        len({tuple((name, row["policy_expected_risk"][name]["numerator"], row["policy_expected_risk"][name]["denominator"]) for name in NONORACLE_POLICIES) for row in rows}) == 1
        for rows in groups.values()
    ]
    summary = {
        "case_count": len(cases),
        "candidate_count_values": sorted({case["candidate_count"] for case in cases}),
        "class_coverage_values": sorted({len(case["candidate_class_counts"]) for case in cases}),
        "target_candidate_retention": sum(case["target_retained"] for case in cases) / len(cases),
        "mean_expected_risk": {name: fraction_payload(value) for name, value in mean_risks.items()},
        "positive_value_of_information_case_count": sum(case["positive_value_of_information"] for case in cases),
        "strict_improvement_over_optimal_open_loop_case_count": sum(case["strict_improvement_over_optimal_open_loop"] for case in cases),
        "unique_exact_bayes_root_queries": sorted({case["exact_bayes_root_query"] for case in cases}),
        "history_dependent_second_action_case_count": sum(case["history_dependent_second_action"] for case in cases),
        "bayes_no_worse_than_every_nonoracle_baseline_case_rate": sum(case["bayes_no_worse_than_every_nonoracle_baseline"] for case in cases) / len(cases),
        "renaming_risk_invariance": sum(renaming_invariant) / len(renaming_invariant),
    }
    return {"cases": cases, "summary": summary}


def evaluate_gates(evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["plannerGates"]
    summary = evaluation["summary"]
    return {
        "case_count": summary["case_count"] == gates["requiredCaseCount"],
        "candidate_count": summary["candidate_count_values"] == [gates["requiredCandidatesPerCase"]],
        "class_coverage": summary["class_coverage_values"] == [gates["requiredExpressibilityClassCoverage"]],
        "target_retention": summary["target_candidate_retention"] == gates["requiredTargetCandidateRetention"],
        "positive_value_of_information": summary["positive_value_of_information_case_count"] == gates["requiredPositiveValueOfInformationCaseCount"],
        "root_action_variation": len(summary["unique_exact_bayes_root_queries"]) >= gates["minimumUniqueBayesRootQueryCount"],
        "history_dependent_second_action": summary["history_dependent_second_action_case_count"] >= gates["minimumHistoryDependentSecondActionCaseCount"],
        "strict_improvement_over_open_loop": summary["strict_improvement_over_optimal_open_loop_case_count"] >= gates["minimumStrictBayesImprovementOverOpenLoopCaseCount"],
        "bayes_nonoracle_dominance": summary["bayes_no_worse_than_every_nonoracle_baseline_case_rate"] == gates["requiredBayesNoWorseThanEveryNonOracleBaselineCaseRate"],
        "renaming_risk_invariance": summary["renaming_risk_invariance"] == gates["requiredRenamingRiskInvariance"],
        "zero_disallowed_access": all(access[key] <= gates[maximum] for key, maximum in {
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
        }.items()),
    }


__all__ = [
    "NONORACLE_POLICIES", "available_queries", "build_planner_evaluation", "condition",
    "evaluate_case", "evaluate_gates", "exact_bayes_policy", "fraction_payload",
    "initial_belief", "parse_fraction", "terminal_risk",
]
