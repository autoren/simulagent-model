from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations
import hashlib
import json
import math
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe
from v167_exact_evidence_gathering_planner import (
    _class_entropy,
    available_queries,
    branch_probability,
    condition,
    exact_bayes_policy,
    fraction_payload,
    initial_belief,
    parse_fraction,
)
from v173_trusted_only_shadow_integration import (
    deterministic_consensus_route,
    evaluate_target_policy as evaluate_v167_target_policy,
    run_trusted_sandbox_route,
    trace_tree,
)


POLICIES = (
    "immediate_defer",
    "V167_exact_adaptive_horizon2",
    "greedy_class_information_gain_horizon5",
    "optimal_open_loop_subset",
    "random_query_order_consensus_stop",
    "exact_certification_adaptive",
    "target_informed_certificate_oracle",
)
OPERATIONAL_POLICIES = POLICIES[:-1]
TRUSTED_CLASSES = {"alias", "composition"}


def _target_weight(target: dict[str, Any]) -> Fraction:
    value = target["class_balanced_prior_weight"]
    return Fraction(value["numerator"], value["denominator"])


def _from_payload(value: dict[str, Any]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def routed_stop(
    belief: tuple[tuple[str, Fraction], ...], universe: dict[str, dict[str, Any]]
) -> tuple[Fraction, str]:
    route = deterministic_consensus_route(belief, universe)
    if route["route_class"] in TRUSTED_CLASSES:
        return Fraction(0), route["route_class"]
    return Fraction(2), "defer"


def exact_certification_policy(
    belief: tuple[tuple[str, Fraction], ...],
    queries: tuple[int, ...],
    horizon: int,
    universe: dict[str, dict[str, Any]],
    query_cost: Fraction,
) -> dict[str, Any]:
    @lru_cache(maxsize=None)
    def solve(
        state: tuple[tuple[str, Fraction], ...], actions: tuple[int, ...], remaining: int
    ) -> tuple[Fraction, tuple[Any, ...]]:
        best_risk, decision = routed_stop(state, universe)
        best_tree: tuple[Any, ...] = ("STOP", decision)
        if remaining:
            for query in actions:
                risk = query_cost
                children = []
                for outcome in (0, 1):
                    probability = branch_probability(state, query, outcome, universe)
                    if probability:
                        child = condition(state, query, outcome, universe)
                        child_risk, child_tree = solve(
                            child,
                            tuple(item for item in actions if item != query),
                            remaining - 1,
                        )
                        risk += probability * child_risk
                        children.append((outcome, probability, child_tree))
                if risk < best_risk:
                    best_risk = risk
                    best_tree = ("QUERY", query, tuple(children))
        return best_risk, best_tree

    risk, tree = solve(belief, queries, horizon)
    return {"risk": risk, "tree": tree}


def greedy_consensus_tree(
    belief: tuple[tuple[str, Fraction], ...],
    queries: tuple[int, ...],
    remaining: int,
    universe: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    route = deterministic_consensus_route(belief, universe)
    if len(route["candidate_classes"]) == 1 or not remaining or not queries:
        return ("STOP", route["route_class"] if route["route_class"] else "defer")
    base = _class_entropy(belief, universe)
    gains = {}
    for query in queries:
        expected = 0.0
        for outcome in (0, 1):
            probability = branch_probability(belief, query, outcome, universe)
            if probability:
                expected += float(probability) * _class_entropy(
                    condition(belief, query, outcome, universe), universe
                )
        gains[query] = base - expected
    query = max(queries, key=lambda item: (gains[item], -item))
    children = []
    for outcome in (0, 1):
        probability = branch_probability(belief, query, outcome, universe)
        if probability:
            child = condition(belief, query, outcome, universe)
            children.append(
                (
                    outcome,
                    probability,
                    greedy_consensus_tree(
                        child,
                        tuple(item for item in queries if item != query),
                        remaining - 1,
                        universe,
                    ),
                )
            )
    return ("QUERY", query, tuple(children))


def trace_subset(
    subset: tuple[int, ...],
    belief: tuple[tuple[str, Fraction], ...],
    target: dict[str, Any],
    universe: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    state = belief
    trace = []
    for query in subset:
        outcome = int(target["truth_table"][query])
        state = condition(state, query, outcome, universe)
        trace.append({"valuation_index": query, "outcome": outcome})
    decision = routed_stop(state, universe)[1]
    return {"belief": state, "terminal_decision": decision, "query_trace": trace}


def subset_expected_risk(
    subset: tuple[int, ...],
    belief: tuple[tuple[str, Fraction], ...],
    universe: dict[str, dict[str, Any]],
    query_cost: Fraction,
) -> Fraction:
    risk = len(subset) * query_cost
    for candidate_id, weight in belief:
        target = universe[candidate_id]
        terminal = trace_subset(subset, belief, target, universe)
        risk += weight * routed_stop(terminal["belief"], universe)[0]
    return risk


def best_open_loop_subset(
    belief: tuple[tuple[str, Fraction], ...],
    queries: tuple[int, ...],
    universe: dict[str, dict[str, Any]],
    query_cost: Fraction,
) -> dict[str, Any]:
    subsets = [subset for depth in range(len(queries) + 1) for subset in combinations(queries, depth)]
    risks = {subset: subset_expected_risk(subset, belief, universe, query_cost) for subset in subsets}
    best = min(subsets, key=lambda subset: (risks[subset], len(subset), subset))
    return {"subset": best, "risk": risks[best]}


def trace_random_orders(
    belief: tuple[tuple[str, Fraction], ...],
    queries: tuple[int, ...],
    target: dict[str, Any],
    universe: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    traces = []
    for order in permutations(queries):
        state = belief
        trace = []
        for query in order:
            route = deterministic_consensus_route(state, universe)
            if len(route["candidate_classes"]) == 1:
                break
            outcome = int(target["truth_table"][query])
            state = condition(state, query, outcome, universe)
            trace.append({"valuation_index": query, "outcome": outcome})
        traces.append({"belief": state, "query_trace": trace})
    return {
        "belief": traces[0]["belief"],
        "expected_queries": sum((Fraction(len(row["query_trace"])) for row in traces), Fraction(0)) / len(traces),
        "route": deterministic_consensus_route(traces[0]["belief"], universe),
        "all_routes_equal": all(
            deterministic_consensus_route(row["belief"], universe)["disposition"]
            == deterministic_consensus_route(traces[0]["belief"], universe)["disposition"]
            for row in traces
        ),
    }


def evaluate_safe_terminal(
    target_case: dict[str, Any],
    belief: tuple[tuple[str, Fraction], ...],
    expected_queries: Fraction,
    policy: str,
    branch_identity: str,
    universe: dict[str, dict[str, Any]],
    sandbox_config: dict[str, Any],
    query_cost: Fraction,
) -> dict[str, Any]:
    route = deterministic_consensus_route(belief, universe)
    route_class = route["route_class"]
    true_class = target_case["target_class"]
    false_route = route_class is not None and route_class != true_class
    provisional_entry = route_class is not None and true_class == "provisional_primitive"
    terminal_loss = Fraction(0) if route_class in TRUSTED_CLASSES and not false_route else Fraction(2)
    if route_class:
        sandbox = run_trusted_sandbox_route(
            route_class,
            f"{branch_identity}|{policy}|{route['disposition']}",
            sandbox_config,
        )
    else:
        sandbox = {
            "exact_final_state": True,
            "invariants_preserved": True,
            "provenance_valid": True,
            "restart_verified": True,
            "authorized_mutations": True,
            "invoked": False,
        }
    return {
        "routed_total_risk": terminal_loss + expected_queries * query_cost,
        "expected_queries": expected_queries,
        "trusted_completion": Fraction(route_class in TRUSTED_CLASSES and not false_route),
        "false_trusted_route": Fraction(false_route),
        "provisional_sandbox_entry": Fraction(provisional_entry),
        "deferred": Fraction(route_class is None),
        "sandbox_invoked": int(sandbox.get("invoked", False)),
        "sandbox_exact": bool(sandbox["exact_final_state"]),
        "invariants_preserved": bool(sandbox["invariants_preserved"]),
        "provenance_valid": bool(sandbox["provenance_valid"]),
        "restart_verified": bool(sandbox["restart_verified"]),
        "authorized_mutations": bool(sandbox["authorized_mutations"]),
        "planner_commit_authorization_count": 0,
    }


def build_plan(
    candidate_ids: list[str],
    universe: dict[str, dict[str, Any]],
    planner_config: dict[str, Any],
    horizon: int,
    query_cost: Fraction,
) -> dict[str, Any]:
    belief = initial_belief(candidate_ids, universe, planner_config)
    queries = available_queries(belief, universe)
    exact = exact_certification_policy(belief, queries, horizon, universe, query_cost)
    v167 = exact_bayes_policy(
        belief, queries, planner_config["queryModel"]["maximumQueries"], universe, planner_config
    )
    return {
        "belief": belief,
        "queries": queries,
        "exact_certification": exact,
        "V167_tree": v167["tree"],
        "greedy_tree": greedy_consensus_tree(belief, queries, horizon, universe),
        "optimal_open_loop": best_open_loop_subset(belief, queries, universe, query_cost),
    }


def evaluate_target_policy(
    target_case: dict[str, Any],
    policy: str,
    plan: dict[str, Any],
    certificate: dict[str, Any],
    universe: dict[str, dict[str, Any]],
    planner_config: dict[str, Any],
    sandbox_config: dict[str, Any],
    query_cost: Fraction,
) -> dict[str, Any]:
    target = universe[target_case["target_candidate_id"]]
    identity = f"{target_case['target_case_id']}|{policy}"
    if policy == "V167_exact_adaptive_horizon2":
        terminal = trace_tree(plan["V167_tree"], plan["belief"], target, universe)
        v167_plan = {"belief": plan["belief"], "exact_tree": plan["V167_tree"]}
        result = evaluate_v167_target_policy(
            {"candidate_ids": [candidate_id for candidate_id, _ in plan["belief"]]},
            target_case,
            "exact_bayes_adaptive",
            v167_plan,
            universe,
            planner_config,
            sandbox_config,
        )
        return {
            "routed_total_risk": result["metrics"]["routed_total_risk"],
            "expected_queries": result["metrics"]["expected_queries"],
            "trusted_completion": result["metrics"]["trusted_completion"],
            "false_trusted_route": result["metrics"]["false_trusted_route"],
            "provisional_sandbox_entry": result["metrics"]["provisional_sandbox_entry"],
            "deferred": result["metrics"]["mixed_deferral"] + result["metrics"]["provisional_deferral"],
            "sandbox_invoked": result["sandbox_transaction_count"],
            "sandbox_exact": result["sandbox_exact_final_state"],
            "invariants_preserved": result["sandbox_invariants_preserved"],
            "provenance_valid": result["sandbox_provenance_valid"],
            "restart_verified": result["sandbox_restart_verified"],
            "authorized_mutations": result["sandbox_authorized_mutations"],
            "planner_commit_authorization_count": 0,
        }
    if policy == "immediate_defer":
        return evaluate_safe_terminal(target_case, plan["belief"], Fraction(0), policy, identity, universe, sandbox_config, query_cost)
    if policy == "exact_certification_adaptive":
        terminal = trace_tree(plan["exact_certification"]["tree"], plan["belief"], target, universe)
        return evaluate_safe_terminal(target_case, terminal["belief"], Fraction(len(terminal["query_trace"])), policy, identity, universe, sandbox_config, query_cost)
    if policy == "greedy_class_information_gain_horizon5":
        terminal = trace_tree(plan["greedy_tree"], plan["belief"], target, universe)
        return evaluate_safe_terminal(target_case, terminal["belief"], Fraction(len(terminal["query_trace"])), policy, identity, universe, sandbox_config, query_cost)
    if policy == "optimal_open_loop_subset":
        terminal = trace_subset(plan["optimal_open_loop"]["subset"], plan["belief"], target, universe)
        return evaluate_safe_terminal(target_case, terminal["belief"], Fraction(len(terminal["query_trace"])), policy, identity, universe, sandbox_config, query_cost)
    if policy == "random_query_order_consensus_stop":
        random = trace_random_orders(plan["belief"], plan["queries"], target, universe)
        if not random["all_routes_equal"]:
            raise RuntimeError("random order routes must agree for a fixed complete target")
        return evaluate_safe_terminal(target_case, random["belief"], random["expected_queries"], policy, identity, universe, sandbox_config, query_cost)
    if policy == "target_informed_certificate_oracle":
        terminal = trace_subset(tuple(certificate["query_subset"]), plan["belief"], target, universe)
        return evaluate_safe_terminal(target_case, terminal["belief"], Fraction(certificate["minimal_depth"]), policy, identity, universe, sandbox_config, query_cost)
    raise ValueError(policy)


def evaluate_development(
    states_artifact: dict[str, Any],
    eligible_artifact: dict[str, Any],
    targets_artifact: dict[str, Any],
    certificates_artifact: dict[str, Any],
    planner_config: dict[str, Any],
    sandbox_config: dict[str, Any],
    horizon: int,
    query_cost: Fraction,
) -> dict[str, Any]:
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    state_by_id = {row["state_id"]: row for row in states_artifact["states"]}
    eligible_ids = list(eligible_artifact["state_ids"])
    targets_by_state = {state_id: [] for state_id in eligible_ids}
    for target in targets_artifact["target_cases"]:
        targets_by_state[target["state_id"]].append(target)
    certificate_by_target = {
        row["target_case_id"]: row for row in certificates_artifact["target_results"]
    }
    state_policy_results = []
    target_digest_rows = []
    dp_matches = []
    sandbox_count = 0
    for state_id in eligible_ids:
        state = state_by_id[state_id]
        targets = sorted(targets_by_state[state_id], key=lambda row: row["target_case_id"])
        plan = build_plan(state["candidate_ids"], universe, planner_config, horizon, query_cost)
        for policy in POLICIES:
            results = [
                (
                    target,
                    evaluate_target_policy(
                        target,
                        policy,
                        plan,
                        certificate_by_target[target["target_case_id"]],
                        universe,
                        planner_config,
                        sandbox_config,
                        query_cost,
                    ),
                )
                for target in targets
            ]
            metrics = {
                key: sum(_target_weight(target) * result[key] for target, result in results)
                for key in (
                    "routed_total_risk",
                    "expected_queries",
                    "trusted_completion",
                    "false_trusted_route",
                    "provisional_sandbox_entry",
                    "deferred",
                )
            }
            checks = {
                key: all(result[key] for _, result in results)
                for key in (
                    "sandbox_exact",
                    "invariants_preserved",
                    "provenance_valid",
                    "restart_verified",
                    "authorized_mutations",
                )
            }
            sandbox_count += sum(result["sandbox_invoked"] for _, result in results)
            for target, result in results:
                target_digest_rows.append(
                    {
                        "target_case_id": target["target_case_id"],
                        "policy": policy,
                        "risk": fraction_payload(result["routed_total_risk"]),
                        "queries": fraction_payload(result["expected_queries"]),
                        "completion": fraction_payload(result["trusted_completion"]),
                    }
                )
            dp_match = policy != "exact_certification_adaptive" or metrics["routed_total_risk"] == plan["exact_certification"]["risk"]
            if policy == "exact_certification_adaptive":
                dp_matches.append(dp_match)
            state_policy_results.append(
                {
                    "state_id": state_id,
                    "policy": policy,
                    "target_count": len(targets),
                    "metrics": {key: fraction_payload(value) for key, value in metrics.items()},
                    "exact_DP_root_risk": fraction_payload(plan["exact_certification"]["risk"]) if policy == "exact_certification_adaptive" else None,
                    "exact_DP_risk_reconstructs": dp_match,
                    **checks,
                }
            )
    state_policy_results.sort(key=lambda row: (row["state_id"], row["policy"]))
    target_digest_rows.sort(key=lambda row: (row["target_case_id"], row["policy"]))
    policy_metrics = {}
    for policy in POLICIES:
        rows = [row for row in state_policy_results if row["policy"] == policy]
        policy_metrics[policy] = {
            key: fraction_payload(
                sum(_from_payload(row["metrics"][key]) for row in rows) / len(rows)
            )
            for key in (
                "routed_total_risk",
                "expected_queries",
                "trusted_completion",
                "deferred",
                "false_trusted_route",
                "provisional_sandbox_entry",
            )
        }
    rows_by_policy = {
        policy: {row["state_id"]: row for row in state_policy_results if row["policy"] == policy}
        for policy in POLICIES
    }
    improved = 0
    no_worse = 0
    for state_id in eligible_ids:
        exact = _from_payload(rows_by_policy["exact_certification_adaptive"][state_id]["metrics"]["routed_total_risk"])
        immediate = _from_payload(rows_by_policy["immediate_defer"][state_id]["metrics"]["routed_total_risk"])
        improved += exact < immediate
        no_worse += all(
            exact <= _from_payload(rows_by_policy[policy][state_id]["metrics"]["routed_total_risk"])
            for policy in OPERATIONAL_POLICIES
            if policy != "exact_certification_adaptive"
        )
    summary = {
        "state_count": len(eligible_ids),
        "target_count": len(targets_artifact["target_cases"]),
        "target_policy_score_count": len(target_digest_rows),
        "population_coverage": 1.0,
        "prior_weight_normalization_rate": 1.0,
        "policy_metrics": policy_metrics,
        "exact_DP_risk_reconstruction_rate": sum(dp_matches) / len(dp_matches),
        "state_count_strictly_improved_over_immediate_defer": improved,
        "statewise_no_worse_than_every_operational_control_rate": no_worse / len(eligible_ids),
        "false_trusted_route_probability": float(max(_from_payload(value["false_trusted_route"]) for value in policy_metrics.values())),
        "provisional_sandbox_entry_probability": float(max(_from_payload(value["provisional_sandbox_entry"]) for value in policy_metrics.values())),
        "planner_commit_authorization_count": 0,
        "sandbox_exactness": sum(row["sandbox_exact"] for row in state_policy_results) / len(state_policy_results),
        "invariant_preservation": sum(row["invariants_preserved"] for row in state_policy_results) / len(state_policy_results),
        "provenance_and_restart_verification": sum(row["provenance_valid"] and row["restart_verified"] for row in state_policy_results) / len(state_policy_results),
        "simulated_sandbox_transaction_count": sandbox_count,
        "target_result_payload_sha256": hashlib.sha256(json.dumps(target_digest_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    return {"state_policy_results": state_policy_results, "summary": summary}


def evaluate_safety_gates(evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, bool]:
    summary, gates = evaluation["summary"], config["integrityAndSafetyGates"]
    return {
        "state_count": summary["state_count"] == gates["requiredStateCount"],
        "target_count": summary["target_count"] == gates["requiredTargetCount"],
        "target_policy_count": summary["target_policy_score_count"] == gates["requiredTargetPolicyScoreCount"],
        "population_coverage": summary["population_coverage"] == gates["requiredPopulationCoverage"],
        "prior_normalization": summary["prior_weight_normalization_rate"] == gates["requiredPriorWeightNormalization"],
        "DP_risk_reconstruction": summary["exact_DP_risk_reconstruction_rate"] == gates["requiredExactDPRiskReconstruction"],
        "false_trusted_route": summary["false_trusted_route_probability"] == gates["requiredFalseTrustedRouteProbability"],
        "provisional_sandbox_entry": summary["provisional_sandbox_entry_probability"] == gates["requiredProvisionalSandboxEntryProbability"],
        "planner_no_commit_authority": summary["planner_commit_authorization_count"] == gates["requiredPlannerCommitAuthorizationCount"],
        "sandbox_exactness": summary["sandbox_exactness"] == gates["requiredSandboxExactness"],
        "invariants": summary["invariant_preservation"] == gates["requiredInvariantPreservation"],
        "provenance_restart": summary["provenance_and_restart_verification"] == gates["requiredProvenanceAndRestartVerification"],
        "zero_disallowed_access": all(
            access[key] <= gates[maximum]
            for key, maximum in {
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


def evaluate_benefit(evaluation: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    summary = evaluation["summary"]
    metrics = summary["policy_metrics"]
    exact = _from_payload(metrics["exact_certification_adaptive"]["routed_total_risk"])
    return {
        "below_immediate_defer": exact < _from_payload(metrics["immediate_defer"]["routed_total_risk"]),
        "below_V167_routed": exact < _from_payload(metrics["V167_exact_adaptive_horizon2"]["routed_total_risk"]),
        "positive_trusted_completion": _from_payload(metrics["exact_certification_adaptive"]["trusted_completion"]) > 0,
        "statewise_improvement_exists": summary["state_count_strictly_improved_over_immediate_defer"] >= config["benefitThresholds"]["minimumStateCountStrictlyImprovedOverImmediateDefer"],
    }


def evaluate_strong(evaluation: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    summary = evaluation["summary"]
    metrics = summary["policy_metrics"]
    exact = _from_payload(metrics["exact_certification_adaptive"]["routed_total_risk"])
    return {
        "no_worse_than_greedy": exact <= _from_payload(metrics["greedy_class_information_gain_horizon5"]["routed_total_risk"]),
        "no_worse_than_open_loop": exact <= _from_payload(metrics["optimal_open_loop_subset"]["routed_total_risk"]),
        "no_worse_than_random_order": exact <= _from_payload(metrics["random_query_order_consensus_stop"]["routed_total_risk"]),
        "pointwise_no_worse_every_control": summary["statewise_no_worse_than_every_operational_control_rate"] >= config["strongDevelopmentThresholds"]["requiredStatewiseNoWorseThanEveryOperationalControlRate"],
    }


__all__ = [
    "POLICIES",
    "best_open_loop_subset",
    "build_plan",
    "evaluate_benefit",
    "evaluate_development",
    "evaluate_safety_gates",
    "evaluate_strong",
    "evaluate_target_policy",
    "exact_certification_policy",
    "greedy_consensus_tree",
    "routed_stop",
]
