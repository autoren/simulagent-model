from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations
import hashlib
import json
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe
from v167_exact_evidence_gathering_planner import (
    NONORACLE_POLICIES,
    _greedy_tree,
    _open_loop_pair_risk,
    available_queries,
    condition,
    evaluate_case,
    exact_bayes_policy,
    forced_map_risk,
    fraction_payload,
    initial_belief,
    parse_fraction,
    terminal_risk,
)
from v168_fixed_ontology_reversible_sandbox import apply_operations, canonical_hash, initial_state
from v171_stateful_sandbox_sequence_confirmation import (
    DurableSandboxHarness,
    atomic_operations,
    owner_operation,
    proposal,
)


POLICIES = (*NONORACLE_POLICIES, "oracle_class")
NUMERIC_METRICS = (
    "recommendation_total_risk",
    "routed_total_risk",
    "expected_queries",
    "trusted_completion",
    "mixed_deferral",
    "provisional_deferral",
    "false_trusted_route",
    "provisional_sandbox_entry",
)
SANDBOX_CHECKS = (
    "sandbox_exact_final_state",
    "sandbox_invariants_preserved",
    "sandbox_provenance_valid",
    "sandbox_restart_verified",
    "sandbox_authorized_mutations",
)


def _fraction_from_payload(value: dict[str, Any]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _target_weight(target: dict[str, Any]) -> Fraction:
    value = target["class_balanced_prior_weight"]
    return Fraction(value["numerator"], value["denominator"])


def _serialize_fractions(values: dict[str, Fraction]) -> dict[str, dict[str, Any]]:
    return {key: fraction_payload(value) for key, value in values.items()}


def build_state_plan(
    candidate_ids: list[str],
    universe: dict[str, dict[str, Any]],
    planner_config: dict[str, Any],
) -> dict[str, Any]:
    belief = initial_belief(candidate_ids, universe, planner_config)
    queries = available_queries(belief, universe)
    horizon = planner_config["queryModel"]["maximumQueries"]
    pairs = tuple(combinations(queries, 2))
    pair_risks = {
        pair: _open_loop_pair_risk(belief, pair, universe, planner_config)
        for pair in pairs
    }
    optimal_pair = min(pairs, key=lambda pair: (pair_risks[pair], pair))
    return {
        "belief": belief,
        "queries": queries,
        "pairs": pairs,
        "no_query_decision": terminal_risk(belief, universe, planner_config)[1],
        "forced_map_decision": forced_map_risk(belief, universe, planner_config)[1],
        "greedy_tree": _greedy_tree(belief, queries, horizon, universe, planner_config),
        "optimal_open_loop_pair": optimal_pair,
        "exact_tree": exact_bayes_policy(belief, queries, horizon, universe, planner_config)["tree"],
    }


def trace_tree(
    tree: tuple[Any, ...],
    belief: tuple[tuple[str, Fraction], ...],
    target: dict[str, Any],
    universe: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    node = tree
    state = belief
    query_trace: list[dict[str, int]] = []
    while node[0] == "QUERY":
        query = int(node[1])
        outcome = int(target["truth_table"][query])
        state = condition(state, query, outcome, universe)
        query_trace.append({"valuation_index": query, "outcome": outcome})
        child = next((row[2] for row in node[2] if int(row[0]) == outcome), None)
        if child is None:
            raise RuntimeError("policy tree omitted a target-reachable outcome")
        node = child
    return {
        "belief": state,
        "terminal_decision": node[1],
        "query_trace": query_trace,
    }


def trace_open_loop(
    pair: tuple[int, int],
    belief: tuple[tuple[str, Fraction], ...],
    target: dict[str, Any],
    universe: dict[str, dict[str, Any]],
    planner_config: dict[str, Any],
) -> dict[str, Any]:
    state = belief
    query_trace = []
    for query in pair:
        outcome = int(target["truth_table"][query])
        state = condition(state, query, outcome, universe)
        query_trace.append({"valuation_index": query, "outcome": outcome})
    return {
        "belief": state,
        "terminal_decision": terminal_risk(state, universe, planner_config)[1],
        "query_trace": query_trace,
    }


def deterministic_consensus_route(
    belief: tuple[tuple[str, Fraction], ...],
    universe: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    classes = sorted({universe[candidate_id]["expressibility_class"] for candidate_id, _ in belief})
    if classes == ["alias"]:
        route = "alias"
        disposition = "trusted_alias"
    elif classes == ["composition"]:
        route = "composition"
        disposition = "trusted_composition"
    elif classes == ["provisional_primitive"]:
        route = None
        disposition = "defer_provisional"
    else:
        route = None
        disposition = "defer_mixed"
    return {
        "candidate_classes": classes,
        "route_class": route,
        "disposition": disposition,
    }


def run_trusted_sandbox_route(
    route_class: str,
    branch_identity: str,
    sandbox_config: dict[str, Any],
) -> dict[str, Any]:
    variant = int(hashlib.sha256(branch_identity.encode()).hexdigest()[:12], 16) % 100000 + 1000
    before = initial_state(variant)
    harness = DurableSandboxHarness(before, sandbox_config)
    if route_class == "alias":
        operations = [owner_operation(before, "D1", 1)]
    elif route_class == "composition":
        operations = atomic_operations(before)
    else:
        raise ValueError("only trusted routes may enter the sandbox")
    item = proposal(f"v173-{hashlib.sha256(branch_identity.encode()).hexdigest()[:20]}", before, operations)
    result = harness.execute(item)
    expected = apply_operations(before, operations)
    harness, restart = DurableSandboxHarness.restart(harness.durable_image(), sandbox_config)
    return {
        "invoked": True,
        "committed": bool(result["committed"]),
        "exact_final_state": harness.store.state == expected,
        "invariants_preserved": harness.visible_invariants_hold(),
        "provenance_valid": harness.provenance_valid(),
        "restart_verified": restart["action"] == "none" and harness.lifecycle[item["transaction_id"]]["status"] == "retained",
        "authorized_mutations": harness.authorized_retained_mutations_hold(),
        "final_state_hash": canonical_hash(harness.store.state),
    }


def evaluate_terminal_path(
    target_case: dict[str, Any],
    target: dict[str, Any],
    terminal: dict[str, Any],
    policy: str,
    branch_identity: str,
    universe: dict[str, dict[str, Any]],
    planner_config: dict[str, Any],
    sandbox_config: dict[str, Any],
) -> dict[str, Any]:
    true_class = target_case["target_class"]
    consensus = deterministic_consensus_route(terminal["belief"], universe)
    route_class = consensus["route_class"]
    routed_decision = route_class if route_class is not None else "defer"
    query_count = len(terminal["query_trace"])
    query_cost = query_count * parse_fraction(planner_config["queryModel"]["queryCost"])
    recommendation_loss = Fraction(planner_config["terminalLoss"][true_class][terminal["terminal_decision"]])
    routed_loss = Fraction(planner_config["terminalLoss"][true_class][routed_decision])
    false_route = route_class is not None and route_class != true_class
    provisional_entry = route_class is not None and true_class == "provisional_primitive"
    if route_class is None:
        sandbox = {
            "invoked": False,
            "committed": False,
            "exact_final_state": True,
            "invariants_preserved": True,
            "provenance_valid": True,
            "restart_verified": True,
            "authorized_mutations": True,
        }
    else:
        sandbox = run_trusted_sandbox_route(
            route_class,
            f"{branch_identity}|{policy}|{consensus['disposition']}",
            sandbox_config,
        )
    return {
        "metrics": {
            "recommendation_total_risk": recommendation_loss + query_cost,
            "routed_total_risk": routed_loss + query_cost,
            "expected_queries": Fraction(query_count),
            "trusted_completion": Fraction(route_class in {"alias", "composition"} and not false_route),
            "mixed_deferral": Fraction(consensus["disposition"] == "defer_mixed"),
            "provisional_deferral": Fraction(consensus["disposition"] == "defer_provisional"),
            "false_trusted_route": Fraction(false_route),
            "provisional_sandbox_entry": Fraction(provisional_entry),
        },
        "terminal_decision_distribution": {terminal["terminal_decision"]: Fraction(1)},
        "route_distribution": {consensus["disposition"]: Fraction(1)},
        "sandbox_transaction_count": int(sandbox["invoked"]),
        "sandbox_exact_final_state": bool(sandbox["exact_final_state"]),
        "sandbox_invariants_preserved": bool(sandbox["invariants_preserved"]),
        "sandbox_provenance_valid": bool(sandbox["provenance_valid"]),
        "sandbox_restart_verified": bool(sandbox["restart_verified"]),
        "sandbox_authorized_mutations": bool(sandbox["authorized_mutations"]),
        "gate_reconstructs": consensus == deterministic_consensus_route(terminal["belief"], universe),
        "planner_commit_authorization_count": 0,
    }


def _average_path_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = len(results)
    terminal_distribution: Counter[str] = Counter()
    route_distribution: Counter[str] = Counter()
    for result in results:
        terminal_distribution.update(result["terminal_decision_distribution"])
        route_distribution.update(result["route_distribution"])
    return {
        "metrics": {
            key: sum((row["metrics"][key] for row in results), Fraction(0)) / denominator
            for key in NUMERIC_METRICS
        },
        "terminal_decision_distribution": {
            key: value / denominator for key, value in sorted(terminal_distribution.items())
        },
        "route_distribution": {
            key: value / denominator for key, value in sorted(route_distribution.items())
        },
        "sandbox_transaction_count": sum(row["sandbox_transaction_count"] for row in results),
        **{key: all(row[key] for row in results) for key in SANDBOX_CHECKS},
        "gate_reconstructs": all(row["gate_reconstructs"] for row in results),
        "planner_commit_authorization_count": 0,
    }


def evaluate_target_policy(
    state: dict[str, Any],
    target_case: dict[str, Any],
    policy: str,
    plan: dict[str, Any],
    universe: dict[str, dict[str, Any]],
    planner_config: dict[str, Any],
    sandbox_config: dict[str, Any],
) -> dict[str, Any]:
    target = universe[target_case["target_candidate_id"]]
    belief = plan["belief"]
    branch_identity = f"{target_case['target_case_id']}|{policy}"
    if policy == "no_query_bayes_terminal":
        terminal = {"belief": belief, "terminal_decision": plan["no_query_decision"], "query_trace": []}
        return evaluate_terminal_path(target_case, target, terminal, policy, branch_identity, universe, planner_config, sandbox_config)
    if policy == "forced_map_no_query":
        terminal = {"belief": belief, "terminal_decision": plan["forced_map_decision"], "query_trace": []}
        return evaluate_terminal_path(target_case, target, terminal, policy, branch_identity, universe, planner_config, sandbox_config)
    if policy == "greedy_class_information_gain":
        terminal = trace_tree(plan["greedy_tree"], belief, target, universe)
        return evaluate_terminal_path(target_case, target, terminal, policy, branch_identity, universe, planner_config, sandbox_config)
    if policy == "optimal_open_loop_pair":
        terminal = trace_open_loop(plan["optimal_open_loop_pair"], belief, target, universe, planner_config)
        return evaluate_terminal_path(target_case, target, terminal, policy, branch_identity, universe, planner_config, sandbox_config)
    if policy == "exact_bayes_adaptive":
        terminal = trace_tree(plan["exact_tree"], belief, target, universe)
        return evaluate_terminal_path(target_case, target, terminal, policy, branch_identity, universe, planner_config, sandbox_config)
    if policy == "random_open_loop_pair":
        results = []
        for pair in plan["pairs"]:
            terminal = trace_open_loop(pair, belief, target, universe, planner_config)
            results.append(
                evaluate_terminal_path(
                    target_case,
                    target,
                    terminal,
                    policy,
                    f"{branch_identity}|V{pair[0]}-V{pair[1]}",
                    universe,
                    planner_config,
                    sandbox_config,
                )
            )
        return _average_path_results(results)
    if policy == "oracle_class":
        singleton = ((target_case["target_candidate_id"], Fraction(1)),)
        terminal = {"belief": singleton, "terminal_decision": target_case["target_class"], "query_trace": []}
        return evaluate_terminal_path(target_case, target, terminal, policy, branch_identity, universe, planner_config, sandbox_config)
    raise ValueError(f"unknown policy: {policy}")


def _compact_target_result(
    target_case: dict[str, Any], policy: str, result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "target_case_id": target_case["target_case_id"],
        "state_id": target_case["state_id"],
        "target_class": target_case["target_class"],
        "policy": policy,
        "metrics": _serialize_fractions(result["metrics"]),
        "terminal_decision_distribution": _serialize_fractions(result["terminal_decision_distribution"]),
        "route_distribution": _serialize_fractions(result["route_distribution"]),
        "sandbox_transaction_count": result["sandbox_transaction_count"],
        **{key: result[key] for key in SANDBOX_CHECKS},
        "gate_reconstructs": result["gate_reconstructs"],
        "planner_commit_authorization_count": result["planner_commit_authorization_count"],
    }


def evaluate_integration(
    states_artifact: dict[str, Any],
    eligible_artifact: dict[str, Any],
    target_artifact: dict[str, Any],
    planner_config: dict[str, Any],
    sandbox_config: dict[str, Any],
) -> dict[str, Any]:
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    state_by_id = {row["state_id"]: row for row in states_artifact["states"]}
    eligible_ids = list(eligible_artifact["state_ids"])
    targets_by_state: dict[str, list[dict[str, Any]]] = {state_id: [] for state_id in eligible_ids}
    for target in target_artifact["target_cases"]:
        targets_by_state[target["state_id"]].append(target)

    state_policy_results = []
    compact_target_results = []
    source_risk_matches = []
    sandbox_transaction_count = 0
    planner_commit_authorization_count = 0
    for state_id in eligible_ids:
        state = state_by_id[state_id]
        plan = build_state_plan(state["candidate_ids"], universe, planner_config)
        targets = sorted(targets_by_state[state_id], key=lambda row: row["target_case_id"])
        if sum((_target_weight(target) for target in targets), Fraction(0)) != 1:
            raise RuntimeError(f"target weights do not normalize for {state_id}")
        source_case = evaluate_case(
            state_id,
            state["candidate_ids"],
            state["candidate_ids"][0],
            state_id,
            universe,
            planner_config,
        )
        for policy in POLICIES:
            target_results = [
                (target, evaluate_target_policy(state, target, policy, plan, universe, planner_config, sandbox_config))
                for target in targets
            ]
            metrics = {
                key: sum(_target_weight(target) * result["metrics"][key] for target, result in target_results)
                for key in NUMERIC_METRICS
            }
            terminal_distribution: dict[str, Fraction] = {}
            route_distribution: dict[str, Fraction] = {}
            for target, result in target_results:
                weight = _target_weight(target)
                for key, value in result["terminal_decision_distribution"].items():
                    terminal_distribution[key] = terminal_distribution.get(key, Fraction(0)) + weight * value
                for key, value in result["route_distribution"].items():
                    route_distribution[key] = route_distribution.get(key, Fraction(0)) + weight * value
                compact_target_results.append(_compact_target_result(target, policy, result))
                sandbox_transaction_count += result["sandbox_transaction_count"]
                planner_commit_authorization_count += result["planner_commit_authorization_count"]
            checks = {
                key: all(result[key] for _, result in target_results)
                for key in SANDBOX_CHECKS
            }
            checks["gate_reconstructs"] = all(result["gate_reconstructs"] for _, result in target_results)
            expected_source_risk = _fraction_from_payload(source_case["policy_expected_risk"][policy])
            source_match = metrics["recommendation_total_risk"] == expected_source_risk
            source_risk_matches.append(source_match)
            state_policy_results.append(
                {
                    "state_id": state_id,
                    "policy": policy,
                    "target_count": len(targets),
                    "metrics": _serialize_fractions(metrics),
                    "terminal_decision_distribution": _serialize_fractions(terminal_distribution),
                    "route_distribution": _serialize_fractions(route_distribution),
                    "source_V167_expected_risk": fraction_payload(expected_source_risk),
                    "source_V167_risk_reconstructs": source_match,
                    **checks,
                }
            )

    state_policy_results.sort(key=lambda row: (row["state_id"], row["policy"]))
    compact_target_results.sort(key=lambda row: (row["target_case_id"], row["policy"]))
    by_policy: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        rows = [row for row in state_policy_results if row["policy"] == policy]
        by_policy[policy] = {
            "metrics": {
                key: fraction_payload(
                    sum(_fraction_from_payload(row["metrics"][key]) for row in rows) / len(rows)
                )
                for key in NUMERIC_METRICS
            },
            "state_count": len(rows),
        }

    exact_rows = {row["state_id"]: row for row in state_policy_results if row["policy"] == "exact_bayes_adaptive"}
    policy_rows = {
        policy: {row["state_id"]: row for row in state_policy_results if row["policy"] == policy}
        for policy in POLICIES
    }
    nonoracle_comparators = [policy for policy in NONORACLE_POLICIES if policy != "exact_bayes_adaptive"]
    strict_vs_no_query = 0
    no_worse_every = 0
    for state_id, exact_row in exact_rows.items():
        exact_risk = _fraction_from_payload(exact_row["metrics"]["routed_total_risk"])
        no_query_risk = _fraction_from_payload(policy_rows["no_query_bayes_terminal"][state_id]["metrics"]["routed_total_risk"])
        strict_vs_no_query += exact_risk < no_query_risk
        no_worse_every += all(
            exact_risk <= _fraction_from_payload(policy_rows[policy][state_id]["metrics"]["routed_total_risk"])
            for policy in nonoracle_comparators
        )

    class_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for target_class in ("alias", "composition", "provisional_primitive"):
        class_metrics[target_class] = {}
        for policy in POLICIES:
            rows = [row for row in compact_target_results if row["target_class"] == target_class and row["policy"] == policy]
            total_weight = Fraction(len(eligible_ids), 3)
            target_lookup = {row["target_case_id"]: row for row in target_artifact["target_cases"]}
            class_metrics[target_class][policy] = {
                key: fraction_payload(
                    sum(
                        _target_weight(target_lookup[row["target_case_id"]])
                        * _fraction_from_payload(row["metrics"][key])
                        for row in rows
                    )
                    / total_weight
                )
                for key in ("routed_total_risk", "trusted_completion", "expected_queries")
            }

    target_result_sha256 = hashlib.sha256(
        json.dumps(compact_target_results, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "eligible_state_count": len(eligible_ids),
        "target_case_count": len(target_artifact["target_cases"]),
        "target_policy_score_count": len(compact_target_results),
        "policy_count": len(POLICIES),
        "policy_metrics": by_policy,
        "per_class_policy_metrics": class_metrics,
        "state_count_with_strict_exact_routed_risk_improvement_over_no_query": strict_vs_no_query,
        "statewise_exact_no_worse_than_every_nonoracle_policy_rate": no_worse_every / len(eligible_ids),
        "source_V167_recommendation_risk_reconstruction_rate": sum(source_risk_matches) / len(source_risk_matches),
        "target_membership_coverage": len({row["target_case_id"] for row in compact_target_results}) / len(target_artifact["target_cases"]),
        "policy_coverage": len({row["policy"] for row in compact_target_results}) / len(POLICIES),
        "prior_weight_normalization_rate": 1.0,
        "deterministic_gate_reconstruction_rate": sum(row["gate_reconstructs"] for row in compact_target_results) / len(compact_target_results),
        "sandbox_exact_final_state": sum(row["sandbox_exact_final_state"] for row in compact_target_results) / len(compact_target_results),
        "sandbox_invariant_preservation": sum(row["sandbox_invariants_preserved"] for row in compact_target_results) / len(compact_target_results),
        "sandbox_provenance_validity": sum(row["sandbox_provenance_valid"] for row in compact_target_results) / len(compact_target_results),
        "sandbox_restart_verification": sum(row["sandbox_restart_verified"] for row in compact_target_results) / len(compact_target_results),
        "sandbox_authorized_mutations": sum(row["sandbox_authorized_mutations"] for row in compact_target_results) / len(compact_target_results),
        "simulated_sandbox_transaction_count": sandbox_transaction_count,
        "planner_commit_authorization_count": planner_commit_authorization_count,
        "target_result_payload_sha256": target_result_sha256,
    }
    return {
        "state_policy_results": state_policy_results,
        "summary": summary,
        "target_result_payload_sha256": target_result_sha256,
    }


def evaluate_integrity_and_safety_gates(
    evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, bool]:
    summary = evaluation["summary"]
    gates = config["integrityAndSafetyGates"]
    false_route = _fraction_from_payload(summary["policy_metrics"]["exact_bayes_adaptive"]["metrics"]["false_trusted_route"])
    provisional_entry = _fraction_from_payload(summary["policy_metrics"]["exact_bayes_adaptive"]["metrics"]["provisional_sandbox_entry"])
    all_policies_zero_false = all(
        _fraction_from_payload(value["metrics"]["false_trusted_route"]) == 0
        and _fraction_from_payload(value["metrics"]["provisional_sandbox_entry"]) == 0
        for value in summary["policy_metrics"].values()
    )
    return {
        "eligible_state_count": summary["eligible_state_count"] == gates["requiredEligibleStateCount"],
        "target_case_count": summary["target_case_count"] == gates["requiredTargetCaseCount"],
        "target_membership_coverage": summary["target_membership_coverage"] == gates["requiredTargetMembershipCoverage"],
        "policy_coverage": summary["policy_coverage"] == gates["requiredPolicyCoverage"],
        "prior_weight_normalization": summary["prior_weight_normalization_rate"] == gates["requiredPriorWeightNormalization"],
        "false_trusted_route": false_route == gates["requiredFalseTrustedRouteProbability"] and all_policies_zero_false,
        "provisional_sandbox_entry": provisional_entry == gates["requiredProvisionalSandboxEntryProbability"] and all_policies_zero_false,
        "planner_has_no_commit_authority": summary["planner_commit_authorization_count"] == gates["requiredPlannerCommitAuthorizationCount"],
        "sandbox_exact_final_state": summary["sandbox_exact_final_state"] == gates["requiredSandboxExactFinalState"],
        "sandbox_invariant_preservation": summary["sandbox_invariant_preservation"] == gates["requiredSandboxInvariantPreservation"],
        "sandbox_provenance_validity": summary["sandbox_provenance_validity"] == gates["requiredSandboxProvenanceValidity"],
        "sandbox_restart_verification": summary["sandbox_restart_verification"] == gates["requiredSandboxRestartVerification"],
        "deterministic_gate_reconstruction": summary["deterministic_gate_reconstruction_rate"] == gates["requiredDeterministicGateReconstruction"],
        "source_policy_risk_reconstruction": summary["source_V167_recommendation_risk_reconstruction_rate"] == 1.0,
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


def evaluate_benefit_thresholds(evaluation: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    summary = evaluation["summary"]
    threshold = config["benefitThresholds"]
    metrics = summary["policy_metrics"]
    exact_risk = _fraction_from_payload(metrics["exact_bayes_adaptive"]["metrics"]["routed_total_risk"])
    no_query_risk = _fraction_from_payload(metrics["no_query_bayes_terminal"]["metrics"]["routed_total_risk"])
    exact_completion = _fraction_from_payload(metrics["exact_bayes_adaptive"]["metrics"]["trusted_completion"])
    no_query_completion = _fraction_from_payload(metrics["no_query_bayes_terminal"]["metrics"]["trusted_completion"])
    return {
        "strict_mean_risk_improvement_over_no_query": exact_risk < no_query_risk,
        "strict_completion_improvement_over_no_query": exact_completion > no_query_completion,
        "statewise_improvement_exists": summary["state_count_with_strict_exact_routed_risk_improvement_over_no_query"] >= threshold["minimumStateCountWithStrictRoutedRiskImprovementOverNoQuery"],
    }


def evaluate_strong_thresholds(evaluation: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    summary = evaluation["summary"]
    metrics = summary["policy_metrics"]
    exact = _fraction_from_payload(metrics["exact_bayes_adaptive"]["metrics"]["routed_total_risk"])
    threshold = config["strongIntegrationThresholds"]
    return {
        "no_worse_than_greedy_mean": exact <= _fraction_from_payload(metrics["greedy_class_information_gain"]["metrics"]["routed_total_risk"]),
        "no_worse_than_optimal_open_loop_mean": exact <= _fraction_from_payload(metrics["optimal_open_loop_pair"]["metrics"]["routed_total_risk"]),
        "no_worse_than_random_open_loop_mean": exact <= _fraction_from_payload(metrics["random_open_loop_pair"]["metrics"]["routed_total_risk"]),
        "pointwise_no_worse_than_every_nonoracle": summary["statewise_exact_no_worse_than_every_nonoracle_policy_rate"] >= threshold["minimumStatewiseNoWorseThanEveryNonOraclePolicyRate"],
    }


__all__ = [
    "POLICIES",
    "build_state_plan",
    "deterministic_consensus_route",
    "evaluate_benefit_thresholds",
    "evaluate_integration",
    "evaluate_integrity_and_safety_gates",
    "evaluate_strong_thresholds",
    "evaluate_target_policy",
    "run_trusted_sandbox_route",
    "trace_open_loop",
    "trace_tree",
]
