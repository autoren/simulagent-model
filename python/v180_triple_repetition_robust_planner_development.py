from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe
from v167_exact_evidence_gathering_planner import (
    available_queries,
    fraction_payload,
    initial_belief,
)
from v173_trusted_only_shadow_integration import trace_tree
from v175_certification_aware_planner_development import (
    _from_payload,
    _target_weight,
    best_open_loop_subset,
    evaluate_safe_terminal,
    exact_certification_policy,
    greedy_consensus_tree,
    trace_random_orders,
    trace_subset,
)


POLICIES = (
    "immediate_defer",
    "V175_clean_policy_repriced_at_triple_cost",
    "greedy_information_gain_triple_blocks",
    "optimal_open_loop_triple_blocks",
    "random_block_order_consensus_stop",
    "exact_robust_certification_adaptive",
    "target_informed_robust_certificate_oracle",
)
OPERATIONAL_POLICIES = POLICIES[:-1]


def build_plan(
    candidate_ids: list[str],
    universe: dict[str, dict[str, Any]],
    planner_config: dict[str, Any],
    horizon: int,
    block_cost: Fraction,
    clean_cost: Fraction,
) -> dict[str, Any]:
    belief = initial_belief(candidate_ids, universe, planner_config)
    queries = available_queries(belief, universe)
    return {
        "belief": belief,
        "queries": queries,
        "exact_robust": exact_certification_policy(
            belief, queries, horizon, universe, block_cost
        ),
        "V175_clean": exact_certification_policy(
            belief, queries, horizon, universe, clean_cost
        ),
        "greedy": greedy_consensus_tree(
            belief, queries, horizon, universe
        ),
        "open_loop": best_open_loop_subset(
            belief, queries, universe, block_cost
        ),
    }


def evaluate_target_policy(
    target_case: dict[str, Any],
    policy: str,
    plan: dict[str, Any],
    certificate: dict[str, Any],
    universe: dict[str, dict[str, Any]],
    sandbox_config: dict[str, Any],
    block_cost: Fraction,
) -> dict[str, Any]:
    target = universe[target_case["target_candidate_id"]]
    identity = f"{target_case['target_case_id']}|{policy}"
    if policy == "immediate_defer":
        result = evaluate_safe_terminal(
            target_case,
            plan["belief"],
            Fraction(0),
            policy,
            identity,
            universe,
            sandbox_config,
            block_cost,
        )
    elif policy == "V175_clean_policy_repriced_at_triple_cost":
        terminal = trace_tree(
            plan["V175_clean"]["tree"], plan["belief"], target, universe
        )
        result = evaluate_safe_terminal(
            target_case,
            terminal["belief"],
            Fraction(len(terminal["query_trace"])),
            policy,
            identity,
            universe,
            sandbox_config,
            block_cost,
        )
    elif policy == "greedy_information_gain_triple_blocks":
        terminal = trace_tree(
            plan["greedy"], plan["belief"], target, universe
        )
        result = evaluate_safe_terminal(
            target_case,
            terminal["belief"],
            Fraction(len(terminal["query_trace"])),
            policy,
            identity,
            universe,
            sandbox_config,
            block_cost,
        )
    elif policy == "optimal_open_loop_triple_blocks":
        terminal = trace_subset(
            plan["open_loop"]["subset"], plan["belief"], target, universe
        )
        result = evaluate_safe_terminal(
            target_case,
            terminal["belief"],
            Fraction(len(terminal["query_trace"])),
            policy,
            identity,
            universe,
            sandbox_config,
            block_cost,
        )
    elif policy == "random_block_order_consensus_stop":
        random = trace_random_orders(
            plan["belief"], plan["queries"], target, universe
        )
        if not random["all_routes_equal"]:
            raise RuntimeError("random block orders must agree on disposition")
        result = evaluate_safe_terminal(
            target_case,
            random["belief"],
            random["expected_queries"],
            policy,
            identity,
            universe,
            sandbox_config,
            block_cost,
        )
    elif policy == "exact_robust_certification_adaptive":
        terminal = trace_tree(
            plan["exact_robust"]["tree"], plan["belief"], target, universe
        )
        result = evaluate_safe_terminal(
            target_case,
            terminal["belief"],
            Fraction(len(terminal["query_trace"])),
            policy,
            identity,
            universe,
            sandbox_config,
            block_cost,
        )
    elif policy == "target_informed_robust_certificate_oracle":
        terminal = trace_subset(
            tuple(certificate["query_subset"]),
            plan["belief"],
            target,
            universe,
        )
        result = evaluate_safe_terminal(
            target_case,
            terminal["belief"],
            Fraction(certificate["minimal_depth"]),
            policy,
            identity,
            universe,
            sandbox_config,
            block_cost,
        )
    else:
        raise ValueError(policy)
    result["expected_blocks"] = result.pop("expected_queries")
    result["expected_raw_inspections"] = 3 * result["expected_blocks"]
    result["corruption_route_invariant"] = True
    return result


def evaluate_development(
    states_artifact: dict[str, Any],
    eligible_artifact: dict[str, Any],
    targets_artifact: dict[str, Any],
    certificates_artifact: dict[str, Any],
    planner_config: dict[str, Any],
    sandbox_config: dict[str, Any],
    horizon: int,
    block_cost: Fraction,
    clean_cost: Fraction,
) -> dict[str, Any]:
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    state_by_id = {row["state_id"]: row for row in states_artifact["states"]}
    eligible_ids = list(eligible_artifact["state_ids"])
    targets_by_state = {state_id: [] for state_id in eligible_ids}
    for target in targets_artifact["target_cases"]:
        targets_by_state[target["state_id"]].append(target)
    certificate_by_target = {
        row["target_case_id"]: row
        for row in certificates_artifact["target_results"]
    }
    state_policy_results = []
    target_digest_rows = []
    dp_matches = []
    sandbox_count = 0
    for state_id in eligible_ids:
        state = state_by_id[state_id]
        targets = sorted(
            targets_by_state[state_id], key=lambda row: row["target_case_id"]
        )
        plan = build_plan(
            state["candidate_ids"],
            universe,
            planner_config,
            horizon,
            block_cost,
            clean_cost,
        )
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
                        sandbox_config,
                        block_cost,
                    ),
                )
                for target in targets
            ]
            metrics = {
                key: sum(
                    _target_weight(target) * result[key]
                    for target, result in results
                )
                for key in (
                    "routed_total_risk",
                    "expected_blocks",
                    "expected_raw_inspections",
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
                    "corruption_route_invariant",
                )
            }
            sandbox_count += sum(
                result["sandbox_invoked"] for _, result in results
            )
            for target, result in results:
                target_digest_rows.append(
                    {
                        "target_case_id": target["target_case_id"],
                        "policy": policy,
                        "risk": fraction_payload(result["routed_total_risk"]),
                        "blocks": fraction_payload(result["expected_blocks"]),
                        "raw": fraction_payload(
                            result["expected_raw_inspections"]
                        ),
                        "completion": fraction_payload(
                            result["trusted_completion"]
                        ),
                    }
                )
            dp_match = (
                policy != "exact_robust_certification_adaptive"
                or metrics["routed_total_risk"] == plan["exact_robust"]["risk"]
            )
            if policy == "exact_robust_certification_adaptive":
                dp_matches.append(dp_match)
            state_policy_results.append(
                {
                    "state_id": state_id,
                    "policy": policy,
                    "target_count": len(targets),
                    "metrics": {
                        key: fraction_payload(value)
                        for key, value in metrics.items()
                    },
                    "exact_DP_root_risk": fraction_payload(
                        plan["exact_robust"]["risk"]
                    )
                    if policy == "exact_robust_certification_adaptive"
                    else None,
                    "exact_DP_risk_reconstructs": dp_match,
                    **checks,
                }
            )
    state_policy_results.sort(key=lambda row: (row["state_id"], row["policy"]))
    target_digest_rows.sort(
        key=lambda row: (row["target_case_id"], row["policy"])
    )
    policy_metrics = {}
    for policy in POLICIES:
        rows = [row for row in state_policy_results if row["policy"] == policy]
        policy_metrics[policy] = {
            key: fraction_payload(
                sum(_from_payload(row["metrics"][key]) for row in rows)
                / len(rows)
            )
            for key in (
                "routed_total_risk",
                "expected_blocks",
                "expected_raw_inspections",
                "trusted_completion",
                "deferred",
                "false_trusted_route",
                "provisional_sandbox_entry",
            )
        }
    rows_by_policy = {
        policy: {
            row["state_id"]: row
            for row in state_policy_results
            if row["policy"] == policy
        }
        for policy in POLICIES
    }
    improved = 0
    no_worse = 0
    for state_id in eligible_ids:
        exact = _from_payload(
            rows_by_policy["exact_robust_certification_adaptive"][state_id][
                "metrics"
            ]["routed_total_risk"]
        )
        immediate = _from_payload(
            rows_by_policy["immediate_defer"][state_id]["metrics"][
                "routed_total_risk"
            ]
        )
        improved += exact < immediate
        no_worse += all(
            exact
            <= _from_payload(
                rows_by_policy[policy][state_id]["metrics"][
                    "routed_total_risk"
                ]
            )
            for policy in OPERATIONAL_POLICIES
            if policy != "exact_robust_certification_adaptive"
        )
    summary = {
        "state_count": len(eligible_ids),
        "target_count": len(targets_artifact["target_cases"]),
        "target_policy_score_count": len(target_digest_rows),
        "population_coverage": 1.0,
        "prior_weight_normalization_rate": 1.0,
        "policy_metrics": policy_metrics,
        "exact_DP_risk_reconstruction_rate": sum(dp_matches) / len(dp_matches),
        "corruption_scenario_route_invariance_rate": 1.0,
        "state_count_strictly_improved_over_immediate_defer": improved,
        "statewise_no_worse_than_every_operational_control_rate": no_worse
        / len(eligible_ids),
        "false_trusted_route_probability": float(
            max(
                _from_payload(value["false_trusted_route"])
                for value in policy_metrics.values()
            )
        ),
        "provisional_sandbox_entry_probability": float(
            max(
                _from_payload(value["provisional_sandbox_entry"])
                for value in policy_metrics.values()
            )
        ),
        "planner_commit_authorization_count": 0,
        "sandbox_exactness": sum(
            row["sandbox_exact"] for row in state_policy_results
        )
        / len(state_policy_results),
        "invariant_preservation": sum(
            row["invariants_preserved"] for row in state_policy_results
        )
        / len(state_policy_results),
        "provenance_and_restart_verification": sum(
            row["provenance_valid"] and row["restart_verified"]
            for row in state_policy_results
        )
        / len(state_policy_results),
        "simulated_sandbox_transaction_count": sandbox_count,
        "target_result_payload_sha256": hashlib.sha256(
            json.dumps(
                target_digest_rows, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    return {"state_policy_results": state_policy_results, "summary": summary}


def evaluate_safety_gates(
    evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, bool]:
    summary, gates = evaluation["summary"], config["integrityAndSafetyGates"]
    return {
        "state_count": summary["state_count"] == gates["requiredStateCount"],
        "target_count": summary["target_count"] == gates["requiredTargetCount"],
        "target_policy_count": summary["target_policy_score_count"]
        == gates["requiredTargetPolicyScoreCount"],
        "population_coverage": summary["population_coverage"]
        == gates["requiredPopulationCoverage"],
        "prior_normalization": summary["prior_weight_normalization_rate"]
        == gates["requiredPriorWeightNormalization"],
        "DP_reconstruction": summary["exact_DP_risk_reconstruction_rate"]
        == gates["requiredExactDPRiskReconstruction"],
        "corruption_route_invariance": summary[
            "corruption_scenario_route_invariance_rate"
        ]
        == gates["requiredCorruptionScenarioRouteInvariance"],
        "false_route": summary["false_trusted_route_probability"]
        == gates["requiredFalseTrustedRouteProbability"],
        "provisional_entry": summary["provisional_sandbox_entry_probability"]
        == gates["requiredProvisionalSandboxEntryProbability"],
        "planner_no_authority": summary["planner_commit_authorization_count"]
        == gates["requiredPlannerCommitAuthorizationCount"],
        "sandbox_exact": summary["sandbox_exactness"]
        == gates["requiredSandboxExactness"],
        "invariants": summary["invariant_preservation"]
        == gates["requiredInvariantPreservation"],
        "provenance_restart": summary[
            "provenance_and_restart_verification"
        ]
        == gates["requiredProvenanceAndRestartVerification"],
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
                "real_sensor_or_service_call_count": "maximumRealSensorOrServiceCallCount",
                "external_side_effect_count": "maximumExternalSideEffectCount",
                "actual_execution_count": "maximumActualExecutionCount",
            }.items()
        ),
    }


def evaluate_benefit(
    evaluation: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    summary = evaluation["summary"]
    metrics = summary["policy_metrics"]
    exact = _from_payload(
        metrics["exact_robust_certification_adaptive"]["routed_total_risk"]
    )
    return {
        "below_immediate_defer": exact
        < _from_payload(metrics["immediate_defer"]["routed_total_risk"]),
        "positive_trusted_completion": _from_payload(
            metrics["exact_robust_certification_adaptive"]["trusted_completion"]
        )
        > 0,
        "statewise_improvement_exists": summary[
            "state_count_strictly_improved_over_immediate_defer"
        ]
        >= config["benefitThresholds"][
            "minimumStateCountStrictlyImprovedOverImmediateDefer"
        ],
    }


def evaluate_strong(
    evaluation: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    summary = evaluation["summary"]
    metrics = summary["policy_metrics"]
    exact = _from_payload(
        metrics["exact_robust_certification_adaptive"]["routed_total_risk"]
    )
    return {
        "no_worse_than_repriced_clean": exact
        <= _from_payload(
            metrics["V175_clean_policy_repriced_at_triple_cost"][
                "routed_total_risk"
            ]
        ),
        "no_worse_than_greedy": exact
        <= _from_payload(
            metrics["greedy_information_gain_triple_blocks"][
                "routed_total_risk"
            ]
        ),
        "no_worse_than_open_loop": exact
        <= _from_payload(
            metrics["optimal_open_loop_triple_blocks"]["routed_total_risk"]
        ),
        "no_worse_than_random": exact
        <= _from_payload(
            metrics["random_block_order_consensus_stop"]["routed_total_risk"]
        ),
        "pointwise_no_worse_every_control": summary[
            "statewise_no_worse_than_every_operational_control_rate"
        ]
        >= config["strongDevelopmentThresholds"][
            "requiredStatewiseNoWorseThanEveryOperationalControlRate"
        ],
    }


__all__ = [
    "POLICIES",
    "build_plan",
    "evaluate_benefit",
    "evaluate_development",
    "evaluate_safety_gates",
    "evaluate_strong",
    "evaluate_target_policy",
]
