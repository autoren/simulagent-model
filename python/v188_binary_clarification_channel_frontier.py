from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import hashlib
import heapq
import json
import math
from typing import Any

from v187_clean_typed_clarification_planner import (
    Question,
    _mass,
    _split,
    build_problem,
    evaluate_adaptive,
    evaluate_open_loop,
    solve_exact,
    solve_open_loop,
)


def build_frozen_problem(
    question_payload: dict[str, Any],
    vector_payload: dict[str, list[int]],
    development_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    frontier = config["costFrontier"]
    adapter = {
        "problem": {
            "maximumTypedQuestionCount": frontier["maximumTypedQuestionCount"],
            "typedQuestionCost": frontier["V187ControlQuestionCost"],
            "genericTrustedClarificationCost": frontier["genericTrustedClarificationCost"],
            "safeDeferralCost": 0.50,
        }
    }
    return build_problem(question_payload, vector_payload, development_payload, adapter)


def information_controls(problem: dict[str, Any]) -> dict[str, Any]:
    prior = problem["prior"]
    entropy = -sum(float(weight) * math.log2(float(weight)) for weight in prior.values())
    heap = []
    serial = 0
    for contract_id in problem["contract_ids"]:
        node = {"leaf": contract_id}
        heapq.heappush(heap, (prior[contract_id], (contract_id,), serial, node))
        serial += 1
    while len(heap) > 1:
        left_weight, left_leaves, _, left = heapq.heappop(heap)
        right_weight, right_leaves, _, right = heapq.heappop(heap)
        leaves = tuple(sorted(left_leaves + right_leaves))
        node = {"left": left, "right": right, "leaves": leaves}
        heapq.heappush(heap, (left_weight + right_weight, leaves, serial, node))
        serial += 1
    _, _, _, root = heap[0]
    depths: dict[str, int] = {}

    def walk(node: dict[str, Any], depth: int) -> None:
        if "leaf" in node:
            depths[node["leaf"]] = depth
            return
        walk(node["left"], depth + 1)
        walk(node["right"], depth + 1)

    walk(root, 0)
    expected = sum((prior[key] * depths[key] for key in problem["contract_ids"]), Fraction(0))
    return {
        "shannon_entropy_bits": entropy,
        "huffman_expected_depth_fraction": f"{expected.numerator}/{expected.denominator}",
        "huffman_expected_depth": float(expected),
        "huffman_minimum_depth": min(depths.values()),
        "huffman_maximum_depth": max(depths.values()),
        "huffman_depth_by_contract": dict(sorted(depths.items())),
        "entropy_lower_bound_pass": entropy <= float(expected) + 1e-15,
        "entropy_plus_one_upper_bound_pass": float(expected) < entropy + 1.0,
    }


def restricted_exact_depth_tree(problem: dict[str, Any], horizon: int) -> dict[str, Any]:
    questions: tuple[Question, ...] = problem["questions"]

    @lru_cache(maxsize=None)
    def value(state: tuple[str, ...]) -> tuple[Fraction, str]:
        if len(state) == 1:
            return Fraction(0), "SINGLETON"
        parent_mass = _mass(problem, state)
        best_cost: Fraction | None = None
        best_action: str | None = None
        for question in questions:
            zero, one = _split(problem, state, question)
            if not zero or not one:
                continue
            cost = Fraction(1)
            cost += _mass(problem, zero) / parent_mass * value(zero)[0]
            cost += _mass(problem, one) / parent_mass * value(one)[0]
            if best_cost is None or cost < best_cost or (cost == best_cost and question.question_id < best_action):
                best_cost, best_action = cost, question.question_id
        if best_cost is None or best_action is None:
            raise RuntimeError(f"nonidentifying V188 state: {state}")
        return best_cost, best_action

    by_id = {question.question_id: question for question in questions}
    rows = {}
    for target in problem["contract_ids"]:
        state = problem["contract_ids"]
        trace = []
        while len(state) > 1:
            if len(trace) >= horizon:
                raise RuntimeError("V188 restricted exact tree exceeded frozen horizon")
            action = value(state)[1]
            question = by_id[action]
            answer = question.column[problem["contract_index"][target]]
            zero, one = _split(problem, state, question)
            next_state = one if answer else zero
            trace.append({"question_id": action, "answer": answer, "pre_size": len(state), "post_size": len(next_state)})
            state = next_state
        rows[target] = {
            "target_contract_id": target,
            "depth": len(trace),
            "target_retained": target in state,
            "exact_singleton": state == (target,),
            "trace": trace,
        }
    root_cost = value(problem["contract_ids"])[0]
    depths = [row["depth"] for row in rows.values()]
    return {
        "expected_depth_fraction": f"{root_cost.numerator}/{root_cost.denominator}",
        "expected_depth": float(root_cost),
        "minimum_depth": min(depths),
        "maximum_depth": max(depths),
        "leaf_count": sum(row["exact_singleton"] for row in rows.values()),
        "target_retention_rate": sum(row["target_retained"] for row in rows.values()) / len(rows),
        "exactness_rate": sum(row["exact_singleton"] for row in rows.values()) / len(rows),
        "by_target": rows,
        "reachable_state_count": value.cache_info().currsize,
    }


def _signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cost_frontier(problem: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    frontier = config["costFrontier"]
    low, high = frontier["questionCostGridNumeratorsInclusive"]
    denominator = frontier["questionCostGridDenominator"]
    rows = []
    previous_signature = None
    breakpoints = []
    for numerator in range(low, high + 1):
        cell_problem = dict(problem)
        cell_problem["typed_cost"] = Fraction(numerator, denominator)
        cell_problem["horizon"] = frontier["maximumTypedQuestionCount"]
        exact = solve_exact(cell_problem)
        open_loop = solve_open_loop(cell_problem)
        exact_paths = {target: evaluate_adaptive(cell_problem, exact, target) for target in problem["contract_ids"]}
        open_paths = {target: evaluate_open_loop(cell_problem, open_loop, target) for target in problem["contract_ids"]}
        exact_cost = exact["value"]
        open_cost = open_loop["value"]
        typed_rate = sum(problem["prior"][target] for target, row in exact_paths.items() if row["terminal_mode"] == "TYPED_SINGLETON")
        mean_questions = sum(problem["prior"][target] * row["question_count"] for target, row in exact_paths.items())
        signature_payload = {
            "exact": {target: [(s["question_id"], s["answer"]) for s in row["trace"]] + [(row["terminal_mode"], None)] for target, row in exact_paths.items()},
            "open_loop_sequence": list(open_loop["sequence"]),
        }
        signature = _signature(signature_payload)
        root_action = exact["choice"](problem["contract_ids"], cell_problem["horizon"])[1]
        row = {
            "question_cost_fraction": f"{numerator}/{denominator}",
            "question_cost": numerator / denominator,
            "exact_cost_fraction": f"{exact_cost.numerator}/{exact_cost.denominator}",
            "exact_cost": float(exact_cost),
            "open_loop_cost_fraction": f"{open_cost.numerator}/{open_cost.denominator}",
            "open_loop_cost": float(open_cost),
            "exact_improvement_over_generic": float(problem["generic_cost"] - exact_cost),
            "exact_advantage_over_open_loop": float(open_cost - exact_cost),
            "exact_typed_completion_rate": float(typed_rate),
            "exact_mean_question_count": float(mean_questions),
            "exact_root_action": root_action,
            "open_loop_sequence": list(open_loop["sequence"]),
            "exact_final_exactness_rate": sum(problem["prior"][target] for target, item in exact_paths.items() if item["final_exact"]).__float__(),
            "exact_target_retention_rate": sum(problem["prior"][target] for target, item in exact_paths.items() if item["target_retained"]).__float__(),
            "open_loop_final_exactness_rate": sum(problem["prior"][target] for target, item in open_paths.items() if item["final_exact"]).__float__(),
            "open_loop_target_retention_rate": sum(problem["prior"][target] for target, item in open_paths.items() if item["target_retained"]).__float__(),
            "complete_policy_signature_sha256": signature,
        }
        rows.append(row)
        if signature != previous_signature:
            breakpoints.append({
                "grid_index": len(rows) - 1,
                "question_cost_fraction": row["question_cost_fraction"],
                "question_cost": row["question_cost"],
                "exact_root_action": root_action,
                "open_loop_sequence": row["open_loop_sequence"],
                "complete_policy_signature_sha256": signature,
            })
            previous_signature = signature
    positive = [row for row in rows if row["exact_improvement_over_generic"] > 0 and row["exact_typed_completion_rate"] > 0]
    adaptive = [row for row in positive if row["exact_advantage_over_open_loop"] >= frontier["adaptiveAdvantageThreshold"]]
    v187 = next(row for row in rows if row["question_cost"] == frontier["V187ControlQuestionCost"])
    return {
        "grid": rows,
        "policy_breakpoints": breakpoints,
        "positive_value_cell_count": len(positive),
        "adaptive_advantage_cell_count": len(adaptive),
        "largest_positive_value_question_cost": max((row["question_cost"] for row in positive), default=None),
        "largest_adaptive_advantage_question_cost": max((row["question_cost"] for row in adaptive), default=None),
        "v187_control_cell": v187,
    }


def evaluate_frontier(problem: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    info = information_controls(problem)
    restricted = restricted_exact_depth_tree(problem, config["restrictedExactDepthTree"]["horizon"])
    frontier = cost_frontier(problem, config)
    v187_source = config["_source_v187_result"]
    v187_exact = v187_source["summary"]["policy_summary"]["exact_adaptive"]
    successor = config["successorRule"]
    largest_positive = frontier["largest_positive_value_question_cost"]
    successor_checks = {
        "V187_is_at_generic_boundary": v187_exact["mean_cost"] == config["costFrontier"]["genericTrustedClarificationCost"],
        "some_lower_binary_cost_has_positive_value": frontier["positive_value_cell_count"] > 0,
        "target_informed_oracle_gap_is_positive": v187_source["summary"]["policy_summary"]["target_informed_oracle"]["mean_cost"] < v187_exact["mean_cost"],
        "binary_break_even_is_below_V187_question_cost": largest_positive is not None and largest_positive < config["costFrontier"]["V187ControlQuestionCost"],
    }
    summary = {
        "contract_count": len(problem["contract_ids"]),
        "positive_prior_contract_count": sum(value > 0 for value in problem["prior"].values()),
        "raw_question_count": config["frozenPopulation"]["rawQuestionCount"],
        "partition_distinct_question_count": len(problem["questions"]),
        "information_controls": info,
        "restricted_exact_tree": {key: value for key, value in restricted.items() if key != "by_target"},
        "cost_frontier": {key: value for key, value in frontier.items() if key != "grid"},
        "successor_checks": successor_checks,
        "authorize_multiway_feasibility_design": all(successor_checks.values()),
        "utterance_or_dialogue_language_read_count": 0,
        "protected_utterance_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    return {"summary": summary, "information": info, "restricted": restricted, "frontier": frontier}


def audit_frontier(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    gates = config["frontierGates"]
    info = summary["information_controls"]
    restricted = summary["restricted_exact_tree"]
    frontier = result["frontier"]
    v187 = frontier["v187_control_cell"]
    all_grid_exact = min(row["exact_final_exactness_rate"] for row in frontier["grid"])
    all_grid_retention = min(row["exact_target_retention_rate"] for row in frontier["grid"])
    checks = {
        "frozen_population_and_question_partitions_are_exact": bool(
            summary["contract_count"] == gates["requiredContractCount"]
            and summary["positive_prior_contract_count"] == gates["requiredPositivePriorContractCount"]
            and summary["raw_question_count"] == gates["requiredRawQuestionCount"]
            and summary["partition_distinct_question_count"] == gates["requiredPartitionDistinctQuestionCount"]
        ),
        "entropy_and_Huffman_controls_pass": bool(
            (info["shannon_entropy_bits"] > 0) == gates["requiredEntropyPositive"]
            and info["entropy_lower_bound_pass"]
            and info["entropy_plus_one_upper_bound_pass"]
        ),
        "restricted_exact_tree_is_complete_and_safe": bool(
            restricted["leaf_count"] == gates["requiredRestrictedExactLeafCount"]
            and restricted["target_retention_rate"] == gates["requiredRestrictedTargetRetentionRate"]
            and restricted["exactness_rate"] == gates["requiredRestrictedExactnessRate"]
            and restricted["maximum_depth"] <= gates["maximumRestrictedDepth"]
        ),
        "cost_grid_is_complete_exact_and_safe": bool(
            len(frontier["grid"]) == gates["requiredGridCellCount"]
            and all_grid_exact == gates["requiredAllGridExactnessRate"]
            and all_grid_retention == gates["requiredAllGridTargetRetentionRate"]
        ),
        "zero_cost_and_V187_controls_reconstruct": bool(
            frontier["grid"][0]["exact_final_exactness_rate"] == 1.0
            and v187["exact_cost"] == config["costFrontier"]["V187ControlExactCost"]
            and v187["exact_typed_completion_rate"] == config["costFrontier"]["V187ControlTypedCompletionRate"]
        ),
        "policy_breakpoints_are_explicit": len(frontier["policy_breakpoints"]) >= gates["minimumPolicyBreakpointCount"],
        "language_model_authority_and_effect_access_is_zero": all(summary[key] == gates[gate] for key, gate in (
            ("utterance_or_dialogue_language_read_count", "maximumUtteranceOrDialogueLanguageReadCount"),
            ("protected_utterance_language_read_count", "maximumProtectedUtteranceLanguageReadCount"),
            ("model_load_count", "maximumModelLoadCount"),
            ("model_generation_count", "maximumModelGenerationCount"),
            ("API_call_count", "maximumAPICallCount"),
            ("training_run_count", "maximumTrainingRunCount"),
            ("ontology_registration_count", "maximumOntologyRegistrationCount"),
            ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
            ("service_call_count", "maximumServiceCallCount"),
            ("external_side_effect_count", "maximumExternalSideEffectCount"),
            ("actual_execution_count", "maximumActualExecutionCount"),
        )),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = ["audit_frontier", "build_frozen_problem", "cost_frontier", "evaluate_frontier", "information_controls", "restricted_exact_depth_tree"]
