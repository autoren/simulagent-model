from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import math
from typing import Any


@dataclass(frozen=True)
class MultiwayQuestion:
    question_id: str
    field: str
    outcomes: tuple[Any, ...]


def build_problem(contract_catalog: dict[str, Any], bindings_payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    contracts = sorted(contract_catalog["contracts"], key=lambda row: row["capability_contract_id"])
    contract_ids = tuple(row["capability_contract_id"] for row in contracts)
    payload_by_id = {row["capability_contract_id"]: row["semantic_payload"] for row in contracts}
    fields = {
        "domain": lambda row: row["domain"],
        "intent_concept": lambda row: row["normalized_intent_name"],
        "transactionality": lambda row: row["is_transactional"],
    }
    questions = []
    for field in config["multiwayQuestions"]["allQuestionSet"]:
        outcomes = tuple(fields[field](payload_by_id[contract_id]) for contract_id in contract_ids)
        if len(set(outcomes)) > 1:
            questions.append(MultiwayQuestion(f"M189_{field}", field, outcomes))
    observed = [row for row in bindings_payload["bindings"] if row["observation_available"]]
    counts = {contract_id: 0 for contract_id in contract_ids}
    for row in observed:
        counts[row["target_contract_id"]] += 1
    if any(value <= 0 for value in counts.values()):
        raise ValueError("V189 requires positive prior mass on all contracts")
    total = sum(counts.values())
    prior = {key: counts[key] / total for key in contract_ids}
    return {
        "contract_ids": contract_ids,
        "index": {key: index for index, key in enumerate(contract_ids)},
        "questions": tuple(questions),
        "coarse_question_ids": tuple(f"M189_{field}" for field in config["multiwayQuestions"]["coarseQuestionSet"]),
        "prior_counts": counts,
        "prior": prior,
        "generic_cost": config["pricing"]["genericTrustedClarificationCost"],
        "horizon": config["pricing"]["maximumMultiwayTurns"],
    }


def scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    pricing = config["pricing"]
    low, high = pricing["turnOverheadGridNumeratorsInclusive"]
    denominator = pricing["turnOverheadGridDenominator"]
    rows = [
        {"scenario_id": f"bit_slot_o{numerator:02d}", "rule": "bit_slot", "turn_overhead": numerator / denominator}
        for numerator in range(low, high + 1)
    ]
    rows.append({"scenario_id": "entropy_lower_bound_o00", "rule": "entropy_lower_bound", "turn_overhead": 0.0})
    return rows


def _mass(problem: dict[str, Any], state: tuple[str, ...]) -> float:
    return sum(problem["prior"][item] for item in state)


def _partition(problem: dict[str, Any], state: tuple[str, ...], question: MultiwayQuestion) -> dict[Any, tuple[str, ...]]:
    groups: dict[Any, list[str]] = {}
    for contract_id in state:
        outcome = question.outcomes[problem["index"][contract_id]]
        groups.setdefault(outcome, []).append(contract_id)
    return {key: tuple(value) for key, value in groups.items()}


def question_cost(problem: dict[str, Any], state: tuple[str, ...], question: MultiwayQuestion, scenario: dict[str, Any]) -> float:
    groups = _partition(problem, state, question)
    k = len(groups)
    if k <= 1:
        return 0.0
    if scenario["rule"] == "bit_slot":
        overhead = scenario["turn_overhead"]
        marginal = 0.10 - overhead
        return overhead + marginal * math.ceil(math.log2(k))
    if scenario["rule"] == "entropy_lower_bound":
        parent = _mass(problem, state)
        entropy = 0.0
        for child in groups.values():
            p = _mass(problem, child) / parent
            entropy -= p * math.log2(p)
        return 0.10 * entropy
    raise ValueError(scenario["rule"])


def solve_exact(problem: dict[str, Any], scenario: dict[str, Any], question_ids: tuple[str, ...]) -> dict[str, Any]:
    questions = {question.question_id: question for question in problem["questions"] if question.question_id in question_ids}

    @lru_cache(maxsize=None)
    def value(state: tuple[str, ...], remaining: int) -> tuple[float, str]:
        if len(state) == 1:
            return 0.0, "SINGLETON"
        best_cost, best_action = problem["generic_cost"], "GENERIC"
        if remaining == 0:
            return best_cost, best_action
        parent = _mass(problem, state)
        for question_id in sorted(questions):
            question = questions[question_id]
            groups = _partition(problem, state, question)
            if len(groups) <= 1:
                continue
            cost = question_cost(problem, state, question, scenario)
            cost += sum(_mass(problem, child) / parent * value(child, remaining - 1)[0] for child in groups.values())
            if cost < best_cost - 1e-12 or (abs(cost - best_cost) <= 1e-12 and best_action not in {"GENERIC", "SINGLETON"} and question_id < best_action):
                best_cost, best_action = cost, question_id
        return best_cost, best_action

    root = problem["contract_ids"]
    return {"value": value(root, problem["horizon"])[0], "choice": value, "root": root, "questions": questions}


def evaluate_exact(problem: dict[str, Any], solver: dict[str, Any], scenario: dict[str, Any], target: str) -> dict[str, Any]:
    state = solver["root"]
    remaining = problem["horizon"]
    cost = 0.0
    trace = []
    while True:
        action = solver["choice"](state, remaining)[1]
        if action == "SINGLETON":
            return _terminal(target, state, trace, cost, "TYPED_SINGLETON")
        if action == "GENERIC":
            return _terminal(target, state, trace, cost + problem["generic_cost"], "GENERIC_TRUSTED")
        question = solver["questions"][action]
        groups = _partition(problem, state, question)
        outcome = question.outcomes[problem["index"][target]]
        next_state = groups[outcome]
        step_cost = question_cost(problem, state, question, scenario)
        trace.append({
            "question_id": action, "outcome": outcome, "reachable_category_count": len(groups),
            "step_cost": step_cost, "pre_size": len(state), "post_size": len(next_state),
        })
        cost += step_cost
        state = next_state
        remaining -= 1


def solve_open_loop(problem: dict[str, Any], scenario: dict[str, Any], question_ids: tuple[str, ...]) -> dict[str, Any]:
    candidates = [()] + [sequence for length in range(1, problem["horizon"] + 1) for sequence in itertools.permutations(sorted(question_ids), length)]
    best_cost = problem["generic_cost"]
    best_sequence: tuple[str, ...] = ()
    for sequence in candidates[1:]:
        expected = 0.0
        for target in problem["contract_ids"]:
            row = evaluate_sequence(problem, scenario, sequence, target)
            expected += problem["prior"][target] * row["cost"]
        if expected < best_cost - 1e-12 or (abs(expected - best_cost) <= 1e-12 and best_sequence and sequence < best_sequence):
            best_cost, best_sequence = expected, sequence
    return {"value": best_cost, "sequence": best_sequence}


def evaluate_sequence(problem: dict[str, Any], scenario: dict[str, Any], sequence: tuple[str, ...], target: str) -> dict[str, Any]:
    by_id = {question.question_id: question for question in problem["questions"]}
    state = problem["contract_ids"]
    cost = 0.0
    trace = []
    for action in sequence:
        if len(state) == 1:
            break
        question = by_id[action]
        groups = _partition(problem, state, question)
        if len(groups) <= 1:
            continue
        outcome = question.outcomes[problem["index"][target]]
        next_state = groups[outcome]
        step_cost = question_cost(problem, state, question, scenario)
        trace.append({
            "question_id": action, "outcome": outcome, "reachable_category_count": len(groups),
            "step_cost": step_cost, "pre_size": len(state), "post_size": len(next_state),
        })
        cost += step_cost
        state = next_state
    if len(state) == 1:
        return _terminal(target, state, trace, cost, "TYPED_SINGLETON")
    return _terminal(target, state, trace, cost + problem["generic_cost"], "GENERIC_TRUSTED")


def _terminal(target: str, state: tuple[str, ...], trace: list[dict[str, Any]], cost: float, mode: str) -> dict[str, Any]:
    return {
        "target_contract_id": target,
        "terminal_mode": mode,
        "cost": cost,
        "turn_count": len(trace),
        "final_version_space": list(state),
        "target_retained": target in state,
        "final_exact": mode == "GENERIC_TRUSTED" or state == (target,),
        "trace": trace,
    }


def _summarize(problem: dict[str, Any], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_cost": sum(problem["prior"][target] * row["cost"] for target, row in rows.items()),
        "mean_turn_count": sum(problem["prior"][target] * row["turn_count"] for target, row in rows.items()),
        "typed_completion_rate": sum(problem["prior"][target] for target, row in rows.items() if row["terminal_mode"] == "TYPED_SINGLETON"),
        "final_exactness_rate": sum(problem["prior"][target] for target, row in rows.items() if row["final_exact"]),
        "target_retention_rate": sum(problem["prior"][target] for target, row in rows.items() if row["target_retained"]),
    }


def evaluate(problem: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    all_ids = tuple(question.question_id for question in problem["questions"])
    coarse_ids = problem["coarse_question_ids"]
    scenario_rows = []
    paths = {}
    for scenario in scenarios(config):
        exact = solve_exact(problem, scenario, all_ids)
        coarse = solve_exact(problem, scenario, coarse_ids)
        open_loop = solve_open_loop(problem, scenario, all_ids)
        exact_rows = {target: evaluate_exact(problem, exact, scenario, target) for target in problem["contract_ids"]}
        coarse_rows = {target: evaluate_exact(problem, coarse, scenario, target) for target in problem["contract_ids"]}
        open_rows = {target: evaluate_sequence(problem, scenario, open_loop["sequence"], target) for target in problem["contract_ids"]}
        exact_summary = _summarize(problem, exact_rows)
        open_summary = _summarize(problem, open_rows)
        coarse_summary = _summarize(problem, coarse_rows)
        root_action = exact["choice"](problem["contract_ids"], problem["horizon"])[1]
        scenario_rows.append({
            **scenario,
            "exact_all": exact_summary | {"root_action": root_action},
            "best_open_loop_all": open_summary | {"sequence": list(open_loop["sequence"])},
            "exact_coarse_only": coarse_summary | {"root_action": coarse["choice"](problem["contract_ids"], problem["horizon"])[1]},
            "always_generic_cost": problem["generic_cost"],
            "exact_improvement_over_generic": problem["generic_cost"] - exact_summary["mean_cost"],
            "exact_advantage_over_open_loop": open_summary["mean_cost"] - exact_summary["mean_cost"],
        })
        paths[scenario["scenario_id"]] = {"exact_all": exact_rows, "best_open_loop_all": open_rows, "exact_coarse_only": coarse_rows}
    pure = next(row for row in scenario_rows if row["scenario_id"] == "bit_slot_o00")
    conditional = [row for row in scenario_rows if row["scenario_id"] != "bit_slot_o00" and row["exact_improvement_over_generic"] > 1e-12]
    robust = pure["exact_improvement_over_generic"] > 1e-12
    summary = {
        "contract_count": len(problem["contract_ids"]),
        "positive_prior_contract_count": sum(value > 0 for value in problem["prior"].values()),
        "all_question_count": len(all_ids),
        "coarse_question_count": len(coarse_ids),
        "pricing_scenario_count": len(scenario_rows),
        "pure_bit_slot_exact_cost": pure["exact_all"]["mean_cost"],
        "pure_bit_slot_exact_not_below_generic": pure["exact_all"]["mean_cost"] >= problem["generic_cost"] - 1e-12,
        "robust_multiway_value": robust,
        "positive_conditional_scenario_count": len(conditional),
        "conditional_multiway_value": bool(conditional) and not robust,
        "minimum_conditional_exact_cost": min((row["exact_all"]["mean_cost"] for row in conditional), default=None),
        "maximum_mean_turn_count": max(row["exact_all"]["mean_turn_count"] for row in scenario_rows),
        "all_scenario_minimum_exactness_rate": min(row["exact_all"]["final_exactness_rate"] for row in scenario_rows),
        "all_scenario_minimum_retention_rate": min(row["exact_all"]["target_retention_rate"] for row in scenario_rows),
        "adaptive_advantage_scenario_count": sum(row["exact_advantage_over_open_loop"] > 1e-12 for row in scenario_rows),
        "global_intent_menu_drives_positive_conditions": all(row["exact_all"]["root_action"] == "M189_intent_concept" for row in conditional),
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
    return {"summary": summary, "scenarios": scenario_rows, "paths": paths}


def audit(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    gates = config["feasibilityGates"]
    checks = {
        "population_question_sets_and_pricing_census_are_exact": bool(
            summary["contract_count"] == gates["requiredContractCount"]
            and summary["positive_prior_contract_count"] == gates["requiredPositivePriorContractCount"]
            and summary["all_question_count"] == gates["requiredAllQuestionCount"]
            and summary["coarse_question_count"] == gates["requiredCoarseQuestionCount"]
            and summary["pricing_scenario_count"] == gates["requiredPricingScenarioCount"]
        ),
        "all_terminal_paths_are_exact_and_safe": bool(
            summary["all_scenario_minimum_exactness_rate"] == gates["requiredObservedFinalExactnessRate"]
            and summary["all_scenario_minimum_retention_rate"] == gates["requiredTargetRetentionRate"]
            and summary["maximum_mean_turn_count"] <= gates["maximumMeanTurnCount"]
        ),
        "pure_bit_and_conditional_controls_are_present": bool(
            summary["pure_bit_slot_exact_not_below_generic"] == gates["requiredPureBitSlotAllQuestionCostNotBelowGeneric"]
            and summary["positive_conditional_scenario_count"] >= gates["minimumPositiveConditionalScenarioCount"]
        ),
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


__all__ = ["audit", "build_problem", "evaluate", "question_cost", "scenarios", "solve_exact"]
