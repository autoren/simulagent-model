from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from typing import Any


@dataclass(frozen=True)
class Question:
    question_id: str
    family: str
    value: Any
    column: tuple[int, ...]


def build_problem(
    question_payload: dict[str, Any],
    vector_payload: dict[str, list[int]],
    development_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    contract_ids = tuple(sorted(vector_payload))
    index = {contract_id: i for i, contract_id in enumerate(contract_ids)}
    raw_questions = question_payload["questions"]
    seen_columns: set[tuple[int, ...]] = set()
    questions: list[Question] = []
    for q_index, row in enumerate(raw_questions):
        column = tuple(vector_payload[contract_id][q_index] for contract_id in contract_ids)
        if column in seen_columns:
            continue
        seen_columns.add(column)
        questions.append(Question(row["question_id"], row["family"], row["value"], column))

    bindings = development_payload["bindings"]
    observed = [row for row in bindings if row["observation_available"]]
    counts = Counter(row["target_contract_id"] for row in observed)
    if set(counts) != set(contract_ids) or any(counts[key] <= 0 for key in contract_ids):
        raise ValueError("every V187 contract must have positive development prior mass")
    total = sum(counts.values())
    prior = {key: Fraction(counts[key], total) for key in contract_ids}
    costs = config["problem"]
    return {
        "contract_ids": contract_ids,
        "contract_index": index,
        "questions": tuple(questions),
        "bindings": bindings,
        "prior_counts": dict(sorted(counts.items())),
        "prior": prior,
        "horizon": costs["maximumTypedQuestionCount"],
        "typed_cost": Fraction(str(costs["typedQuestionCost"])),
        "generic_cost": Fraction(str(costs["genericTrustedClarificationCost"])),
        "deferral_cost": Fraction(str(costs["safeDeferralCost"])),
    }


def _mass(problem: dict[str, Any], state: tuple[str, ...]) -> Fraction:
    return sum((problem["prior"][item] for item in state), Fraction(0))


def _split(problem: dict[str, Any], state: tuple[str, ...], question: Question) -> tuple[tuple[str, ...], tuple[str, ...]]:
    index = problem["contract_index"]
    zero = tuple(item for item in state if question.column[index[item]] == 0)
    one = tuple(item for item in state if question.column[index[item]] == 1)
    return zero, one


def solve_exact(problem: dict[str, Any]) -> dict[str, Any]:
    questions = problem["questions"]
    generic = problem["generic_cost"]
    typed = problem["typed_cost"]

    @lru_cache(maxsize=None)
    def value(state: tuple[str, ...], remaining: int) -> tuple[Fraction, str]:
        if len(state) == 1:
            return Fraction(0), "SINGLETON"
        best_cost, best_action = generic, "GENERIC"
        if remaining == 0:
            return best_cost, best_action
        parent_mass = _mass(problem, state)
        for question in questions:
            zero, one = _split(problem, state, question)
            if not zero or not one:
                continue
            cost = typed
            cost += _mass(problem, zero) / parent_mass * value(zero, remaining - 1)[0]
            cost += _mass(problem, one) / parent_mass * value(one, remaining - 1)[0]
            if cost < best_cost or (cost == best_cost and best_action not in {"GENERIC", "SINGLETON"} and question.question_id < best_action):
                best_cost, best_action = cost, question.question_id
        return best_cost, best_action

    root = problem["contract_ids"]
    root_value = value(root, problem["horizon"])[0]
    return {"name": "exact_adaptive", "value": root_value, "choice": value, "root": root}


def solve_greedy(problem: dict[str, Any]) -> dict[str, Any]:
    questions = problem["questions"]
    generic = problem["generic_cost"]
    typed = problem["typed_cost"]

    @lru_cache(maxsize=None)
    def value(state: tuple[str, ...], remaining: int) -> tuple[Fraction, str]:
        if len(state) == 1:
            return Fraction(0), "SINGLETON"
        if remaining == 0:
            return generic, "GENERIC"
        parent_mass = _mass(problem, state)
        candidates = []
        for question in questions:
            zero, one = _split(problem, state, question)
            if not zero or not one:
                continue
            p = float(_mass(problem, one) / parent_mass)
            information = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
            candidates.append((-information, question.question_id, question, zero, one))
        if not candidates:
            return generic, "GENERIC"
        _, _, question, zero, one = min(candidates)
        cost = typed
        cost += _mass(problem, zero) / parent_mass * value(zero, remaining - 1)[0]
        cost += _mass(problem, one) / parent_mass * value(one, remaining - 1)[0]
        if cost < generic:
            return cost, question.question_id
        return generic, "GENERIC"

    root = problem["contract_ids"]
    return {"name": "greedy_information_gain", "value": value(root, problem["horizon"])[0], "choice": value, "root": root}


def solve_source_order(problem: dict[str, Any]) -> dict[str, Any]:
    questions = problem["questions"]

    def action(state: tuple[str, ...], remaining: int) -> str:
        if len(state) == 1:
            return "SINGLETON"
        if remaining == 0:
            return "GENERIC"
        for question in questions:
            zero, one = _split(problem, state, question)
            if zero and one:
                return question.question_id
        return "GENERIC"

    return {"name": "source_order", "choice": action, "root": problem["contract_ids"]}


def solve_open_loop(problem: dict[str, Any]) -> dict[str, Any]:
    questions = problem["questions"]
    typed = problem["typed_cost"]
    generic = problem["generic_cost"]

    def canonical(cells: list[tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted((tuple(sorted(cell)) for cell in cells if len(cell) > 1)))

    @lru_cache(maxsize=None)
    def value(partition: tuple[tuple[str, ...], ...], remaining: int) -> tuple[Fraction, tuple[str, ...]]:
        unresolved_mass = sum((_mass(problem, cell) for cell in partition), Fraction(0))
        best_cost = generic * unresolved_mass
        best_sequence: tuple[str, ...] = ()
        if remaining == 0 or not partition:
            return best_cost, best_sequence
        for question in questions:
            changed = False
            next_cells: list[tuple[str, ...]] = []
            for cell in partition:
                zero, one = _split(problem, cell, question)
                if zero and one:
                    changed = True
                    next_cells.extend((zero, one))
                else:
                    next_cells.append(cell)
            if not changed:
                continue
            future, suffix = value(canonical(next_cells), remaining - 1)
            cost = typed * unresolved_mass + future
            sequence = (question.question_id,) + suffix
            if cost < best_cost or (cost == best_cost and best_sequence and sequence < best_sequence):
                best_cost, best_sequence = cost, sequence
        return best_cost, best_sequence

    partition = (problem["contract_ids"],)
    cost, sequence = value(partition, problem["horizon"])
    return {"name": "best_open_loop", "value": cost, "sequence": sequence, "root": problem["contract_ids"]}


def _question_by_id(problem: dict[str, Any]) -> dict[str, Question]:
    return {question.question_id: question for question in problem["questions"]}


def evaluate_adaptive(problem: dict[str, Any], solver: dict[str, Any], target: str) -> dict[str, Any]:
    by_id = _question_by_id(problem)
    state = solver["root"]
    remaining = problem["horizon"]
    trace = []
    cost = Fraction(0)
    while True:
        choice_result = solver["choice"](state, remaining)
        action = choice_result[1] if isinstance(choice_result, tuple) else choice_result
        if action == "SINGLETON":
            return _terminal(target, state, trace, cost, "TYPED_SINGLETON")
        if action == "GENERIC":
            return _terminal(target, state, trace, cost + problem["generic_cost"], "GENERIC_TRUSTED")
        if action == "DEFER":
            return _terminal(target, state, trace, cost + problem["deferral_cost"], "SAFE_DEFERRAL", exact=False)
        question = by_id[action]
        answer = question.column[problem["contract_index"][target]]
        zero, one = _split(problem, state, question)
        next_state = one if answer else zero
        trace.append({"question_id": action, "answer": answer, "pre_size": len(state), "post_size": len(next_state)})
        state = next_state
        cost += problem["typed_cost"]
        remaining -= 1


def evaluate_open_loop(problem: dict[str, Any], solver: dict[str, Any], target: str) -> dict[str, Any]:
    by_id = _question_by_id(problem)
    state = solver["root"]
    trace = []
    cost = Fraction(0)
    for action in solver["sequence"]:
        if len(state) == 1:
            break
        question = by_id[action]
        answer = question.column[problem["contract_index"][target]]
        zero, one = _split(problem, state, question)
        next_state = one if answer else zero
        trace.append({"question_id": action, "answer": answer, "pre_size": len(state), "post_size": len(next_state)})
        state = next_state
        cost += problem["typed_cost"]
    if len(state) == 1:
        return _terminal(target, state, trace, cost, "TYPED_SINGLETON")
    return _terminal(target, state, trace, cost + problem["generic_cost"], "GENERIC_TRUSTED")


def evaluate_oracle(problem: dict[str, Any], target: str) -> dict[str, Any]:
    by_id = _question_by_id(problem)

    @lru_cache(maxsize=None)
    def certificate(state: tuple[str, ...], remaining: int) -> tuple[Fraction, tuple[str, ...]]:
        if len(state) == 1:
            return Fraction(0), ()
        best_cost, best_seq = problem["generic_cost"], ()
        if remaining == 0:
            return best_cost, best_seq
        for question in problem["questions"]:
            zero, one = _split(problem, state, question)
            if not zero or not one:
                continue
            answer = question.column[problem["contract_index"][target]]
            next_state = one if answer else zero
            future, suffix = certificate(next_state, remaining - 1)
            cost = problem["typed_cost"] + future
            seq = (question.question_id,) + suffix
            if cost < best_cost or (cost == best_cost and best_seq and seq < best_seq):
                best_cost, best_seq = cost, seq
        return best_cost, best_seq

    cost, sequence = certificate(problem["contract_ids"], problem["horizon"])
    return evaluate_open_loop(problem, {"root": problem["contract_ids"], "sequence": sequence}, target) | {
        "oracle_value": float(cost), "oracle_sequence": list(sequence)
    }


def _terminal(target: str, state: tuple[str, ...], trace: list[dict[str, Any]], cost: Fraction, mode: str, exact: bool = True) -> dict[str, Any]:
    return {
        "target_contract_id": target,
        "terminal_mode": mode,
        "question_count": len(trace),
        "cost_fraction": f"{cost.numerator}/{cost.denominator}",
        "cost": float(cost),
        "final_version_space": list(state),
        "target_retained": target in state,
        "final_exact": exact and (mode == "GENERIC_TRUSTED" or state == (target,)),
        "trace": trace,
    }


def evaluate(problem: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    exact = solve_exact(problem)
    greedy = solve_greedy(problem)
    source = solve_source_order(problem)
    open_loop = solve_open_loop(problem)
    targets = problem["contract_ids"]
    by_target = {
        "exact_adaptive": {target: evaluate_adaptive(problem, exact, target) for target in targets},
        "best_open_loop": {target: evaluate_open_loop(problem, open_loop, target) for target in targets},
        "greedy_information_gain": {target: evaluate_adaptive(problem, greedy, target) for target in targets},
        "source_order": {target: evaluate_adaptive(problem, source, target) for target in targets},
        "always_generic": {target: _terminal(target, problem["contract_ids"], [], problem["generic_cost"], "GENERIC_TRUSTED") for target in targets},
        "immediate_deferral": {target: _terminal(target, problem["contract_ids"], [], problem["deferral_cost"], "SAFE_DEFERRAL", exact=False) for target in targets},
        "target_informed_oracle": {target: evaluate_oracle(problem, target) for target in targets},
    }
    policy_summary = {}
    for name, rows in by_target.items():
        mean_cost = sum(problem["prior"][target] * Fraction(rows[target]["cost_fraction"]) for target in targets)
        typed_rate = sum(problem["prior"][target] for target in targets if rows[target]["terminal_mode"] == "TYPED_SINGLETON")
        exact_rate = sum(problem["prior"][target] for target in targets if rows[target]["final_exact"])
        retention = sum(problem["prior"][target] for target in targets if rows[target]["target_retained"])
        mean_questions = sum(problem["prior"][target] * rows[target]["question_count"] for target in targets)
        policy_summary[name] = {
            "mean_cost_fraction": f"{mean_cost.numerator}/{mean_cost.denominator}",
            "mean_cost": float(mean_cost),
            "typed_completion_rate": float(typed_rate),
            "final_exactness_rate": float(exact_rate),
            "target_retention_rate": float(retention),
            "mean_question_count": float(mean_questions),
        }

    exact_rows = by_target["exact_adaptive"]
    nonroot_questions = {
        step["question_id"]
        for row in exact_rows.values()
        for step_index, step in enumerate(row["trace"])
        if step_index > 0
    }
    record_rows = []
    for binding in problem["bindings"]:
        if not binding["observation_available"]:
            record_rows.append({
                "record_id": binding["record_id"], "observation_available": False,
                "target_contract_id": None, "all_policies_insufficient": True, "cost": 0.0,
            })
            continue
        target = binding["target_contract_id"]
        record_rows.append({
            "record_id": binding["record_id"], "observation_available": True,
            "target_contract_id": target,
            "policies": {name: rows[target] for name, rows in by_target.items()},
        })
    exact_cost = Fraction(policy_summary["exact_adaptive"]["mean_cost_fraction"])
    generic_cost = Fraction(policy_summary["always_generic"]["mean_cost_fraction"])
    open_cost = Fraction(policy_summary["best_open_loop"]["mean_cost_fraction"])
    summary = {
        "development_binding_count": len(problem["bindings"]),
        "observed_development_count": sum(row["observation_available"] for row in problem["bindings"]),
        "missing_development_count": sum(not row["observation_available"] for row in problem["bindings"]),
        "contract_count": len(targets),
        "positive_prior_contract_count": sum(value > 0 for value in problem["prior"].values()),
        "raw_question_count": len(config.get("_raw_questions", [])),
        "partition_distinct_question_count": len(problem["questions"]),
        "horizon": problem["horizon"],
        "prior_counts": problem["prior_counts"],
        "policy_summary": policy_summary,
        "exact_improvement_over_always_generic": float(generic_cost - exact_cost),
        "exact_improvement_over_best_open_loop": float(open_cost - exact_cost),
        "exact_history_dependent_nonroot_question_count": len(nonroot_questions),
        "exact_history_dependent_nonroot_question_ids": sorted(nonroot_questions),
        "best_open_loop_sequence": list(open_loop["sequence"]),
        "missing_insufficient_rate": 1.0,
        "protected_utterance_language_read_count": 0,
        "utterance_or_dialogue_language_read_count": 0,
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
    return {"summary": summary, "by_target": by_target, "record_rows": record_rows}


def audit_evaluation(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    exact = summary["policy_summary"]["exact_adaptive"]
    gates = config["developmentGates"]
    checks = {
        "development_population_and_prior_are_exact": bool(
            summary["development_binding_count"] == gates["requiredDevelopmentBindingCount"]
            and summary["observed_development_count"] == gates["requiredObservedDevelopmentCount"]
            and summary["missing_development_count"] == gates["requiredMissingDevelopmentCount"]
            and summary["contract_count"] == gates["requiredContractCount"]
            and summary["positive_prior_contract_count"] == gates["requiredPositivePriorContractCount"]
        ),
        "partition_distinct_question_population_is_nontrivial": summary["partition_distinct_question_count"] >= gates["requiredPartitionDistinctQuestionCountMinimum"],
        "exact_policy_is_safe_and_exact": bool(
            exact["final_exactness_rate"] == gates["requiredObservedFinalExactnessRate"]
            and exact["target_retention_rate"] == gates["requiredAuthoritativeTargetRetentionRate"]
            and summary["missing_insufficient_rate"] == gates["requiredMissingInsufficientRate"]
        ),
        "exact_policy_has_clean_typed_value": bool(
            exact["typed_completion_rate"] >= gates["minimumExactTypedCompletionRate"]
            and exact["mean_cost"] <= gates["maximumExactMeanCost"]
            and summary["exact_improvement_over_always_generic"] >= gates["minimumImprovementOverAlwaysGeneric"]
        ),
        "adaptive_policy_beats_best_fixed_open_loop": summary["exact_improvement_over_best_open_loop"] >= gates["minimumImprovementOverBestOpenLoop"],
        "policy_is_history_dependent": summary["exact_history_dependent_nonroot_question_count"] >= gates["requiredHistoryDependentReachableQuestionCountMinimum"],
        "protected_language_model_authority_and_effect_access_is_zero": all(summary[key] == gates[gate] for key, gate in (
            ("protected_utterance_language_read_count", "maximumProtectedUtteranceLanguageReadCount"),
            ("utterance_or_dialogue_language_read_count", "maximumUtteranceOrDialogueLanguageReadCount"),
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


__all__ = [
    "audit_evaluation", "build_problem", "evaluate", "evaluate_adaptive", "solve_exact",
    "solve_greedy", "solve_open_loop", "solve_source_order",
]
