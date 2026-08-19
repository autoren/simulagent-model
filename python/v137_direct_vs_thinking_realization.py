from __future__ import annotations

import json
from collections import Counter
from typing import Any

from v136_controlled_clarification_value import choice_kind, decision_cost


def render_prompt(catalog: dict[str, Any], fixture: dict[str, Any], config: dict[str, Any]) -> str:
    payload = {
        "instruction": config["prompt"]["instruction"],
        "presented_candidate_under_review": fixture["presented_candidate_choice_id"],
        "choices": catalog["choices"],
        "conversation": fixture["conversation"],
        "response_contract": config["prompt"]["responseContract"],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def validate_final_answer(raw: str, catalog: dict[str, Any], thinking_enabled: bool) -> dict[str, Any]:
    stripped = raw.strip()
    open_count = stripped.count("<think>")
    close_count = stripped.count("</think>")
    trace_present = open_count > 0 or close_count > 0
    trace_closed = open_count == close_count and (open_count > 0 or not trace_present)
    final_text = stripped
    if thinking_enabled:
        if open_count != close_count:
            return {
                "answer_choice_id": "A00",
                "response_valid": False,
                "validation_reason": "unclosed_thinking_trace",
                "thinking_trace_present": trace_present,
                "thinking_trace_closed": False,
                "final_text": "",
            }
        if close_count:
            final_text = stripped.rsplit("</think>", 1)[1].strip()
    elif trace_present:
        return {
            "answer_choice_id": "A00",
            "response_valid": False,
            "validation_reason": "unexpected_thinking_trace",
            "thinking_trace_present": True,
            "thinking_trace_closed": trace_closed,
            "final_text": "",
        }
    try:
        value = json.loads(final_text)
    except json.JSONDecodeError:
        return {
            "answer_choice_id": "A00",
            "response_valid": False,
            "validation_reason": "invalid_final_json",
            "thinking_trace_present": trace_present,
            "thinking_trace_closed": trace_closed,
            "final_text": final_text,
        }
    valid_ids = {row["choice_id"] for row in catalog["choices"]}
    if not isinstance(value, dict) or set(value) != {"choice_id"}:
        return {
            "answer_choice_id": "A00",
            "response_valid": False,
            "validation_reason": "invalid_final_keys",
            "thinking_trace_present": trace_present,
            "thinking_trace_closed": trace_closed,
            "final_text": final_text,
        }
    choice = value.get("choice_id")
    if not isinstance(choice, str) or choice not in valid_ids:
        return {
            "answer_choice_id": "A00",
            "response_valid": False,
            "validation_reason": "unknown_choice_id",
            "thinking_trace_present": trace_present,
            "thinking_trace_closed": trace_closed,
            "final_text": final_text,
        }
    return {
        "answer_choice_id": choice,
        "response_valid": True,
        "validation_reason": "valid",
        "thinking_trace_present": trace_present,
        "thinking_trace_closed": trace_closed,
        "final_text": final_text,
    }


def _phase_class(phase: str) -> str:
    if phase == "ambiguous":
        return "ambiguous"
    if phase.startswith("clarified_"):
        return "clarified"
    return "clear"


def evaluate_condition(
    condition_id: str,
    completed: dict[str, dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    catalog: dict[str, Any],
    v136_config: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_fixture = {row["fixture_id"]: row for row in hidden_rows}
    expected_names = {f"{condition_id}::{fixture_id}" for fixture_id in by_fixture}
    if set(completed) != expected_names:
        raise ValueError(f"V137 {condition_id} fixture completion mismatch")
    rows = []
    for name, output in completed.items():
        fixture_id = output["fixture_id"]
        fixture = by_fixture[fixture_id]
        answer = output["answer_choice_id"]
        truth = fixture["truth_choice_id"]
        rows.append(
            {
                "fixture_id": fixture_id,
                "group_id": fixture["group_id"],
                "phase": fixture["phase"],
                "phase_class": _phase_class(fixture["phase"]),
                "truth": truth,
                "answer": answer,
                "candidate": fixture["presented_candidate_choice_id"],
                "truth_kind": choice_kind(truth, catalog),
                "correct": answer == truth,
                "valid": output["response_valid"],
                "thinking_trace_present": output["thinking_trace_present"],
                "generated_token_count": output["generated_token_count"],
                "generation_seconds": output["generation_seconds"],
            }
        )
    rows.sort(key=lambda row: row["fixture_id"])
    total = len(rows)
    errors = [row for row in rows if not row["correct"]]
    known_ids = {row["choice_id"] for row in catalog["choices"] if row["kind"] == "KNOWN"}
    nonknown = [row for row in rows if row["truth_kind"] != "known"]
    phase_rows = {phase: [row for row in rows if row["phase_class"] == phase] for phase in ("clear", "ambiguous", "clarified")}
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(row)
    full_group_accuracy = sum(all(row["correct"] for row in group) for group in by_group.values()) / len(by_group)

    sequential = []
    for group_id, group in sorted(by_group.items()):
        phase_map = {row["phase"]: row for row in group}
        initial_action = phase_map["ambiguous"]["answer"]
        queried = initial_action == "A00"
        for side in ("left", "right"):
            truth = phase_map[f"clear_{side}"]["truth"]
            final_action = phase_map[f"clarified_{side}"]["answer"] if queried else initial_action
            sequential_cost = decision_cost(truth, final_action, catalog, v136_config) + (
                v136_config["clarificationChannel"]["queryCost"] if queried else 0.0
            )
            no_query_cost = decision_cost(truth, initial_action, catalog, v136_config)
            sequential.append(
                {
                    "group_id": group_id,
                    "side": side,
                    "truth": truth,
                    "truth_kind": choice_kind(truth, catalog),
                    "queried": queried,
                    "initial_action": initial_action,
                    "final_action": final_action,
                    "sequential_cost": sequential_cost,
                    "no_query_cost": no_query_cost,
                    "false_known": side == "right" and final_action in known_ids,
                    "safe_nonknown": side == "right" and final_action not in known_ids,
                }
            )
    right_episodes = [row for row in sequential if row["side"] == "right"]
    metrics = {
        "fixture_count": total,
        "structured_validity": sum(row["valid"] for row in rows) / total,
        "overall_exact_accuracy": sum(row["correct"] for row in rows) / total,
        "clear_exact_accuracy": sum(row["correct"] for row in phase_rows["clear"]) / len(phase_rows["clear"]),
        "ambiguous_abstention_accuracy": sum(row["correct"] for row in phase_rows["ambiguous"]) / len(phase_rows["ambiguous"]),
        "clarified_exact_accuracy": sum(row["correct"] for row in phase_rows["clarified"]) / len(phase_rows["clarified"]),
        "full_five_stage_group_accuracy": full_group_accuracy,
        "false_known_rate_on_nonknown_truths": sum(row["answer"] in known_ids for row in nonknown) / len(nonknown),
        "candidate_attracted_fraction_of_errors": sum(row["answer"] == row["candidate"] for row in errors) / len(errors) if errors else 0.0,
        "thinking_trace_presence": sum(row["thinking_trace_present"] for row in rows) / total,
        "mean_generated_tokens": sum(row["generated_token_count"] for row in rows) / total,
        "total_generation_seconds": sum(row["generation_seconds"] for row in rows),
        "sequential_query_rate": sum(row["queried"] for row in sequential) / len(sequential),
        "sequential_mean_decision_cost": sum(row["sequential_cost"] for row in sequential) / len(sequential),
        "no_query_mean_decision_cost": sum(row["no_query_cost"] for row in sequential) / len(sequential),
        "sequential_improvement_over_no_query": sum(row["no_query_cost"] - row["sequential_cost"] for row in sequential) / len(sequential),
        "sequential_false_known_on_right_truth": sum(row["false_known"] for row in right_episodes) / len(right_episodes),
        "sequential_safe_nonknown_on_right_truth": sum(row["safe_nonknown"] for row in right_episodes) / len(right_episodes),
        "confusion_counts": {
            f"{truth}->{answer}": count
            for (truth, answer), count in sorted(Counter((row["truth"], row["answer"]) for row in rows).items())
        },
    }
    gates = config["qualificationGates"]
    checks = {
        "structured_validity": metrics["structured_validity"] >= gates["minimumStructuredValidity"],
        "overall_exact_accuracy": metrics["overall_exact_accuracy"] >= gates["minimumOverallExactAccuracy"],
        "clear_exact_accuracy": metrics["clear_exact_accuracy"] >= gates["minimumClearExactAccuracy"],
        "ambiguous_abstention_accuracy": metrics["ambiguous_abstention_accuracy"] >= gates["minimumAmbiguousAbstentionAccuracy"],
        "clarified_exact_accuracy": metrics["clarified_exact_accuracy"] >= gates["minimumClarifiedExactAccuracy"],
        "full_group_accuracy": metrics["full_five_stage_group_accuracy"] >= gates["minimumFullFiveStageGroupAccuracy"],
        "false_known": metrics["false_known_rate_on_nonknown_truths"] <= gates["maximumFalseKnownRateOnNonKnownTruths"],
        "candidate_attraction": metrics["candidate_attracted_fraction_of_errors"] <= gates["maximumCandidateAttractedFractionOfErrors"],
        "sequential_cost": metrics["sequential_mean_decision_cost"] <= gates["maximumSequentialMeanDecisionCost"],
        "sequential_improvement": metrics["sequential_improvement_over_no_query"] >= gates["minimumSequentialImprovementOverNoQuery"],
        "sequential_false_known": metrics["sequential_false_known_on_right_truth"] <= gates["maximumSequentialFalseKnownOnRightTruth"],
        "sequential_safe_nonknown": metrics["sequential_safe_nonknown_on_right_truth"] >= gates["minimumSequentialSafeNonKnownOnRightTruth"],
        "thinking_mode_behavior": (
            metrics["thinking_trace_presence"] >= gates["minimumThinkingTracePresenceForThinkingCondition"]
            if condition_id == "thinking"
            else metrics["thinking_trace_presence"] <= gates["maximumThinkingTracePresenceForDirectCondition"]
        ),
        "true_hypothesis_retention": gates["requiredTrueHypothesisRetention"] == 1.0,
        "zero_execution": gates["maximumActualExecutionCount"] == 0,
    }
    return {"condition_id": condition_id, "metrics": metrics, "gates": checks, "qualified": all(checks.values())}


def evaluate_experiment(
    completed: dict[str, dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    catalog: dict[str, Any],
    v136_config: dict[str, Any],
    access: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    conditions = {}
    for condition in config["conditions"]:
        condition_id = condition["id"]
        subset = {name: row for name, row in completed.items() if row["condition_id"] == condition_id}
        conditions[condition_id] = evaluate_condition(condition_id, subset, hidden_rows, catalog, v136_config, config)
    direct = conditions["direct"]
    thinking = conditions["thinking"]
    if thinking["qualified"] and not direct["qualified"]:
        decision = config["decisionRule"]["ifThinkingQualifiesAndDirectFails"]
    elif direct["qualified"] and not thinking["qualified"]:
        decision = config["decisionRule"]["ifDirectQualifiesAndThinkingFails"]
    elif direct["qualified"] and thinking["qualified"]:
        gain = direct["metrics"]["sequential_mean_decision_cost"] - thinking["metrics"]["sequential_mean_decision_cost"]
        decision = (
            config["decisionRule"]["ifBothQualifyWithMaterialThinkingGain"]
            if gain >= config["decisionRule"]["ifBothQualifyAndThinkingReducesSequentialCostByAtLeast"]
            else config["decisionRule"]["ifBothQualifyWithoutMaterialThinkingGain"]
        )
    else:
        decision = config["decisionRule"]["ifNeitherQualifies"]
    access_gates = config["accessGates"]
    access_checks = {
        "zero_V134_language": access["V134_language_read_count"] <= access_gates["maximumV134LanguageReadCount"],
        "zero_external_language": access["external_language_read_count"] <= access_gates["maximumExternalLanguageReadCount"],
        "model_load_budget": access["model_load_count"] <= access_gates["maximumModelLoadCount"],
        "generation_budget": access["model_generation_count"] <= access_gates["maximumModelGenerationCount"],
        "zero_retries": access["retry_count"] <= access_gates["maximumRetryCount"],
        "zero_manual_inspection": access["manual_raw_response_or_trace_inspection_count"] <= access_gates["maximumManualRawResponseOrTraceInspectionCount"],
        "zero_persisted_raw": access["persisted_raw_response_or_trace_count"] <= access_gates["maximumPersistedRawResponseOrTraceCount"],
        "zero_API": access["API_call_count"] <= access_gates["maximumAPICallCount"],
        "zero_training": access["training_run_count"] <= access_gates["maximumTrainingRunCount"],
        "zero_services": access["real_service_call_count"] <= access_gates["maximumRealServiceCallCount"],
        "zero_side_effects": access["external_side_effect_count"] <= access_gates["maximumExternalSideEffectCount"],
        "zero_execution": access["actual_execution_count"] <= access_gates["maximumActualExecutionCount"],
    }
    return {
        "conditions": conditions,
        "at_least_one_condition_qualified": any(row["qualified"] for row in conditions.values()),
        "access_gates": access_checks,
        "access_pass": all(access_checks.values()),
        "decision": decision,
        "true_hypothesis_retention": 1.0,
        "actual_execution_count": 0,
    }


__all__ = ["evaluate_condition", "evaluate_experiment", "render_prompt", "validate_final_answer"]
