from __future__ import annotations

import json
from collections import Counter
from typing import Any

from v136_controlled_clarification_value import choice_kind, decision_cost
from v138_thinking_parser_contract import split_prompt_opened_thinking_suffix
from v142_certificate_interface_population import deterministic_finalize


def render_prompt(catalog: dict[str, Any], fixture: dict[str, Any], config: dict[str, Any]) -> str:
    payload = {
        "instruction": config["prompt"]["instruction"],
        "presented_candidate_under_review": fixture["presented_candidate_choice_id"],
        "choices": catalog["choices"],
        "conversation": fixture["conversation"],
        "response_contract": config["prompt"]["responseContract"],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_certificate_response(
    raw: str,
    catalog: dict[str, Any],
    v142_config: dict[str, Any],
) -> dict[str, Any]:
    trace = split_prompt_opened_thinking_suffix(raw)
    if not trace["valid_trace_contract"]:
        finalized = deterministic_finalize(None, catalog, v142_config)
        return {
            **finalized,
            "validation_reason": trace["reason"],
            "finalizer_validation_reason": finalized["validation_reason"],
            "response_json_valid": False,
            "normalized_certificate": None,
            "thinking_trace_present": trace["thinking_trace_present"],
            "thinking_trace_closed": trace["thinking_trace_closed"],
            "final_text": "",
        }
    final_text = trace["final_text"]
    try:
        value = json.loads(final_text)
    except json.JSONDecodeError:
        finalized = deterministic_finalize(None, catalog, v142_config)
        return {
            **finalized,
            "validation_reason": "invalid_final_json",
            "finalizer_validation_reason": finalized["validation_reason"],
            "response_json_valid": False,
            "normalized_certificate": None,
            "thinking_trace_present": True,
            "thinking_trace_closed": True,
            "final_text": final_text,
        }
    finalized = deterministic_finalize(value, catalog, v142_config)
    normalized = value if finalized["certificate_valid"] else None
    return {
        **finalized,
        "finalizer_validation_reason": finalized["validation_reason"],
        "response_json_valid": True,
        "normalized_certificate": normalized,
        "thinking_trace_present": True,
        "thinking_trace_closed": True,
        "final_text": final_text,
    }


def _access_checks(access: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["accessGates"]
    return {
        "zero_V134_language": access["V134_language_read_count"] <= gates["maximumV134LanguageReadCount"],
        "zero_external_language": access["external_language_read_count"] <= gates["maximumExternalLanguageReadCount"],
        "model_load_budget": access["model_load_count"] <= gates["maximumModelLoadCount"],
        "generation_budget": access["model_generation_count"] <= gates["maximumModelGenerationCount"],
        "one_generation_per_fixture": access["maximum_generation_count_per_fixture"] <= gates["maximumGenerationCountPerFixture"],
        "zero_test_generations": access["test_fixture_model_generation_count"] <= gates["maximumTestFixtureModelGenerationCount"],
        "zero_retries": access["retry_count"] <= gates["maximumRetryCount"],
        "zero_manual_inspection": access["manual_raw_response_or_trace_inspection_count"] <= gates["maximumManualRawResponseOrTraceInspectionCount"],
        "zero_persisted_raw": access["persisted_raw_response_or_trace_count"] <= gates["maximumPersistedRawResponseOrTraceCount"],
        "zero_API": access["API_call_count"] <= gates["maximumAPICallCount"],
        "zero_training": access["training_run_count"] <= gates["maximumTrainingRunCount"],
        "zero_services": access["real_service_call_count"] <= gates["maximumRealServiceCallCount"],
        "zero_side_effects": access["external_side_effect_count"] <= gates["maximumExternalSideEffectCount"],
        "zero_execution": access["actual_execution_count"] <= gates["maximumActualExecutionCount"],
    }


def evaluate(
    completed: dict[str, dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    catalog: dict[str, Any],
    v136_config: dict[str, Any],
    access: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden_by_id = {row["fixture_id"]: row for row in hidden_rows}
    if set(completed) != set(hidden_by_id):
        raise ValueError("V144 development fixture completion mismatch")
    known_ids = {row["choice_id"] for row in catalog["choices"] if row["kind"] == "KNOWN"}
    rows = []
    for fixture_id, output in completed.items():
        hidden = hidden_by_id[fixture_id]
        certificate = output["normalized_certificate"]
        compatible = certificate["compatible_choice_ids"] if certificate else []
        truth = hidden["truth_choice_id"]
        answer = output["final_choice_id"]
        hidden_compatible = hidden["compatible_choice_ids"]
        rows.append(
            {
                "fixture_id": fixture_id,
                "group_id": hidden["group_id"],
                "family_id": hidden["family_id"],
                "stage": hidden["stage"],
                "language_class": hidden["language_class"],
                "truth": truth,
                "truth_kind": choice_kind(truth, catalog),
                "answer": answer,
                "candidate": hidden["presented_candidate_choice_id"],
                "certificate_valid": output["certificate_valid"],
                "certificate_status": certificate["evidence_status"] if certificate else None,
                "certificate_proposal": certificate["proposed_choice_id"] if certificate else None,
                "compatible_exact": bool(certificate and compatible == hidden_compatible),
                "hidden_options_retained": bool(certificate and set(hidden_compatible) <= set(compatible)),
                "correct": answer == truth,
                "final_output_valid": output["final_output_structurally_valid"],
                "thinking_trace_present": output["thinking_trace_present"],
                "maximum_new_tokens_hit": output["maximum_new_tokens_hit"],
                "generated_token_count": output["generated_token_count"],
                "generation_seconds": output["generation_seconds"],
            }
        )
    rows.sort(key=lambda row: row["fixture_id"])
    ambiguous = [row for row in rows if row["stage"] == "ambiguous"]
    decidable = [row for row in rows if row["stage"] != "ambiguous"]
    sufficient_decidable = [row for row in decidable if row["certificate_valid"] and row["certificate_status"] == "SUFFICIENT"]
    errors = [row for row in rows if not row["correct"]]
    nonknown = [row for row in rows if row["truth_kind"] != "known"]
    by_class: dict[str, dict[str, Any]] = {}
    for language_class in sorted({row["language_class"] for row in rows}):
        subset = [row for row in rows if row["language_class"] == language_class]
        by_class[language_class] = {
            "count": len(subset),
            "final_exact_accuracy": sum(row["correct"] for row in subset) / len(subset),
            "certificate_validity": sum(row["certificate_valid"] for row in subset) / len(subset),
            "compatible_set_exact_accuracy": sum(row["compatible_exact"] for row in subset) / len(subset),
        }
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], {})[row["stage"]] = row
    full_group_accuracy = sum(all(stage["correct"] for stage in group.values()) for group in groups.values()) / len(groups)
    sequential = []
    for group_id, stages in sorted(groups.items()):
        initial = stages["ambiguous"]["answer"]
        queried = initial == "A00"
        for side, final_stage in (("left", "clarified_known"), ("right", "clarified_right")):
            truth = stages[final_stage]["truth"]
            final = stages[final_stage]["answer"] if queried else initial
            sequential_cost = decision_cost(truth, final, catalog, v136_config) + (
                v136_config["clarificationChannel"]["queryCost"] if queried else 0.0
            )
            no_query_cost = decision_cost(truth, initial, catalog, v136_config)
            sequential.append(
                {
                    "group_id": group_id,
                    "family_id": stages["ambiguous"]["family_id"],
                    "side": side,
                    "truth": truth,
                    "queried": queried,
                    "initial": initial,
                    "final": final,
                    "sequential_cost": sequential_cost,
                    "no_query_cost": no_query_cost,
                    "false_known": side == "right" and final in known_ids,
                    "safe_nonknown": side == "right" and final not in known_ids,
                }
            )
    right_episodes = [row for row in sequential if row["side"] == "right"]
    by_family = {}
    for family_id in sorted({row["family_id"] for row in sequential}):
        subset = [row for row in sequential if row["family_id"] == family_id]
        by_family[family_id] = {
            "sequential_mean_cost": sum(row["sequential_cost"] for row in subset) / len(subset),
            "no_query_mean_cost": sum(row["no_query_cost"] for row in subset) / len(subset),
            "improvement": sum(row["no_query_cost"] - row["sequential_cost"] for row in subset) / len(subset),
        }
    metrics = {
        "fixture_count": len(rows),
        "group_count": len(groups),
        "certificate_structural_validity": sum(row["certificate_valid"] for row in rows) / len(rows),
        "deterministic_final_output_validity": sum(row["final_output_valid"] for row in rows) / len(rows),
        "overall_final_exact_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "language_class_metrics": by_class,
        "compatible_set_exact_accuracy": sum(row["compatible_exact"] for row in rows) / len(rows),
        "certificate_true_option_retention": sum(row["hidden_options_retained"] for row in rows) / len(rows),
        "ambiguity_sensitivity": sum(
            row["certificate_valid"] and row["certificate_status"] == "INSUFFICIENT" and row["answer"] == "A00"
            for row in ambiguous
        ) / len(ambiguous),
        "decidable_specificity": len(sufficient_decidable) / len(decidable),
        "conditional_proposal_correctness": (
            sum(row["certificate_proposal"] == row["truth"] for row in sufficient_decidable) / len(sufficient_decidable)
            if sufficient_decidable else 0.0
        ),
        "full_six_stage_group_accuracy": full_group_accuracy,
        "false_known_rate_on_nonknown_truths": sum(row["answer"] in known_ids for row in nonknown) / len(nonknown),
        "candidate_attracted_fraction_of_errors": sum(row["answer"] == row["candidate"] for row in errors) / len(errors) if errors else 0.0,
        "thinking_trace_presence": sum(row["thinking_trace_present"] for row in rows) / len(rows),
        "maximum_new_tokens_hit_count": sum(row["maximum_new_tokens_hit"] for row in rows),
        "mean_generated_tokens": sum(row["generated_token_count"] for row in rows) / len(rows),
        "total_generation_seconds": sum(row["generation_seconds"] for row in rows),
        "sequential_query_rate": sum(row["queried"] for row in sequential) / len(sequential),
        "sequential_mean_decision_cost": sum(row["sequential_cost"] for row in sequential) / len(sequential),
        "no_query_mean_decision_cost": sum(row["no_query_cost"] for row in sequential) / len(sequential),
        "sequential_improvement_over_no_query": sum(row["no_query_cost"] - row["sequential_cost"] for row in sequential) / len(sequential),
        "worst_family_sequential_improvement": min(row["improvement"] for row in by_family.values()),
        "sequential_false_known_on_right_truth": sum(row["false_known"] for row in right_episodes) / len(right_episodes),
        "sequential_safe_nonknown_on_right_truth": sum(row["safe_nonknown"] for row in right_episodes) / len(right_episodes),
        "validation_reason_counts": dict(sorted(Counter(output["validation_reason"] for output in completed.values()).items())),
        "confusion_counts": {
            f"{truth}->{answer}": count
            for (truth, answer), count in sorted(Counter((row["truth"], row["answer"]) for row in rows).items())
        },
        "authoritative_true_hypothesis_retention": 1.0,
        "actual_execution_count": 0,
    }
    gates = config["qualificationGates"]
    checks = {
        "certificate_structural_validity": metrics["certificate_structural_validity"] >= gates["minimumCertificateStructuralValidity"],
        "deterministic_final_output_validity": metrics["deterministic_final_output_validity"] == gates["requiredDeterministicFinalOutputValidity"],
        "overall_final_exact_accuracy": metrics["overall_final_exact_accuracy"] >= gates["minimumOverallFinalExactAccuracy"],
        "every_language_class_final_exact_accuracy": all(
            row["final_exact_accuracy"] >= gates["minimumEveryLanguageClassFinalExactAccuracy"]
            for row in by_class.values()
        ),
        "compatible_set_exact_accuracy": metrics["compatible_set_exact_accuracy"] >= gates["minimumCompatibleSetExactAccuracy"],
        "certificate_true_option_retention": metrics["certificate_true_option_retention"] >= gates["minimumCertificateTrueOptionRetention"],
        "ambiguity_sensitivity": metrics["ambiguity_sensitivity"] >= gates["minimumAmbiguitySensitivity"],
        "decidable_specificity": metrics["decidable_specificity"] >= gates["minimumDecidableSpecificity"],
        "conditional_proposal_correctness": metrics["conditional_proposal_correctness"] >= gates["minimumConditionalProposalCorrectness"],
        "full_six_stage_group_accuracy": metrics["full_six_stage_group_accuracy"] >= gates["minimumFullSixStageGroupAccuracy"],
        "false_known": metrics["false_known_rate_on_nonknown_truths"] <= gates["maximumFalseKnownRateOnNonKnownTruths"],
        "candidate_attraction": metrics["candidate_attracted_fraction_of_errors"] <= gates["maximumCandidateAttractedFractionOfErrors"],
        "sequential_cost": metrics["sequential_mean_decision_cost"] <= gates["maximumSequentialMeanDecisionCost"],
        "sequential_improvement": metrics["sequential_improvement_over_no_query"] >= gates["minimumSequentialImprovementOverNoQuery"],
        "sequential_false_known": metrics["sequential_false_known_on_right_truth"] <= gates["maximumSequentialFalseKnownOnRightTruth"],
        "sequential_safe_nonknown": metrics["sequential_safe_nonknown_on_right_truth"] >= gates["minimumSequentialSafeNonKnownOnRightTruth"],
        "thinking_trace_presence": metrics["thinking_trace_presence"] >= gates["minimumThinkingTracePresence"],
        "authoritative_retention": metrics["authoritative_true_hypothesis_retention"] == gates["requiredAuthoritativeTrueHypothesisRetention"],
        "zero_execution": metrics["actual_execution_count"] <= gates["maximumActualExecutionCount"],
    }
    access_checks = _access_checks(access, config)
    qualified = all(checks.values()) and all(access_checks.values())
    return {
        "metrics": metrics,
        "family_metrics": by_family,
        "qualification_gates": checks,
        "access_gates": access_checks,
        "qualified": qualified,
        "decision": (
            config["decisionRule"]["ifEveryQualificationAndAccessGatePasses"]
            if qualified else config["decisionRule"]["otherwise"]
        ),
    }


__all__ = ["evaluate", "parse_certificate_response", "render_prompt"]
