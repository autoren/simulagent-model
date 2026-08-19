from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from v148_typed_witness_firewall import finalize_witness
from v149_fresh_closed_interaction_population import witness_from_answer_event
from v150_oracle_closed_interaction_policy import decision_cost


class _DuplicateKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(key)
        value[key] = item
    return value


def render_prompt(catalog: dict[str, Any], fixture: dict[str, Any], config: dict[str, Any]) -> str:
    public_queries = [
        {
            "query_id": row["query_id"],
            "question": row["question"],
            "options": [{"option_id": option["option_id"], "text": option["text"]} for option in row["options"]],
        }
        for row in catalog["queries"]
    ]
    payload = {
        "instruction": config["prompt"]["instruction"],
        "presented_candidate_under_review": fixture["presented_candidate_choice_id"],
        "states": catalog["choices"],
        "registered_clarification_questions": public_queries,
        "conversation": fixture["conversation"],
        "response_contract": {
            "evidence_status": config["prompt"]["allowedEvidenceStatus"],
            "candidate_state_ids": "ordered unique list of 1 to 3 supplied non-A00 state IDs",
            "query_ranking": "every supplied query ID exactly once, most useful first",
            "confidence": "number from 0 to 1",
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _invalid(reason: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_valid": False,
        "validation_reason": reason,
        "normalized_proposal": None,
        "evidence_status": "NEEDS_CLARIFICATION",
        "candidate_state_ids": [],
        "query_ranking": list(config["fallbackQueryRanking"]),
        "confidence": 0.0,
        "permanently_non_authoritative": True,
        "authoritative_hypothesis_universe_pruned": False,
        "capability_defined_or_registered": False,
        "executable": False,
        "actual_execution_count": 0,
    }


def parse_proposal(raw: str, catalog: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw.strip(), object_pairs_hook=_strict_object)
    except (_DuplicateKeyError, json.JSONDecodeError, AttributeError):
        return _invalid("invalid_json", config)
    required = set(config["prompt"]["responseKeys"])
    if not isinstance(value, dict) or set(value) != required:
        return _invalid("invalid_object_shape", config)
    status = value.get("evidence_status")
    if status not in config["prompt"]["allowedEvidenceStatus"]:
        return _invalid("invalid_evidence_status", config)
    valid_state_ids = {row["choice_id"] for row in catalog["choices"] if row["choice_id"] != "A00"}
    candidates = value.get("candidate_state_ids")
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= config["prompt"]["maximumCandidateCount"]
        or any(not isinstance(choice, str) for choice in candidates)
        or len(candidates) != len(set(candidates))
        or any(choice not in valid_state_ids for choice in candidates)
    ):
        return _invalid("invalid_candidate_state_ids", config)
    query_ids = [row["query_id"] for row in catalog["queries"]]
    ranking = value.get("query_ranking")
    if (
        not isinstance(ranking, list)
        or len(ranking) != len(query_ids)
        or any(not isinstance(query, str) for query in ranking)
        or len(ranking) != len(set(ranking))
        or set(ranking) != set(query_ids)
    ):
        return _invalid("invalid_query_ranking", config)
    confidence = value.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        return _invalid("invalid_confidence", config)
    normalized = {
        "evidence_status": status,
        "candidate_state_ids": candidates,
        "query_ranking": ranking,
        "confidence": float(confidence),
    }
    return {
        "proposal_valid": True,
        "validation_reason": "valid_registered_proposal",
        "normalized_proposal": normalized,
        **normalized,
        "permanently_non_authoritative": True,
        "authoritative_hypothesis_universe_pruned": False,
        "capability_defined_or_registered": False,
        "executable": False,
        "actual_execution_count": 0,
    }


def _calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins = []
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        subset = [row for row in rows if lower <= row["confidence"] < upper or (upper == 1.0 and row["confidence"] == 1.0)]
        if not subset:
            continue
        accuracy = sum(row["semantic_exact"] for row in subset) / len(subset)
        confidence = sum(row["confidence"] for row in subset) / len(subset)
        ece += len(subset) / len(rows) * abs(accuracy - confidence)
        bins.append({"lower": lower, "upper": upper, "count": len(subset), "accuracy": accuracy, "mean_confidence": confidence})
    risk_coverage = {}
    for threshold in (0.5, 0.7, 0.9):
        covered = [row for row in rows if row["confidence"] >= threshold]
        risk_coverage[f"{threshold:.1f}"] = {
            "coverage": len(covered) / len(rows),
            "selective_accuracy": sum(row["semantic_exact"] for row in covered) / len(covered) if covered else None,
        }
    return {
        "self_reported_confidence_ECE_10_bin": ece,
        "self_reported_confidence_brier": sum((row["confidence"] - float(row["semantic_exact"])) ** 2 for row in rows) / len(rows),
        "calibration_bins": bins,
        "risk_coverage": risk_coverage,
        "confidence_is_diagnostic_not_fitted_or_authoritative": True,
    }


def evaluate(
    completed: dict[str, dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    answer_metadata: list[dict[str, Any]],
    catalog: dict[str, Any],
    witness_config: dict[str, Any],
    oracle_config: dict[str, Any],
    access: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden_by_id = {row["fixture_id"]: row for row in hidden_rows}
    if set(completed) != set(hidden_by_id):
        raise ValueError("V151 development request completion mismatch")
    rows = []
    for fixture_id, output in completed.items():
        hidden = hidden_by_id[fixture_id]
        expected_status = "NEEDS_CLARIFICATION" if hidden["stage"] == "request_ambiguous" else "DECIDABLE"
        compatible = set(hidden["compatible_state_ids"])
        proposed = output["candidate_state_ids"]
        exact_set = set(proposed) == compatible
        retained = compatible <= set(proposed)
        query_rank = output["query_ranking"].index(hidden["oracle_query_id"]) + 1
        semantic_exact = output["evidence_status"] == expected_status and exact_set
        rows.append(
            {
                "fixture_id": fixture_id,
                "group_id": hidden["group_id"],
                "family_id": hidden["family_id"],
                "stage": hidden["stage"],
                "language_class": hidden["language_class"],
                "truth": hidden["truth_state_id"],
                "compatible": compatible,
                "presented_candidate": hidden["presented_candidate_choice_id"],
                "proposal_valid": output["proposal_valid"],
                "status": output["evidence_status"],
                "expected_status": expected_status,
                "candidates": proposed,
                "query_ranking": list(output["query_ranking"]),
                "compatible_retained": retained,
                "compatible_exact": exact_set,
                "top1_exact": bool(proposed and hidden["truth_state_id"] != "A00" and proposed[0] == hidden["truth_state_id"]),
                "query_rank": query_rank,
                "query_top1": query_rank == 1,
                "semantic_exact": semantic_exact,
                "confidence": output["confidence"],
                "generation_seconds": output["generation_seconds"],
                "generated_token_count": output["generated_token_count"],
            }
        )
    rows.sort(key=lambda row: row["fixture_id"])
    known_ids = set(witness_config["knownIds"])
    ambiguous = [row for row in rows if row["stage"] == "request_ambiguous"]
    decidable = [row for row in rows if row["stage"] != "request_ambiguous"]
    nonknown = [row for row in rows if row["truth"] not in known_ids]
    semantic_errors = [row for row in rows if not row["semantic_exact"]]
    class_metrics = {}
    for language_class in sorted({row["language_class"] for row in rows}):
        subset = [row for row in rows if row["language_class"] == language_class]
        class_metrics[language_class] = {
            "count": len(subset),
            "compatible_state_retention": sum(row["compatible_retained"] for row in subset) / len(subset),
            "compatible_set_exact_accuracy": sum(row["compatible_exact"] for row in subset) / len(subset),
            "query_top1_accuracy": sum(row["query_top1"] for row in subset) / len(subset),
        }

    answer_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for row in answer_metadata:
        answer_by_group.setdefault(row["group_id"], {})[row["stage"]] = row
    sequential = []
    intermediate_outputs = []
    for row in rows:
        sides = ("left", "right") if row["stage"] == "request_ambiguous" else (("right",) if row["stage"] == "request_right" else ("left",))
        for side in sides:
            answer_stage = "closed_answer_known" if side == "left" else "closed_answer_right"
            answer = answer_by_group[row["group_id"]][answer_stage]
            correct_query = answer["oracle_query_id"]
            query_count = row["query_rank"]
            candidate_hint = row["candidates"][0] if row["candidates"] else "A00"
            for asked_query in row["query_ranking"][: query_count - 1]:
                intermediate = finalize_witness(None, candidate_hint, witness_config)
                intermediate_outputs.append(
                    {
                        "fixture_id": row["fixture_id"],
                        "asked_query": asked_query,
                        "oracle_query": correct_query,
                        "final": intermediate["final_state_id"],
                        "witness_valid": intermediate["witness_valid"],
                        "retained": intermediate["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
                        "actual_execution_count": intermediate["actual_execution_count"],
                    }
                )
            witness = witness_from_answer_event(answer["closed_answer_event"], catalog)
            finalized = finalize_witness(witness, candidate_hint, witness_config)
            final = finalized["final_state_id"]
            cost = query_count * oracle_config["policy"]["queryCost"] + decision_cost(
                answer["truth_state_id"], final, known_ids, oracle_config
            )
            sequential.append(
                {
                    "fixture_id": row["fixture_id"],
                    "family_id": row["family_id"],
                    "side": side,
                    "truth": answer["truth_state_id"],
                    "correct_query": correct_query,
                    "query_count": query_count,
                    "final": final,
                    "cost": cost,
                    "false_known": answer["truth_state_id"] not in known_ids and final in known_ids,
                    "safe_nonknown": answer["truth_state_id"] not in known_ids and final not in known_ids,
                    "retained": finalized["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
                }
            )
    nonknown_seq = [row for row in sequential if row["truth"] not in known_ids]
    metrics = {
        "fixture_count": len(rows),
        "group_count": len({row["group_id"] for row in rows}),
        "sequential_episode_count": len(sequential),
        "proposal_structural_validity": sum(row["proposal_valid"] for row in rows) / len(rows),
        "compatible_state_retention": sum(row["compatible_retained"] for row in rows) / len(rows),
        "language_class_metrics": class_metrics,
        "compatible_set_exact_accuracy": sum(row["compatible_exact"] for row in rows) / len(rows),
        "ambiguity_sensitivity": sum(row["status"] == "NEEDS_CLARIFICATION" and row["compatible_retained"] for row in ambiguous) / len(ambiguous),
        "decidable_specificity": sum(row["status"] == "DECIDABLE" for row in decidable) / len(decidable),
        "decidable_top1_accuracy": sum(row["top1_exact"] for row in decidable) / len(decidable),
        "query_top1_accuracy": sum(row["query_top1"] for row in rows) / len(rows),
        "query_mean_reciprocal_rank": sum(1.0 / row["query_rank"] for row in rows) / len(rows),
        "mean_correct_query_rank": sum(row["query_rank"] for row in rows) / len(rows),
        "false_known_singleton_rate_on_nonknown_truth": sum(
            row["status"] == "DECIDABLE" and len(row["candidates"]) == 1 and row["candidates"][0] in known_ids
            for row in nonknown
        ) / len(nonknown),
        "candidate_attracted_fraction_of_semantic_errors": sum(bool(row["candidates"]) and row["candidates"][0] == row["presented_candidate"] for row in semantic_errors) / len(semantic_errors) if semantic_errors else 0.0,
        "sequential_mean_decision_cost": sum(row["cost"] for row in sequential) / len(sequential),
        "sequential_improvement_over_no_query": sum(1.0 - row["cost"] for row in sequential) / len(sequential),
        "final_exact_accuracy_after_trusted_answer": sum(row["final"] == row["truth"] for row in sequential) / len(sequential),
        "false_known_after_trusted_answer": sum(row["false_known"] for row in nonknown_seq) / len(nonknown_seq),
        "safe_nonknown_after_trusted_answer": sum(row["safe_nonknown"] for row in nonknown_seq) / len(nonknown_seq),
        "irrelevant_query_intermediate_fail_closed_rate": (
            sum(
                not row["witness_valid"]
                and row["final"] == witness_config["insufficientId"]
                and row["retained"]
                and row["actual_execution_count"] == 0
                for row in intermediate_outputs
            )
            / len(intermediate_outputs)
            if intermediate_outputs
            else 1.0
        ),
        "irrelevant_query_intermediate_count": len(intermediate_outputs),
        "authoritative_true_hypothesis_retention": sum(row["retained"] for row in sequential) / len(sequential),
        "mean_generated_tokens": sum(row["generated_token_count"] for row in rows) / len(rows),
        "total_generation_seconds": sum(row["generation_seconds"] for row in rows),
        "validation_reason_counts": dict(sorted(Counter(output["validation_reason"] for output in completed.values()).items())),
        "actual_execution_count": 0,
    }
    calibration = _calibration(rows)
    gates = config["qualificationGates"]
    qualification = {
        "proposal_structural_validity": metrics["proposal_structural_validity"] >= gates["minimumProposalStructuralValidity"],
        "compatible_state_retention": metrics["compatible_state_retention"] >= gates["minimumCompatibleStateRetention"],
        "every_language_class_compatible_state_retention": all(row["compatible_state_retention"] >= gates["minimumEveryLanguageClassCompatibleStateRetention"] for row in class_metrics.values()),
        "compatible_set_exact_accuracy": metrics["compatible_set_exact_accuracy"] >= gates["minimumCompatibleSetExactAccuracy"],
        "ambiguity_sensitivity": metrics["ambiguity_sensitivity"] >= gates["minimumAmbiguitySensitivity"],
        "decidable_specificity": metrics["decidable_specificity"] >= gates["minimumDecidableSpecificity"],
        "decidable_top1_accuracy": metrics["decidable_top1_accuracy"] >= gates["minimumDecidableTop1Accuracy"],
        "query_top1_accuracy": metrics["query_top1_accuracy"] >= gates["minimumQueryTop1Accuracy"],
        "query_MRR": metrics["query_mean_reciprocal_rank"] >= gates["minimumQueryMeanReciprocalRank"],
        "mean_correct_query_rank": metrics["mean_correct_query_rank"] <= gates["maximumMeanCorrectQueryRank"],
        "false_known_singleton": metrics["false_known_singleton_rate_on_nonknown_truth"] <= gates["maximumFalseKnownSingletonRateOnNonKnownTruth"],
        "candidate_attraction": metrics["candidate_attracted_fraction_of_semantic_errors"] <= gates["maximumCandidateAttractedFractionOfSemanticErrors"],
        "sequential_cost": metrics["sequential_mean_decision_cost"] <= gates["maximumSequentialMeanDecisionCost"] + 1e-12,
        "sequential_improvement": metrics["sequential_improvement_over_no_query"] + 1e-12 >= gates["minimumSequentialImprovementOverNoQuery"],
        "final_exact_after_trusted_answer": metrics["final_exact_accuracy_after_trusted_answer"] == gates["requiredFinalExactAccuracyAfterTrustedAnswer"],
        "zero_false_known_after_trusted_answer": metrics["false_known_after_trusted_answer"] <= gates["maximumFalseKnownAfterTrustedAnswer"],
        "irrelevant_queries_fail_closed": metrics["irrelevant_query_intermediate_fail_closed_rate"] == gates["requiredIrrelevantQueryIntermediateFailClosedRate"],
        "authoritative_retention": metrics["authoritative_true_hypothesis_retention"] == gates["requiredAuthoritativeTrueHypothesisRetention"],
        "zero_execution": metrics["actual_execution_count"] <= gates["maximumActualExecutionCount"],
    }
    access_gates = config["accessGates"]
    access_checks = {
        "tokenizer_load_budget": access["tokenizer_load_count"] <= access_gates["maximumTokenizerLoadCount"],
        "model_load_budget": access["model_load_count"] <= access_gates["maximumModelLoadCount"],
        "generation_budget": access["model_generation_count"] <= access_gates["maximumModelGenerationCount"],
        "one_generation_per_fixture": access["maximum_generation_count_per_fixture"] <= access_gates["maximumGenerationCountPerFixture"],
        "zero_closed_answer_generation": access["closed_answer_model_generation_count"] <= access_gates["maximumClosedAnswerModelGenerationCount"],
        "zero_evaluation_generation": access["evaluation_fixture_model_generation_count"] <= access_gates["maximumEvaluationFixtureModelGenerationCount"],
        "zero_retries": access["retry_count"] <= access_gates["maximumRetryCount"],
        "zero_manual_raw_inspection": access["manual_raw_response_inspection_count"] <= access_gates["maximumManualRawResponseInspectionCount"],
        "zero_persisted_raw": access["persisted_raw_response_count"] <= access_gates["maximumPersistedRawResponseCount"],
        "zero_API": access["API_call_count"] <= access_gates["maximumAPICallCount"],
        "zero_training": access["training_run_count"] <= access_gates["maximumTrainingRunCount"],
        "zero_services": access["real_service_call_count"] <= access_gates["maximumRealServiceCallCount"],
        "zero_side_effects": access["external_side_effect_count"] <= access_gates["maximumExternalSideEffectCount"],
        "zero_execution": access["actual_execution_count"] <= access_gates["maximumActualExecutionCount"],
    }
    qualified = all(qualification.values()) and all(access_checks.values())
    return {
        "metrics": metrics,
        "calibration_diagnostics": calibration,
        "qualification_gates": qualification,
        "access_gates": access_checks,
        "qualified": qualified,
        "decision": config["decisionRule"]["ifEveryQualificationAndAccessGatePasses"] if qualified else config["decisionRule"]["otherwise"],
    }


__all__ = ["evaluate", "parse_proposal", "render_prompt"]
