from __future__ import annotations

import json
from collections import Counter
from typing import Any

from v148_typed_witness_firewall import finalize_witness
from v150_oracle_closed_interaction_policy import decision_cost
from v152_fresh_question_order_population import witness_from_answer_event


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
            "options": [
                {"option_id": option["option_id"], "text": option["text"]}
                for option in row["options"]
            ],
        }
        for row in catalog["queries"]
    ]
    payload = {
        "instruction": config["prompt"]["instruction"],
        "registered_clarification_questions": public_queries,
        "conversation": fixture["conversation"],
        "response_contract": {
            "query_ranking": "every supplied query ID exactly once, most useful first"
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _invalid(reason: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "ranking_valid": False,
        "validation_reason": reason,
        "normalized_ranking": None,
        "query_ranking": list(config["fallbackQueryRanking"]),
        "permanently_non_authoritative": True,
        "authoritative_hypothesis_universe_pruned": False,
        "capability_defined_or_registered": False,
        "executable": False,
        "actual_execution_count": 0,
    }


def parse_ranking(raw: str, catalog: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw.strip(), object_pairs_hook=_strict_object)
    except (_DuplicateKeyError, json.JSONDecodeError, AttributeError):
        return _invalid("invalid_json", config)
    if not isinstance(value, dict) or set(value) != set(config["prompt"]["responseKeys"]):
        return _invalid("invalid_object_shape", config)
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
    normalized = {"query_ranking": list(ranking)}
    return {
        "ranking_valid": True,
        "validation_reason": "valid_registered_query_ranking",
        "normalized_ranking": normalized,
        **normalized,
        "permanently_non_authoritative": True,
        "authoritative_hypothesis_universe_pruned": False,
        "capability_defined_or_registered": False,
        "executable": False,
        "actual_execution_count": 0,
    }


def prepare_bounded_final_prompt_tokens(
    prompt_tokens: list[int],
    reasoning_tokens: list[int],
    tokenizer: Any,
) -> tuple[list[int], bool, int]:
    """Close thinking mechanically and reserve a distinct final continuation.

    The returned prompt never includes model text after the first natural closing tag.
    When no closing tag appears inside the bounded reasoning phase, one is injected.
    """
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    retained = list(reasoning_tokens)
    while retained and retained[-1] in set(eos_ids):
        retained.pop()
    text = tokenizer.decode(retained)
    close = "</think>"
    natural_close = close in text
    if natural_close:
        continuation = text.split(close, 1)[0] + close + "\n\n"
    else:
        continuation = text + "\n</think>\n\n"
    continuation_tokens = tokenizer.encode(continuation, add_special_tokens=False)
    return list(prompt_tokens) + list(continuation_tokens), natural_close, len(retained)


def evaluate_condition(
    completed: dict[str, dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    answer_metadata: list[dict[str, Any]],
    catalog: dict[str, Any],
    witness_config: dict[str, Any],
    comparator_config: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden_by_id = {row["fixture_id"]: row for row in hidden_rows}
    if set(completed) != set(hidden_by_id):
        raise ValueError("V154 development request completion mismatch")
    rows = []
    for fixture_id, output in completed.items():
        hidden = hidden_by_id[fixture_id]
        ranking = list(output["query_ranking"])
        rank = ranking.index(hidden["oracle_query_id"]) + 1
        rows.append(
            {
                "fixture_id": fixture_id,
                "group_id": hidden["group_id"],
                "family_id": hidden["family_id"],
                "stage": hidden["stage"],
                "ranking_valid": output["ranking_valid"],
                "validation_reason": output["validation_reason"],
                "query_ranking": ranking,
                "query_rank": rank,
                "query_top1": rank == 1,
                "generated_token_count": output["generated_token_count"],
                "generation_seconds": output["generation_seconds"],
                "candidate_proposal_field_count": sum(
                    key in output
                    for key in (
                        "candidate_state_ids", "state_ranking", "compatible_state_proposal",
                        "llm_proposal", "confidence",
                    )
                ),
            }
        )
    rows.sort(key=lambda row: row["fixture_id"])

    answer_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for row in answer_metadata:
        answer_by_group.setdefault(row["group_id"], {})[row["stage"]] = row
    known_ids = set(witness_config["knownIds"])
    sequential = []
    intermediate = []
    for row in rows:
        if row["stage"] == "request_ambiguous":
            sides = ("left", "right")
        elif row["stage"] == "request_right":
            sides = ("right",)
        else:
            sides = ("left",)
        for side in sides:
            answer_stage = "closed_answer_known" if side == "left" else "closed_answer_right"
            answer = answer_by_group[row["group_id"]][answer_stage]
            query_count = row["query_rank"]
            for asked_query in row["query_ranking"][: query_count - 1]:
                output = finalize_witness(None, None, witness_config)
                intermediate.append(
                    {
                        "asked_query": asked_query,
                        "oracle_query": answer["oracle_query_id"],
                        "witness_valid": output["witness_valid"],
                        "final": output["final_state_id"],
                        "retained": output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
                        "actual_execution_count": output["actual_execution_count"],
                    }
                )
            witness = witness_from_answer_event(answer["closed_answer_event"], catalog)
            output = finalize_witness(witness, None, witness_config)
            final = output["final_state_id"]
            cost = query_count * comparator_config["policy"]["queryCost"] + decision_cost(
                answer["truth_state_id"], final, known_ids, comparator_config
            )
            sequential.append(
                {
                    "truth": answer["truth_state_id"],
                    "final": final,
                    "cost": cost,
                    "retained": output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
                    "actual_execution_count": output["actual_execution_count"],
                }
            )

    metrics = {
        "request_fixture_count": len(rows),
        "sequential_episode_count": len(sequential),
        "structural_validity": sum(row["ranking_valid"] for row in rows) / len(rows),
        "query_top1_accuracy": sum(row["query_top1"] for row in rows) / len(rows),
        "query_mean_reciprocal_rank": sum(1.0 / row["query_rank"] for row in rows) / len(rows),
        "mean_correct_query_rank": sum(row["query_rank"] for row in rows) / len(rows),
        "rank_counts": dict(sorted(Counter(row["query_rank"] for row in rows).items())),
        "sequential_mean_decision_cost": sum(row["cost"] for row in sequential) / len(sequential),
        "sequential_improvement_over_no_query": sum(1.0 - row["cost"] for row in sequential) / len(sequential),
        "final_exact_accuracy_after_trusted_answer": sum(row["truth"] == row["final"] for row in sequential) / len(sequential),
        "irrelevant_query_intermediate_count": len(intermediate),
        "irrelevant_query_intermediate_fail_closed_rate": (
            sum(
                not row["witness_valid"]
                and row["final"] == witness_config["insufficientId"]
                and row["retained"]
                and row["actual_execution_count"] == 0
                for row in intermediate
            ) / len(intermediate)
            if intermediate
            else 1.0
        ),
        "authoritative_hypothesis_retention": sum(row["retained"] for row in sequential) / len(sequential),
        "candidate_proposal_field_count": sum(row["candidate_proposal_field_count"] for row in rows),
        "mean_generated_tokens": sum(row["generated_token_count"] for row in rows) / len(rows),
        "total_generation_seconds": sum(row["generation_seconds"] for row in rows),
        "validation_reason_counts": dict(sorted(Counter(row["validation_reason"] for row in rows).items())),
        "actual_execution_count": sum(row["actual_execution_count"] for row in sequential),
    }
    gates = config["qualificationGates"]
    qualification = {
        "structural_validity": metrics["structural_validity"] >= gates["minimumStructuralValidity"],
        "query_top1_accuracy": metrics["query_top1_accuracy"] >= gates["minimumQueryTop1Accuracy"],
        "query_MRR": metrics["query_mean_reciprocal_rank"] >= gates["minimumQueryMeanReciprocalRank"],
        "mean_correct_query_rank": metrics["mean_correct_query_rank"] <= gates["maximumMeanCorrectQueryRank"],
        "sequential_cost": metrics["sequential_mean_decision_cost"] <= gates["maximumSequentialMeanDecisionCost"] + 1e-12,
        "sequential_improvement": metrics["sequential_improvement_over_no_query"] + 1e-12 >= gates["minimumSequentialImprovementOverNoQuery"],
        "final_exact_after_trusted_answer": metrics["final_exact_accuracy_after_trusted_answer"] == gates["requiredFinalExactAccuracyAfterTrustedAnswer"],
        "irrelevant_queries_fail_closed": metrics["irrelevant_query_intermediate_fail_closed_rate"] == gates["requiredIrrelevantQueryIntermediateFailClosedRate"],
        "authoritative_retention": metrics["authoritative_hypothesis_retention"] == gates["requiredAuthoritativeHypothesisRetention"],
        "zero_candidate_proposal_fields": metrics["candidate_proposal_field_count"] <= gates["maximumCandidateProposalFieldCount"],
        "zero_execution": metrics["actual_execution_count"] <= gates["maximumActualExecutionCount"],
    }
    return {"metrics": metrics, "qualification_gates": qualification, "qualified": all(qualification.values())}


__all__ = [
    "evaluate_condition", "parse_ranking", "prepare_bounded_final_prompt_tokens", "render_prompt",
]
