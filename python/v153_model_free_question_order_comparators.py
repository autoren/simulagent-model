from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any

from v148_typed_witness_firewall import finalize_witness
from v150_oracle_closed_interaction_policy import decision_cost
from v152_fresh_question_order_population import CANDIDATE_PROPOSAL_FIELDS, REQUEST_STAGES, witness_from_answer_event


COMPARATORS = ("NO_QUERY", "SOURCE_ORDER", "SEEDED_RANDOM", "ORACLE_ORDER")


def build_episodes(development_metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for row in development_metadata:
        by_group.setdefault(row["group_id"], {})[row["stage"]] = row
    episodes = []
    for request in sorted(
        (row for row in development_metadata if row["stage"] in REQUEST_STAGES),
        key=lambda row: row["fixture_id"],
    ):
        if request["stage"] == "request_ambiguous":
            sides = ("left", "right")
        elif request["stage"] == "request_right":
            sides = ("right",)
        else:
            sides = ("left",)
        for side in sides:
            answer_stage = "closed_answer_known" if side == "left" else "closed_answer_right"
            answer = by_group[request["group_id"]][answer_stage]
            episodes.append(
                {
                    "episode_id": f"{request['fixture_id']}::{side}",
                    "fixture_id": request["fixture_id"],
                    "group_id": request["group_id"],
                    "family_id": request["family_id"],
                    "stage": request["stage"],
                    "side": side,
                    "truth_state_id": answer["truth_state_id"],
                    "oracle_query_id": request["oracle_query_id"],
                    "closed_answer_event": answer["closed_answer_event"],
                }
            )
    return episodes


def comparator_order(
    comparator: str,
    episode: dict[str, Any],
    query_ids: list[str],
    config: dict[str, Any],
) -> list[str]:
    if comparator == "NO_QUERY":
        return []
    if comparator == "SOURCE_ORDER":
        return list(query_ids)
    if comparator == "ORACLE_ORDER":
        return [episode["oracle_query_id"]] + [query for query in query_ids if query != episode["oracle_query_id"]]
    if comparator == "SEEDED_RANDOM":
        material = f"{config['policy']['randomSeed']}|{episode['fixture_id']}"
        seed = int(hashlib.sha256(material.encode()).hexdigest(), 16)
        order = list(query_ids)
        random.Random(seed).shuffle(order)
        return order
    raise ValueError(f"unknown comparator {comparator}")


def evaluate(
    development_metadata: list[dict[str, Any]],
    catalog: dict[str, Any],
    witness_config: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    episodes = build_episodes(development_metadata)
    query_ids = [row["query_id"] for row in catalog["queries"]]
    known_ids = set(witness_config["knownIds"])
    records = []
    intermediate = []
    for episode in episodes:
        for comparator in COMPARATORS:
            order = comparator_order(comparator, episode, query_ids, config)
            if comparator == "NO_QUERY":
                rank = None
                final = witness_config["insufficientId"]
                cost = decision_cost(episode["truth_state_id"], final, known_ids, config)
                retained = True
            else:
                rank = order.index(episode["oracle_query_id"]) + 1
                for asked_query in order[: rank - 1]:
                    output = finalize_witness(None, None, witness_config)
                    intermediate.append(
                        {
                            "episode_id": episode["episode_id"],
                            "comparator": comparator,
                            "asked_query": asked_query,
                            "oracle_query": episode["oracle_query_id"],
                            "witness_valid": output["witness_valid"],
                            "final": output["final_state_id"],
                            "retained": output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
                            "actual_execution_count": output["actual_execution_count"],
                        }
                    )
                witness = witness_from_answer_event(episode["closed_answer_event"], catalog)
                output = finalize_witness(witness, None, witness_config)
                final = output["final_state_id"]
                retained = output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"]
                cost = rank * config["policy"]["queryCost"] + decision_cost(
                    episode["truth_state_id"], final, known_ids, config
                )
            records.append(
                {
                    **episode,
                    "comparator": comparator,
                    "question_order": order,
                    "correct_query_rank": rank,
                    "final_state_id": final,
                    "decision_cost": cost,
                    "final_exact": final == episode["truth_state_id"],
                    "retained": retained,
                    "actual_execution_count": 0,
                }
            )

    metrics: dict[str, Any] = {}
    for comparator in COMPARATORS:
        subset = [row for row in records if row["comparator"] == comparator]
        relevant_intermediate = [row for row in intermediate if row["comparator"] == comparator]
        ranks = [row["correct_query_rank"] for row in subset if row["correct_query_rank"] is not None]
        metrics[comparator] = {
            "episode_count": len(subset),
            "mean_correct_query_rank": sum(ranks) / len(ranks) if ranks else None,
            "rank_counts": dict(sorted(Counter(ranks).items())) if ranks else {},
            "mean_decision_cost": sum(row["decision_cost"] for row in subset) / len(subset),
            "improvement_over_no_query": sum(1.0 - row["decision_cost"] for row in subset) / len(subset),
            "final_exact_accuracy": sum(row["final_exact"] for row in subset) / len(subset),
            "irrelevant_query_intermediate_count": len(relevant_intermediate),
            "irrelevant_query_fail_closed_rate": (
                sum(
                    not row["witness_valid"]
                    and row["final"] == witness_config["insufficientId"]
                    and row["retained"]
                    and row["actual_execution_count"] == 0
                    for row in relevant_intermediate
                ) / len(relevant_intermediate)
                if relevant_intermediate
                else 1.0
            ),
            "authoritative_hypothesis_retention": sum(row["retained"] for row in subset) / len(subset),
            "actual_execution_count": 0,
        }

    candidate_field_count = sum(
        len(CANDIDATE_PROPOSAL_FIELDS & set(row)) for row in development_metadata
    )
    gates = config["gates"]
    trusted_comparators = [metrics[name] for name in COMPARATORS if name != "NO_QUERY"]
    checks = {
        "population_counts": bool(
            len({row["group_id"] for row in development_metadata}) == gates["requiredGroupCount"]
            and sum(row["stage"] in REQUEST_STAGES for row in development_metadata) == gates["requiredRequestFixtureCount"]
            and len(episodes) == gates["requiredSequentialEpisodeCount"]
            and len(metrics) == gates["requiredComparatorCount"]
        ),
        "oracle_reference_exact": bool(
            metrics["ORACLE_ORDER"]["mean_correct_query_rank"] == gates["requiredOracleMeanCorrectQueryRank"]
            and abs(metrics["ORACLE_ORDER"]["mean_decision_cost"] - gates["requiredOracleMeanDecisionCost"]) <= 1e-12
            and abs(metrics["ORACLE_ORDER"]["improvement_over_no_query"] - gates["requiredOracleImprovementOverNoQuery"]) <= 1e-12
        ),
        "source_order_balanced": metrics["SOURCE_ORDER"]["mean_correct_query_rank"] == gates["requiredSourceMeanCorrectQueryRank"],
        "seeded_random_not_accidentally_oracle_or_adversarial": bool(
            gates["minimumSeededRandomMeanCorrectQueryRank"]
            <= metrics["SEEDED_RANDOM"]["mean_correct_query_rank"]
            <= gates["maximumSeededRandomMeanCorrectQueryRank"]
        ),
        "trusted_answer_final_exact": all(
            row["final_exact_accuracy"] == gates["requiredTrustedAnswerFinalExactAccuracy"]
            for row in trusted_comparators
        ),
        "irrelevant_queries_fail_closed": all(
            row["irrelevant_query_fail_closed_rate"] == gates["requiredIrrelevantQueryFailClosedRate"]
            for row in trusted_comparators
        ),
        "authoritative_hypotheses_retained": all(
            row["authoritative_hypothesis_retention"] == gates["requiredAuthoritativeHypothesisRetention"]
            for row in metrics.values()
        ),
        "no_candidate_proposal_surface": candidate_field_count == gates["requiredCandidateProposalFieldCount"],
        "zero_evaluation_model_API_training_execution": bool(
            config["population"]["evaluationLanguageReadCount"] <= gates["maximumEvaluationLanguageReadCount"]
            and all(row["actual_execution_count"] <= gates["maximumActualExecutionCount"] for row in metrics.values())
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "episode_count": len(episodes),
        "candidate_proposal_field_count": candidate_field_count,
        "evaluation_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_or_score_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "actual_execution_count": 0,
    }


__all__ = ["COMPARATORS", "build_episodes", "comparator_order", "evaluate"]
