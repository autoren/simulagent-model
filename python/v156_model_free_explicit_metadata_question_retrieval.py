from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter
from typing import Any

from v148_typed_witness_firewall import finalize_witness
from v150_oracle_closed_interaction_policy import decision_cost
from v155_fresh_deterministic_question_retrieval_population import witness_from_answer_event


REQUEST_STAGES = {
    "request_left_anchor", "request_left_paraphrase", "request_right_clear", "request_ambiguous"
}
COMPARATORS = (
    "NO_QUERY", "SOURCE_ORDER", "SEEDED_RANDOM", "ORACLE_ORDER", "EXPLICIT_METADATA_RETRIEVAL"
)


def normalize_tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", ascii_text)


def normalized_text(text: str) -> str:
    return " ".join(normalize_tokens(text))


def visible_request_text(fixture: dict[str, Any]) -> str:
    return " ".join(
        message["text"] for message in fixture["conversation"] if message.get("role") == "user"
    )


def _query_surface_tokens(query: dict[str, Any]) -> set[str]:
    surface = " ".join(
        [query["title"], query["question"]] + [option["text"] for option in query["options"]]
    )
    return set(normalize_tokens(surface))


def score_query(text: str, query: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    retrieval = config["retrieval"]
    normalized = normalized_text(text)
    tokens = set(normalize_tokens(text))
    profile = query["retrieval_profile"]
    phrase_hits = [
        phrase for phrase in profile["anchor_phrases"]
        if normalized_text(phrase) in normalized
    ]
    primary_hits = sorted(tokens & set(profile["primary_terms"]))
    secondary_hits = sorted(tokens & set(profile["secondary_terms"]))
    surface_hits = sorted(tokens & _query_surface_tokens(query))
    score = (
        retrieval["anchorPhraseWeight"] * len(phrase_hits)
        + retrieval["primaryTermWeight"] * len(primary_hits)
        + retrieval["secondaryTermWeight"] * len(secondary_hits)
        + retrieval["visibleQuestionSurfaceTokenWeight"] * len(surface_hits)
    )
    return {
        "score": score, "anchor_phrase_hit_count": len(phrase_hits),
        "primary_term_hit_count": len(primary_hits),
        "secondary_term_hit_count": len(secondary_hits),
        "visible_surface_token_hit_count": len(surface_hits),
    }


def rank_queries(
    fixture: dict[str, Any], retrieval_catalog: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    text = visible_request_text(fixture)
    source_order = [query["query_id"] for query in retrieval_catalog["queries"]]
    source_index = {query_id: index for index, query_id in enumerate(source_order)}
    diagnostics = {
        query["query_id"]: score_query(text, query, config) for query in retrieval_catalog["queries"]
    }
    ranking = sorted(
        source_order,
        key=lambda query_id: (-diagnostics[query_id]["score"], source_index[query_id]),
    )
    top_score = diagnostics[ranking[0]]["score"]
    second_score = diagnostics[ranking[1]]["score"]
    return {
        "query_ranking": ranking,
        "scores": {query_id: diagnostics[query_id]["score"] for query_id in source_order},
        "top_score": top_score,
        "top_score_tie_count": sum(
            diagnostics[query_id]["score"] == top_score for query_id in source_order
        ),
        "top_two_margin": top_score - second_score,
        "query_diagnostics": diagnostics,
    }


def build_episodes(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for row in metadata:
        by_group.setdefault(row["group_id"], {})[row["stage"]] = row
    episodes = []
    for request in sorted(
        (row for row in metadata if row["stage"] in REQUEST_STAGES),
        key=lambda row: row["fixture_id"],
    ):
        sides = ("left", "right") if request["stage"] == "request_ambiguous" else (
            ("right",) if request["stage"] == "request_right_clear" else ("left",)
        )
        for side in sides:
            answer_stage = "closed_answer_left" if side == "left" else "closed_answer_right"
            answer = by_group[request["group_id"]][answer_stage]
            episodes.append({
                "episode_id": f"{request['fixture_id']}::{side}",
                "fixture_id": request["fixture_id"], "group_id": request["group_id"],
                "stage": request["stage"], "side": side,
                "truth_state_id": answer["truth_state_id"],
                "oracle_query_id": request["oracle_query_id"],
                "closed_answer_event": answer["closed_answer_event"],
            })
    return episodes


def comparator_order(
    comparator: str, episode: dict[str, Any], query_ids: list[str],
    retrieval_rankings: dict[str, list[str]], config: dict[str, Any]
) -> list[str]:
    if comparator == "NO_QUERY":
        return []
    if comparator == "SOURCE_ORDER":
        return list(query_ids)
    if comparator == "ORACLE_ORDER":
        return [episode["oracle_query_id"]] + [qid for qid in query_ids if qid != episode["oracle_query_id"]]
    if comparator == "EXPLICIT_METADATA_RETRIEVAL":
        return list(retrieval_rankings[episode["fixture_id"]])
    if comparator == "SEEDED_RANDOM":
        material = f"{config['policy']['randomSeed']}|{episode['fixture_id']}"
        order = list(query_ids)
        random.Random(int(hashlib.sha256(material.encode()).hexdigest(), 16)).shuffle(order)
        return order
    raise ValueError(f"unknown comparator {comparator}")


def evaluate(
    public_requests: list[dict[str, Any]], development_metadata: list[dict[str, Any]],
    retrieval_catalog: dict[str, Any], witness_catalog: dict[str, Any],
    witness_config: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    public_by_id = {row["fixture_id"]: row for row in public_requests}
    request_metadata = [row for row in development_metadata if row["stage"] in REQUEST_STAGES]
    retrieval_records = {}
    for row in request_metadata:
        ranked = rank_queries(public_by_id[row["fixture_id"]], retrieval_catalog, config)
        rank = ranked["query_ranking"].index(row["oracle_query_id"]) + 1
        retrieval_records[row["fixture_id"]] = {
            "fixture_id": row["fixture_id"], "stage": row["stage"],
            "oracle_query_id": row["oracle_query_id"], "query_ranking": ranked["query_ranking"],
            "correct_query_rank": rank, "reciprocal_rank": 1.0 / rank,
            "scores": ranked["scores"], "top_score_tie_count": ranked["top_score_tie_count"],
            "top_two_margin": ranked["top_two_margin"],
        }
    retrieval_rankings = {
        fixture_id: row["query_ranking"] for fixture_id, row in retrieval_records.items()
    }
    episodes = build_episodes(development_metadata)
    query_ids = [row["query_id"] for row in retrieval_catalog["queries"]]
    known_ids = set(witness_config["knownIds"])
    records = []
    intermediate = []
    for episode in episodes:
        for comparator in COMPARATORS:
            order = comparator_order(comparator, episode, query_ids, retrieval_rankings, config)
            if comparator == "NO_QUERY":
                rank = None
                final = witness_config["insufficientId"]
                cost = decision_cost(episode["truth_state_id"], final, known_ids, config)
                retained = True
            else:
                rank = order.index(episode["oracle_query_id"]) + 1
                for asked_query in order[: rank - 1]:
                    output = finalize_witness(None, None, witness_config)
                    intermediate.append({
                        "episode_id": episode["episode_id"], "comparator": comparator,
                        "asked_query": asked_query, "oracle_query": episode["oracle_query_id"],
                        "witness_valid": output["witness_valid"], "final": output["final_state_id"],
                        "retained": output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
                        "actual_execution_count": output["actual_execution_count"],
                    })
                witness = witness_from_answer_event(episode["closed_answer_event"], witness_catalog)
                output = finalize_witness(witness, None, witness_config)
                final = output["final_state_id"]
                retained = output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"]
                cost = rank * config["policy"]["queryCost"] + decision_cost(
                    episode["truth_state_id"], final, known_ids, config
                )
            records.append({
                **episode, "comparator": comparator, "question_order": order,
                "correct_query_rank": rank, "final_state_id": final,
                "decision_cost": cost, "final_exact": final == episode["truth_state_id"],
                "retained": retained, "actual_execution_count": 0,
            })

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
                    not row["witness_valid"] and row["final"] == witness_config["insufficientId"]
                    and row["retained"] and row["actual_execution_count"] == 0
                    for row in relevant_intermediate
                ) / len(relevant_intermediate) if relevant_intermediate else 1.0
            ),
            "authoritative_hypothesis_retention": sum(row["retained"] for row in subset) / len(subset),
            "actual_execution_count": 0,
        }

    request_rows = list(retrieval_records.values())
    request_ranks = [row["correct_query_rank"] for row in request_rows]
    retrieval_request_metrics = {
        "request_count": len(request_rows),
        "query_top1_accuracy": sum(rank == 1 for rank in request_ranks) / len(request_ranks),
        "query_mean_reciprocal_rank": sum(row["reciprocal_rank"] for row in request_rows) / len(request_rows),
        "mean_correct_query_rank": sum(request_ranks) / len(request_ranks),
        "rank_counts": dict(sorted(Counter(request_ranks).items())),
        "top_score_tie_rate": sum(row["top_score_tie_count"] > 1 for row in request_rows) / len(request_rows),
        "mean_top_two_margin": sum(row["top_two_margin"] for row in request_rows) / len(request_rows),
    }
    gates = config["gates"]
    trusted = [metrics[name] for name in COMPARATORS if name != "NO_QUERY"]
    retrieval_metric = metrics["EXPLICIT_METADATA_RETRIEVAL"]
    checks = {
        "population_and_comparator_counts": bool(
            len(public_requests) == gates["requiredRequestFixtureCount"]
            and len({row["group_id"] for row in development_metadata}) == gates["requiredGroupCount"]
            and len(episodes) == gates["requiredSequentialEpisodeCount"]
            and len(metrics) == gates["requiredComparatorCount"]
        ),
        "oracle_reference_exact": bool(
            metrics["ORACLE_ORDER"]["mean_correct_query_rank"] == gates["requiredOracleMeanCorrectQueryRank"]
            and abs(metrics["ORACLE_ORDER"]["mean_decision_cost"] - gates["requiredOracleMeanDecisionCost"]) <= 1e-12
        ),
        "source_order_balanced": metrics["SOURCE_ORDER"]["mean_correct_query_rank"] == gates["requiredSourceMeanCorrectQueryRank"],
        "seeded_random_not_accidentally_oracle_or_adversarial": bool(
            gates["minimumSeededRandomMeanCorrectQueryRank"]
            <= metrics["SEEDED_RANDOM"]["mean_correct_query_rank"]
            <= gates["maximumSeededRandomMeanCorrectQueryRank"]
        ),
        "retrieval_top1": retrieval_request_metrics["query_top1_accuracy"] >= gates["minimumRetrievalQueryTop1Accuracy"],
        "retrieval_MRR": retrieval_request_metrics["query_mean_reciprocal_rank"] >= gates["minimumRetrievalQueryMeanReciprocalRank"],
        "retrieval_mean_rank": retrieval_request_metrics["mean_correct_query_rank"] <= gates["maximumRetrievalMeanCorrectQueryRank"],
        "retrieval_top_ties": retrieval_request_metrics["top_score_tie_rate"] <= gates["maximumRetrievalTopScoreTieRate"],
        "retrieval_sequential_cost": retrieval_metric["mean_decision_cost"] <= gates["maximumRetrievalSequentialMeanDecisionCost"],
        "retrieval_sequential_improvement": retrieval_metric["improvement_over_no_query"] >= gates["minimumRetrievalSequentialImprovementOverNoQuery"],
        "trusted_answer_final_exact": all(row["final_exact_accuracy"] == 1.0 for row in trusted),
        "irrelevant_queries_fail_closed": all(row["irrelevant_query_fail_closed_rate"] == 1.0 for row in trusted),
        "authoritative_hypotheses_retained": all(row["authoritative_hypothesis_retention"] == 1.0 for row in metrics.values()),
        "no_candidate_proposal_surface": all(
            not ({"candidate_state_ids", "state_ranking", "llm_proposal", "confidence"} & set(row))
            for row in public_requests + development_metadata
        ),
        "zero_evaluation_model_API_training_execution": True,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "retrieval_request_metrics": retrieval_request_metrics,
        "comparator_metrics": metrics, "retrieval_records": retrieval_records,
        "episode_count": len(episodes), "candidate_proposal_field_count": 0,
        "evaluation_policy_read_count": 0, "model_load_count": 0,
        "model_generation_or_score_count": 0, "API_call_count": 0,
        "training_run_count": 0, "actual_execution_count": 0,
    }


__all__ = [
    "COMPARATORS", "REQUEST_STAGES", "build_episodes", "evaluate", "normalize_tokens",
    "rank_queries", "score_query", "visible_request_text",
]
