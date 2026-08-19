from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from v148_typed_witness_firewall import finalize_witness
from v150_oracle_closed_interaction_policy import decision_cost
from v156_model_free_explicit_metadata_question_retrieval import rank_queries
from v159_fresh_controlled_relational_grammar_population import (
    route_from_answer_event,
    witness_from_answer_event,
)


REQUEST_STAGES = {
    "request_lexical_control",
    "request_grammar_unique",
    "request_grammar_conflict",
    "request_insufficient",
}
FALLBACK_STAGES = {"request_grammar_conflict", "request_insufficient"}
COMPARATORS = (
    "NO_QUERY",
    "SOURCE_SPECIFIC_THEN_GENERIC",
    "LEXICAL_MARGIN_ROUTER",
    "ALWAYS_GENERIC",
    "INFORMATION_ORACLE",
    "GRAMMAR_RETRIEVAL_ROUTER",
)


def normalize_alias(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _request_text(fixture: dict[str, Any]) -> str:
    return "\n".join(
        str(turn.get("text", ""))
        for turn in fixture.get("conversation", [])
        if isinstance(turn, dict) and turn.get("role") == "user"
    )


def _alias_map(catalog: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for query in catalog["queries"]:
        for alias in query["grammar_aliases"]:
            normalized = normalize_alias(alias)
            if normalized in aliases:
                raise ValueError(f"duplicate normalized grammar alias: {normalized}")
            aliases[normalized] = query["query_id"]
    return aliases


def choose_lexical_query(
    fixture: dict[str, Any], state_free_catalog: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    ranked = rank_queries(fixture, state_free_catalog, config)
    top_query = ranked["query_ranking"][0]
    use_specific = bool(
        ranked["top_score"] >= config["retrieval"]["minimumTopScoreForSpecific"]
        and ranked["top_two_margin"] >= config["retrieval"]["minimumTopTwoMarginForSpecific"]
        and (
            not config["retrieval"]["requireUniqueTopScore"]
            or ranked["top_score_tie_count"] == 1
        )
    )
    return {
        "initial_query_id": top_query if use_specific else config["policy"]["genericQueryId"],
        "specific_query_ranking": ranked["query_ranking"],
        "scores": ranked["scores"],
        "top_score": ranked["top_score"],
        "top_two_margin": ranked["top_two_margin"],
        "top_score_tie_count": ranked["top_score_tie_count"],
        "specific_selected": use_specific,
        "decision_source": "STRICT_RETRIEVAL",
        "grammar_status": "NOT_APPLICABLE",
        "quoted_surface_count": 0,
        "registered_alias_count": 0,
        "unknown_alias_count": 0,
        "grammar_query_count": 0,
    }


def choose_initial_query(
    fixture: dict[str, Any], state_free_catalog: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    text = unicodedata.normalize("NFKC", _request_text(fixture))
    quoted = re.findall(config["grammar"]["quotedAliasPattern"], text)
    if not quoted:
        return choose_lexical_query(fixture, state_free_catalog, config)

    aliases = _alias_map(state_free_catalog)
    normalized = [normalize_alias(value) for value in quoted]
    registered = [aliases[value] for value in normalized if value in aliases]
    unknown_count = sum(value not in aliases for value in normalized)
    query_ids = sorted(set(registered))
    if unknown_count:
        status = "UNKNOWN_ALIAS"
        selected = config["policy"]["genericQueryId"]
    elif len(query_ids) == 1 and registered:
        status = "UNIQUE_REGISTERED_QUERY"
        selected = query_ids[0]
    elif len(query_ids) > 1:
        status = "CROSS_QUERY_CONFLICT"
        selected = config["policy"]["genericQueryId"]
    else:
        status = "NO_REGISTERED_ALIAS"
        selected = config["policy"]["genericQueryId"]
    return {
        "initial_query_id": selected,
        "specific_query_ranking": [],
        "scores": {},
        "top_score": 0.0,
        "top_two_margin": 0.0,
        "top_score_tie_count": 0,
        "specific_selected": selected != config["policy"]["genericQueryId"],
        "decision_source": "REGISTERED_RELATION_GRAMMAR",
        "grammar_status": status,
        "quoted_surface_count": len(quoted),
        "registered_alias_count": len(registered),
        "unknown_alias_count": unknown_count,
        "grammar_query_count": len(query_ids),
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
        if request["stage"] == "request_lexical_control":
            side = "left"
            route_stage = "closed_route_family"
        elif request["stage"] == "request_grammar_unique":
            side = "right"
            route_stage = "closed_route_family"
        else:
            side = "unclear"
            route_stage = "closed_route_unclear"
        specific = None if side == "unclear" else by_group[request["group_id"]][f"closed_specific_{side}"]
        route = by_group[request["group_id"]][route_stage]
        truth = request["truth_state_id"] if specific is None else specific["truth_state_id"]
        episodes.append(
            {
                "episode_id": f"{request['fixture_id']}::{side}",
                "fixture_id": request["fixture_id"],
                "group_id": request["group_id"],
                "stage": request["stage"],
                "stratum": request["stratum"],
                "side": side,
                "truth_state_id": truth,
                "oracle_specific_query_id": request["oracle_specific_query_id"],
                "target_initial_query_id": request["oracle_initial_query_id"],
                "route_event": route["closed_answer_event"],
                "specific_answer_event": specific["closed_answer_event"] if specific is not None else None,
            }
        )
    return episodes


def evaluate(
    public_requests: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    state_free_catalog: dict[str, Any],
    witness_catalog: dict[str, Any],
    witness_config: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    public_by_id = {row["fixture_id"]: row for row in public_requests}
    request_metadata = [row for row in metadata if row["stage"] in REQUEST_STAGES]
    routing_records: dict[str, dict[str, Any]] = {}
    for row in request_metadata:
        public = public_by_id[row["fixture_id"]]
        grammar = choose_initial_query(public, state_free_catalog, config)
        lexical = choose_lexical_query(public, state_free_catalog, config)
        routing_records[row["fixture_id"]] = {
            "fixture_id": row["fixture_id"],
            "stage": row["stage"],
            "stratum": row["stratum"],
            "target_initial_query_id": row["oracle_initial_query_id"],
            "grammar_initial_query_id": grammar["initial_query_id"],
            "grammar_initial_action_exact": grammar["initial_query_id"] == row["oracle_initial_query_id"],
            "grammar_decision_source": grammar["decision_source"],
            "grammar_status": grammar["grammar_status"],
            "quoted_surface_count": grammar["quoted_surface_count"],
            "registered_alias_count": grammar["registered_alias_count"],
            "unknown_alias_count": grammar["unknown_alias_count"],
            "grammar_query_count": grammar["grammar_query_count"],
            "lexical_initial_query_id": lexical["initial_query_id"],
            "lexical_initial_action_exact": lexical["initial_query_id"] == row["oracle_initial_query_id"],
            "lexical_top_score": lexical["top_score"],
            "lexical_top_two_margin": lexical["top_two_margin"],
            "lexical_top_score_tie_count": lexical["top_score_tie_count"],
        }

    episodes = build_episodes(metadata)
    query_ids = [row["query_id"] for row in state_free_catalog["queries"]]
    known_ids = set(witness_config["knownIds"])
    specific_cost = config["policy"]["specificQueryCost"]
    generic_cost = config["policy"]["genericRouteQueryCost"]
    generic_id = config["policy"]["genericQueryId"]
    records: list[dict[str, Any]] = []
    intermediates: list[dict[str, Any]] = []

    def fail_closed_intermediate(episode: dict[str, Any], comparator: str, query_id: str) -> None:
        output = finalize_witness(None, None, witness_config)
        intermediates.append(
            {
                "episode_id": episode["episode_id"],
                "comparator": comparator,
                "query_id": query_id,
                "witness_valid": output["witness_valid"],
                "final_state_id": output["final_state_id"],
                "retained": output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
                "actual_execution_count": output["actual_execution_count"],
            }
        )

    def finish_specific(episode: dict[str, Any]) -> tuple[str, bool]:
        witness = witness_from_answer_event(episode["specific_answer_event"], witness_catalog)
        output = finalize_witness(witness, None, witness_config)
        return (
            output["final_state_id"],
            output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
        )

    def finish_generic(episode: dict[str, Any]) -> tuple[str, bool, float]:
        route = route_from_answer_event(episode["route_event"], witness_catalog)
        route_output = finalize_witness(None, None, witness_config)
        retained = route_output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"]
        if route["target_query_id"] is None:
            return witness_config["insufficientId"], retained, generic_cost
        final, specific_retained = finish_specific(episode)
        return final, retained and specific_retained, generic_cost + specific_cost

    def finish_router(
        episode: dict[str, Any], comparator: str, initial_query: str
    ) -> tuple[str, bool, float, int]:
        if initial_query == generic_id:
            final, retained, cost = finish_generic(episode)
            return final, retained, cost, 0
        if episode["specific_answer_event"] is not None and initial_query == episode["oracle_specific_query_id"]:
            final, retained = finish_specific(episode)
            return final, retained, specific_cost, 0
        fail_closed_intermediate(episode, comparator, initial_query)
        final, retained, fallback_cost = finish_generic(episode)
        return final, retained, specific_cost + fallback_cost, 1

    for episode in episodes:
        for comparator in COMPARATORS:
            initial_query: str | None = None
            wrong_specific_count = 0
            if comparator == "NO_QUERY":
                final = witness_config["insufficientId"]
                retained = True
                interaction_cost = 0.0
            elif comparator == "ALWAYS_GENERIC":
                initial_query = generic_id
                final, retained, interaction_cost = finish_generic(episode)
            elif comparator == "INFORMATION_ORACLE":
                initial_query = episode["target_initial_query_id"]
                if initial_query == generic_id:
                    final, retained, interaction_cost = finish_generic(episode)
                else:
                    final, retained = finish_specific(episode)
                    interaction_cost = specific_cost
            elif comparator in {"GRAMMAR_RETRIEVAL_ROUTER", "LEXICAL_MARGIN_ROUTER"}:
                field = (
                    "grammar_initial_query_id"
                    if comparator == "GRAMMAR_RETRIEVAL_ROUTER"
                    else "lexical_initial_query_id"
                )
                initial_query = routing_records[episode["fixture_id"]][field]
                final, retained, interaction_cost, wrong_specific_count = finish_router(
                    episode, comparator, initial_query
                )
            else:
                initial_query = query_ids[0]
                if episode["specific_answer_event"] is None:
                    for query_id in query_ids:
                        fail_closed_intermediate(episode, comparator, query_id)
                    wrong_specific_count = len(query_ids)
                    final, retained, fallback_cost = finish_generic(episode)
                    interaction_cost = len(query_ids) * specific_cost + fallback_cost
                else:
                    rank = query_ids.index(episode["oracle_specific_query_id"]) + 1
                    for query_id in query_ids[: rank - 1]:
                        fail_closed_intermediate(episode, comparator, query_id)
                    wrong_specific_count = rank - 1
                    final, retained = finish_specific(episode)
                    interaction_cost = rank * specific_cost
            total_cost = interaction_cost + decision_cost(
                episode["truth_state_id"], final, known_ids, config
            )
            records.append(
                {
                    **episode,
                    "comparator": comparator,
                    "initial_query_id": initial_query,
                    "wrong_specific_count": wrong_specific_count,
                    "final_state_id": final,
                    "interaction_cost": interaction_cost,
                    "decision_cost": total_cost,
                    "final_exact": final == episode["truth_state_id"],
                    "retained": retained,
                    "actual_execution_count": 0,
                }
            )

    metrics: dict[str, dict[str, Any]] = {}
    for comparator in COMPARATORS:
        subset = [row for row in records if row["comparator"] == comparator]
        relevant = [row for row in intermediates if row["comparator"] == comparator]
        metrics[comparator] = {
            "episode_count": len(subset),
            "mean_decision_cost": sum(row["decision_cost"] for row in subset) / len(subset),
            "mean_interaction_cost": sum(row["interaction_cost"] for row in subset) / len(subset),
            "final_exact_accuracy": sum(row["final_exact"] for row in subset) / len(subset),
            "wrong_specific_question_count": sum(row["wrong_specific_count"] for row in subset),
            "irrelevant_intermediate_count": len(relevant),
            "irrelevant_intermediate_fail_closed_rate": (
                sum(
                    not row["witness_valid"]
                    and row["final_state_id"] == witness_config["insufficientId"]
                    and row["retained"]
                    and row["actual_execution_count"] == 0
                    for row in relevant
                )
                / len(relevant)
                if relevant
                else 1.0
            ),
            "authoritative_hypothesis_retention": sum(row["retained"] for row in subset) / len(subset),
            "actual_execution_count": 0,
        }

    routing_rows = list(routing_records.values())
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in routing_rows:
        by_stratum[row["stratum"]].append(row)
    grammar_cost = metrics["GRAMMAR_RETRIEVAL_ROUTER"]["mean_decision_cost"]
    no_query_cost = metrics["NO_QUERY"]["mean_decision_cost"]
    always_generic_cost = metrics["ALWAYS_GENERIC"]["mean_decision_cost"]
    oracle_cost = metrics["INFORMATION_ORACLE"]["mean_decision_cost"]
    routing_metrics = {
        "request_count": len(routing_rows),
        "grammar_initial_action_accuracy": sum(row["grammar_initial_action_exact"] for row in routing_rows) / len(routing_rows),
        "lexical_initial_action_accuracy": sum(row["lexical_initial_action_exact"] for row in routing_rows) / len(routing_rows),
        "lexical_control_specific_rate": sum(row["grammar_initial_query_id"] != generic_id for row in by_stratum["lexical_control"]) / len(by_stratum["lexical_control"]),
        "grammar_unique_specific_rate": sum(row["grammar_initial_query_id"] != generic_id for row in by_stratum["grammar_unique"]) / len(by_stratum["grammar_unique"]),
        "grammar_conflict_generic_rate": sum(row["grammar_initial_query_id"] == generic_id for row in by_stratum["grammar_conflict"]) / len(by_stratum["grammar_conflict"]),
        "insufficient_generic_rate": sum(row["grammar_initial_query_id"] == generic_id for row in by_stratum["insufficient"]) / len(by_stratum["insufficient"]),
        "incorrect_specific_rate_on_grammar_fallback_strata": sum(
            row["grammar_initial_query_id"] != generic_id
            for stage in ("grammar_conflict", "insufficient")
            for row in by_stratum[stage]
        ) / sum(len(by_stratum[stage]) for stage in ("grammar_conflict", "insufficient")),
        "grammar_status_counts": dict(sorted(Counter(row["grammar_status"] for row in routing_rows).items())),
        "grammar_decision_source_counts": dict(sorted(Counter(row["grammar_decision_source"] for row in routing_rows).items())),
        "grammar_initial_action_counts": dict(sorted(Counter(row["grammar_initial_query_id"] for row in routing_rows).items())),
        "lexical_initial_action_counts": dict(sorted(Counter(row["lexical_initial_query_id"] for row in routing_rows).items())),
        "improvement_over_no_query": no_query_cost - grammar_cost,
        "improvement_over_always_generic": always_generic_cost - grammar_cost,
        "cost_gap_from_information_oracle": grammar_cost - oracle_cost,
    }
    gates = config["gates"]
    checks = {
        "population_and_comparator_counts": bool(
            len(public_requests) == 64
            and len(metadata) == 128
            and len({row["group_id"] for row in metadata}) == 16
            and len(episodes) == 64
            and len(metrics) == 6
        ),
        "grammar_router_initial_action_exact": routing_metrics["grammar_initial_action_accuracy"] == gates["requiredGrammarRouterInitialActionAccuracy"],
        "lexical_controls_use_specific": routing_metrics["lexical_control_specific_rate"] == gates["requiredLexicalControlSpecificRate"],
        "unique_grammar_uses_specific": routing_metrics["grammar_unique_specific_rate"] == gates["requiredGrammarUniqueSpecificRate"],
        "grammar_conflicts_use_generic": routing_metrics["grammar_conflict_generic_rate"] == gates["requiredGrammarConflictGenericRate"],
        "insufficient_requests_use_generic": routing_metrics["insufficient_generic_rate"] == gates["requiredInsufficientGenericRate"],
        "no_incorrect_specific_on_grammar_fallback_strata": routing_metrics["incorrect_specific_rate_on_grammar_fallback_strata"] == gates["requiredIncorrectSpecificRateOnGrammarFallbackStrata"],
        "grammar_router_cost": grammar_cost <= gates["maximumGrammarRouterMeanDecisionCost"] + 1e-12,
        "improves_over_no_query": routing_metrics["improvement_over_no_query"] >= gates["minimumImprovementOverNoQuery"] - 1e-12,
        "improves_over_always_generic": routing_metrics["improvement_over_always_generic"] >= gates["minimumImprovementOverAlwaysGeneric"] - 1e-12,
        "matches_information_oracle_cost": routing_metrics["cost_gap_from_information_oracle"] <= gates["maximumCostGapFromInformationOracle"] + 1e-12,
        "all_interactive_policies_final_exact": all(
            metrics[name]["final_exact_accuracy"] == 1.0 for name in COMPARATORS if name != "NO_QUERY"
        ),
        "all_irrelevant_intermediates_fail_closed": all(
            metrics[name]["irrelevant_intermediate_fail_closed_rate"] == 1.0 for name in COMPARATORS
        ),
        "authoritative_hypotheses_retained": all(
            metrics[name]["authoritative_hypothesis_retention"] == 1.0 for name in COMPARATORS
        ),
        "no_candidate_proposal_surface": all(
            not ({"candidate_state_ids", "state_ranking", "llm_proposal", "confidence"} & set(row))
            for row in public_requests + metadata
        ),
        "zero_evaluation_model_API_training_execution": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "routing_metrics": routing_metrics,
        "comparator_metrics": metrics,
        "routing_records": routing_records,
        "episode_count": len(episodes),
        "candidate_proposal_field_count": 0,
        "evaluation_policy_read_count": 0,
        "model_load_count": 0,
        "model_generation_or_score_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "actual_execution_count": 0,
    }


__all__ = [
    "COMPARATORS",
    "REQUEST_STAGES",
    "build_episodes",
    "choose_initial_query",
    "choose_lexical_query",
    "evaluate",
    "normalize_alias",
]
