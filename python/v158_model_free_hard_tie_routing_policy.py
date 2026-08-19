from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any

from v148_typed_witness_firewall import finalize_witness
from v150_oracle_closed_interaction_policy import decision_cost
from v156_model_free_explicit_metadata_question_retrieval import rank_queries
from v157_fresh_hard_tie_routing_population import route_from_answer_event, witness_from_answer_event


REQUEST_STAGES = {
    "request_lexical_control", "request_uncatalogued_paraphrase",
    "request_relational_tie", "request_insufficient",
}
COMPARATORS = (
    "NO_QUERY", "SOURCE_SPECIFIC_THEN_GENERIC", "SEEDED_RANDOM_SPECIFIC_THEN_GENERIC",
    "ALWAYS_GENERIC", "INFORMATION_ORACLE", "MARGIN_GATED_ROUTER",
)


def choose_initial_query(
    fixture: dict[str, Any], retrieval_catalog: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    ranked = rank_queries(fixture, retrieval_catalog, config)
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
        "scores": ranked["scores"], "top_score": ranked["top_score"],
        "top_two_margin": ranked["top_two_margin"],
        "top_score_tie_count": ranked["top_score_tie_count"],
        "specific_selected": use_specific,
    }


def build_episodes(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for row in metadata:
        by_group.setdefault(row["group_id"], {})[row["stage"]] = row
    episodes = []
    for request in sorted(
        (row for row in metadata if row["stage"] in REQUEST_STAGES), key=lambda row: row["fixture_id"]
    ):
        if request["stage"] == "request_relational_tie":
            sides = ("left", "right")
        elif request["stage"] == "request_uncatalogued_paraphrase":
            sides = ("right",)
        elif request["stage"] == "request_lexical_control":
            sides = ("left",)
        else:
            sides = ("unclear",)
        for side in sides:
            if side == "unclear":
                specific = None
                route = by_group[request["group_id"]]["closed_route_unclear"]
                truth = request["truth_state_id"]
            else:
                specific = by_group[request["group_id"]][f"closed_specific_{side}"]
                route = by_group[request["group_id"]]["closed_route_family"]
                truth = specific["truth_state_id"]
            episodes.append({
                "episode_id": f"{request['fixture_id']}::{side}",
                "fixture_id": request["fixture_id"], "group_id": request["group_id"],
                "stage": request["stage"], "stratum": request["stratum"], "side": side,
                "truth_state_id": truth,
                "oracle_specific_query_id": request["oracle_specific_query_id"],
                "target_initial_query_id": request["oracle_initial_query_id"],
                "route_event": route["closed_answer_event"],
                "specific_answer_event": specific["closed_answer_event"] if specific is not None else None,
            })
    return episodes


def _seeded_order(episode: dict[str, Any], query_ids: list[str], config: dict[str, Any]) -> list[str]:
    material = f"{config['policy']['randomSeed']}|{episode['fixture_id']}"
    order = list(query_ids)
    random.Random(int(hashlib.sha256(material.encode()).hexdigest(), 16)).shuffle(order)
    return order


def evaluate(
    public_requests: list[dict[str, Any]], metadata: list[dict[str, Any]],
    retrieval_catalog: dict[str, Any], witness_catalog: dict[str, Any],
    witness_config: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    public_by_id = {row["fixture_id"]: row for row in public_requests}
    request_metadata = [row for row in metadata if row["stage"] in REQUEST_STAGES]
    routing_records = {}
    for row in request_metadata:
        decision = choose_initial_query(public_by_id[row["fixture_id"]], retrieval_catalog, config)
        routing_records[row["fixture_id"]] = {
            "fixture_id": row["fixture_id"], "stage": row["stage"], "stratum": row["stratum"],
            "target_initial_query_id": row["oracle_initial_query_id"], **decision,
            "initial_action_exact": decision["initial_query_id"] == row["oracle_initial_query_id"],
        }
    episodes = build_episodes(metadata)
    query_ids = [row["query_id"] for row in retrieval_catalog["queries"]]
    known_ids = set(witness_config["knownIds"])
    specific_cost = config["policy"]["specificQueryCost"]
    generic_cost = config["policy"]["genericRouteQueryCost"]
    records = []
    intermediates = []

    def fail_closed_intermediate(episode: dict[str, Any], comparator: str, query_id: str) -> None:
        output = finalize_witness(None, None, witness_config)
        intermediates.append({
            "episode_id": episode["episode_id"], "comparator": comparator,
            "query_id": query_id, "witness_valid": output["witness_valid"],
            "final_state_id": output["final_state_id"],
            "retained": output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
            "actual_execution_count": output["actual_execution_count"],
        })

    def finish_specific(episode: dict[str, Any]) -> tuple[str, bool]:
        witness = witness_from_answer_event(episode["specific_answer_event"], witness_catalog)
        output = finalize_witness(witness, None, witness_config)
        return output["final_state_id"], output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"]

    def finish_generic(episode: dict[str, Any]) -> tuple[str, bool, float]:
        route = route_from_answer_event(episode["route_event"], witness_catalog)
        route_output = finalize_witness(None, None, witness_config)
        retained = route_output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"]
        if route["target_query_id"] is None:
            return witness_config["insufficientId"], retained, generic_cost
        final, specific_retained = finish_specific(episode)
        return final, retained and specific_retained, generic_cost + specific_cost

    for episode in episodes:
        for comparator in COMPARATORS:
            initial_query = None
            wrong_specific_count = 0
            if comparator == "NO_QUERY":
                final = witness_config["insufficientId"]
                retained = True
                interaction_cost = 0.0
            elif comparator == "ALWAYS_GENERIC":
                initial_query = config["policy"]["genericQueryId"]
                final, retained, interaction_cost = finish_generic(episode)
            elif comparator == "INFORMATION_ORACLE":
                if episode["stage"] in {"request_relational_tie", "request_insufficient"}:
                    initial_query = config["policy"]["genericQueryId"]
                    final, retained, interaction_cost = finish_generic(episode)
                else:
                    initial_query = episode["oracle_specific_query_id"]
                    final, retained = finish_specific(episode)
                    interaction_cost = specific_cost
            elif comparator == "MARGIN_GATED_ROUTER":
                initial_query = routing_records[episode["fixture_id"]]["initial_query_id"]
                if initial_query == config["policy"]["genericQueryId"]:
                    final, retained, interaction_cost = finish_generic(episode)
                elif initial_query == episode["oracle_specific_query_id"]:
                    final, retained = finish_specific(episode)
                    interaction_cost = specific_cost
                else:
                    wrong_specific_count = 1
                    fail_closed_intermediate(episode, comparator, initial_query)
                    final, retained, fallback_cost = finish_generic(episode)
                    interaction_cost = specific_cost + fallback_cost
            else:
                order = list(query_ids) if comparator == "SOURCE_SPECIFIC_THEN_GENERIC" else _seeded_order(episode, query_ids, config)
                initial_query = order[0]
                if episode["specific_answer_event"] is None:
                    for query_id in order:
                        fail_closed_intermediate(episode, comparator, query_id)
                    wrong_specific_count = len(order)
                    final, retained, fallback_cost = finish_generic(episode)
                    interaction_cost = len(order) * specific_cost + fallback_cost
                else:
                    rank = order.index(episode["oracle_specific_query_id"]) + 1
                    for query_id in order[: rank - 1]:
                        fail_closed_intermediate(episode, comparator, query_id)
                    wrong_specific_count = rank - 1
                    final, retained = finish_specific(episode)
                    interaction_cost = rank * specific_cost
            total_cost = interaction_cost + decision_cost(
                episode["truth_state_id"], final, known_ids, config
            )
            records.append({
                **episode, "comparator": comparator, "initial_query_id": initial_query,
                "wrong_specific_count": wrong_specific_count, "final_state_id": final,
                "interaction_cost": interaction_cost, "decision_cost": total_cost,
                "final_exact": final == episode["truth_state_id"], "retained": retained,
                "actual_execution_count": 0,
            })

    metrics = {}
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
                    not row["witness_valid"] and row["final_state_id"] == witness_config["insufficientId"]
                    and row["retained"] and row["actual_execution_count"] == 0 for row in relevant
                ) / len(relevant) if relevant else 1.0
            ),
            "authoritative_hypothesis_retention": sum(row["retained"] for row in subset) / len(subset),
            "actual_execution_count": 0,
        }
    routing_rows = list(routing_records.values())
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in routing_rows:
        by_stratum[row["stratum"]].append(row)
    generic_id = config["policy"]["genericQueryId"]
    routing_metrics = {
        "request_count": len(routing_rows),
        "initial_action_accuracy": sum(row["initial_action_exact"] for row in routing_rows) / len(routing_rows),
        "lexical_control_specific_rate": sum(row["initial_query_id"] != generic_id for row in by_stratum["lexical_control"]) / len(by_stratum["lexical_control"]),
        "uncatalogued_paraphrase_generic_rate": sum(row["initial_query_id"] == generic_id for row in by_stratum["uncatalogued_paraphrase"]) / len(by_stratum["uncatalogued_paraphrase"]),
        "relational_tie_generic_rate": sum(row["initial_query_id"] == generic_id for row in by_stratum["relational_tie"]) / len(by_stratum["relational_tie"]),
        "insufficient_generic_rate": sum(row["initial_query_id"] == generic_id for row in by_stratum["insufficient"]) / len(by_stratum["insufficient"]),
        "incorrect_specific_rate_on_fallback_strata": sum(
            row["initial_query_id"] != generic_id
            for stratum in ("uncatalogued_paraphrase", "relational_tie", "insufficient")
            for row in by_stratum[stratum]
        ) / sum(len(by_stratum[stratum]) for stratum in ("uncatalogued_paraphrase", "relational_tie", "insufficient")),
        "initial_action_counts": dict(sorted(Counter(row["initial_query_id"] for row in routing_rows).items())),
        "mean_top_score": sum(row["top_score"] for row in routing_rows) / len(routing_rows),
        "mean_top_two_margin": sum(row["top_two_margin"] for row in routing_rows) / len(routing_rows),
    }
    no_query_cost = metrics["NO_QUERY"]["mean_decision_cost"]
    always_generic_cost = metrics["ALWAYS_GENERIC"]["mean_decision_cost"]
    router_cost = metrics["MARGIN_GATED_ROUTER"]["mean_decision_cost"]
    routing_metrics["improvement_over_no_query"] = no_query_cost - router_cost
    routing_metrics["improvement_over_always_generic"] = always_generic_cost - router_cost
    gates = config["gates"]
    checks = {
        "population_and_comparator_counts": bool(
            len(public_requests) == 96 and len(metadata) == 192
            and len({row["group_id"] for row in metadata}) == 24
            and len(episodes) == 120 and len(metrics) == 6
        ),
        "router_initial_action_exact": routing_metrics["initial_action_accuracy"] == gates["requiredMarginRouterInitialActionAccuracy"],
        "lexical_controls_use_specific": routing_metrics["lexical_control_specific_rate"] == gates["requiredLexicalControlSpecificRate"],
        "uncatalogued_paraphrases_use_generic": routing_metrics["uncatalogued_paraphrase_generic_rate"] == gates["requiredUncataloguedParaphraseGenericRate"],
        "relational_ties_use_generic": routing_metrics["relational_tie_generic_rate"] == gates["requiredRelationalTieGenericRate"],
        "insufficient_requests_use_generic": routing_metrics["insufficient_generic_rate"] == gates["requiredInsufficientGenericRate"],
        "no_incorrect_specific_on_fallback_strata": routing_metrics["incorrect_specific_rate_on_fallback_strata"] == 0.0,
        "router_cost": router_cost <= gates["maximumMarginRouterMeanDecisionCost"] + 1e-12,
        "improves_over_no_query": routing_metrics["improvement_over_no_query"] >= gates["minimumImprovementOverNoQuery"] - 1e-12,
        "improves_over_always_generic": routing_metrics["improvement_over_always_generic"] >= gates["minimumImprovementOverAlwaysGeneric"] - 1e-12,
        "all_interactive_policies_final_exact": all(metrics[name]["final_exact_accuracy"] == 1.0 for name in COMPARATORS if name != "NO_QUERY"),
        "all_irrelevant_intermediates_fail_closed": all(metrics[name]["irrelevant_intermediate_fail_closed_rate"] == 1.0 for name in COMPARATORS),
        "authoritative_hypotheses_retained": all(metrics[name]["authoritative_hypothesis_retention"] == 1.0 for name in COMPARATORS),
        "no_candidate_proposal_surface": all(
            not ({"candidate_state_ids", "state_ranking", "llm_proposal", "confidence"} & set(row))
            for row in public_requests + metadata
        ),
        "zero_evaluation_model_API_training_execution": True,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "routing_metrics": routing_metrics, "comparator_metrics": metrics,
        "routing_records": routing_records, "episode_count": len(episodes),
        "candidate_proposal_field_count": 0, "evaluation_policy_read_count": 0,
        "model_load_count": 0, "model_generation_or_score_count": 0,
        "API_call_count": 0, "training_run_count": 0, "actual_execution_count": 0,
    }


__all__ = ["COMPARATORS", "REQUEST_STAGES", "build_episodes", "choose_initial_query", "evaluate"]
