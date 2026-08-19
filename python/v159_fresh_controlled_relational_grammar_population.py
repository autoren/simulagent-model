from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v148_typed_witness_firewall import finalize_witness


STAGES = (
    "request_lexical_control",
    "request_grammar_unique",
    "request_grammar_conflict",
    "request_insufficient",
    "closed_route_family",
    "closed_route_unclear",
    "closed_specific_left",
    "closed_specific_right",
)
REQUEST_STAGES = set(STAGES[:4])
ROUTE_STAGES = set(STAGES[4:6])
SPECIFIC_ANSWER_STAGES = set(STAGES[6:])
CANDIDATE_FIELDS = {
    "candidate_state_ids",
    "state_ranking",
    "compatible_state_proposal",
    "llm_proposal",
    "confidence",
    "presented_candidate_choice_id",
}


def _complete_witness(partial: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    return {"evidence_status": "SUFFICIENT", "source": config["trustedSource"], **partial}


def build_catalog(config: dict[str, Any]) -> dict[str, Any]:
    generic = config["genericQuery"]
    generic_options = [
        {
            "option_id": f"ROUTE_{family['query_id']}",
            "text": family["route_label"],
            "route_query_id": family["query_id"],
        }
        for family in config["families"]
    ] + [
        {
            "option_id": config["genericUnclearOptionId"],
            "text": generic["unclear_option_text"],
            "route_query_id": None,
        }
    ]
    queries = [
        {
            "query_id": generic["query_id"],
            "title": generic["title"],
            "question": generic["question"],
            "query_kind": "GENERIC_ROUTE",
            "options": generic_options,
        }
    ]
    for family in config["families"]:
        queries.append(
            {
                "query_id": family["query_id"],
                "title": family["query_title"],
                "question": family["question"],
                "query_kind": "SPECIFIC_WITNESS",
                "grammar_aliases": list(family["grammar_aliases"]),
                "retrieval_profile": deepcopy(family["retrieval_profile"]),
                "options": [
                    {
                        "option_id": "LEFT",
                        "text": family["left_option_text"],
                        "state_id": family["left_choice_id"],
                        "witness": _complete_witness(family["left_witness"], config),
                    },
                    {
                        "option_id": "RIGHT",
                        "text": family["right_option_text"],
                        "state_id": family["right_choice_id"],
                        "witness": _complete_witness(family["right_witness"], config),
                    },
                ],
            }
        )
    payload = {
        "schema_version": "159-controlled-relational-grammar-interaction-catalog",
        "choices": deepcopy(config["catalog"]["choices"]),
        "queries": queries,
    }
    payload["catalog_sha256"] = payload_hash(payload)
    return payload


def _find_option(
    event: Any, catalog: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(event, dict) or set(event) != {"query_id", "selected_option_id"}:
        return None, None
    query = next((row for row in catalog["queries"] if row["query_id"] == event["query_id"]), None)
    if query is None:
        return None, None
    option = next(
        (row for row in query["options"] if row["option_id"] == event["selected_option_id"]),
        None,
    )
    return query, option


def witness_from_answer_event(event: Any, catalog: dict[str, Any]) -> dict[str, Any] | None:
    query, option = _find_option(event, catalog)
    if query is None or option is None or query["query_kind"] != "SPECIFIC_WITNESS":
        return None
    return deepcopy(option["witness"])


def route_from_answer_event(event: Any, catalog: dict[str, Any]) -> dict[str, Any]:
    query, option = _find_option(event, catalog)
    if query is None or option is None or query["query_kind"] != "GENERIC_ROUTE":
        return {"route_valid": False, "target_query_id": None}
    return {"route_valid": True, "target_query_id": option["route_query_id"]}


def malformed_answer_events(catalog: dict[str, Any]) -> list[Any]:
    generic = next(row for row in catalog["queries"] if row["query_kind"] == "GENERIC_ROUTE")
    specific = next(row for row in catalog["queries"] if row["query_kind"] == "SPECIFIC_WITNESS")
    return [
        None,
        {},
        {"query_id": generic["query_id"]},
        {"selected_option_id": generic["options"][0]["option_id"]},
        {"query_id": "UNKNOWN", "selected_option_id": generic["options"][0]["option_id"]},
        {"query_id": generic["query_id"], "selected_option_id": "UNKNOWN"},
        {
            "query_id": generic["query_id"],
            "selected_option_id": generic["options"][0]["option_id"],
            "extra": True,
        },
        [generic["query_id"], generic["options"][0]["option_id"]],
        {"query_id": specific["query_id"], "selected_option_id": "UNKNOWN"},
    ]


def _slot_variants(family: dict[str, Any]) -> list[dict[str, str]]:
    columns = family["slot_columns"]
    if {len(values) for values in columns.values()} != {8}:
        raise ValueError(f"family {family['family_id']} must have eight values in each slot column")
    names = list(columns)
    return [{name: columns[name][index] for name in names} for index in range(8)]


def _fixture_id(group_id: str, stage: str) -> str:
    return "v159-" + hashlib.sha256(f"{group_id}|{stage}".encode()).hexdigest()[:16]


def build_population(config: dict[str, Any]) -> dict[str, Any]:
    catalog = build_catalog(config)
    query_by_id = {row["query_id"]: row for row in catalog["queries"]}
    family_by_query = {row["query_id"]: row for row in config["families"]}
    public_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(config["families"]):
        specific_query = query_by_id[family["query_id"]]
        decoy = family_by_query[family["decoy_query_id"]]
        left_alias, right_alias = family["grammar_aliases"]
        decoy_alias = decoy["grammar_aliases"][0]
        left, right = family["left_choice_id"], family["right_choice_id"]
        for variant_index, slots in enumerate(_slot_variants(family)):
            split = "development" if variant_index < 4 else "evaluation"
            group_id = f"v159-g{family_index:02d}-{variant_index:02d}"
            grammar_index = variant_index % 4
            lexical = family["lexical_control"].format(**slots)
            unique = config["grammar"]["uniqueTemplates"][grammar_index].format(
                **slots, right_alias=right_alias
            )
            conflict = config["grammar"]["conflictTemplates"][grammar_index].format(
                **slots, left_alias=left_alias, decoy_alias=decoy_alias
            )
            insufficient = config["grammar"]["insufficientTemplates"][grammar_index].format(**slots)
            specs = {
                "request_lexical_control": (
                    lexical,
                    left,
                    [left],
                    None,
                    "lexical_control",
                    family["query_id"],
                    family["query_id"],
                    [],
                ),
                "request_grammar_unique": (
                    unique,
                    right,
                    [right],
                    None,
                    "grammar_unique",
                    family["query_id"],
                    family["query_id"],
                    [right_alias],
                ),
                "request_grammar_conflict": (
                    conflict,
                    config["insufficientId"],
                    list(config["outputIds"]),
                    None,
                    "grammar_conflict",
                    None,
                    config["genericQueryId"],
                    [left_alias, decoy_alias],
                ),
                "request_insufficient": (
                    insufficient,
                    config["insufficientId"],
                    list(config["outputIds"]),
                    None,
                    "insufficient",
                    None,
                    config["genericQueryId"],
                    [],
                ),
                "closed_route_family": (
                    family["route_answer"],
                    config["insufficientId"],
                    sorted([left, right]),
                    {"query_id": config["genericQueryId"], "selected_option_id": f"ROUTE_{family['query_id']}"},
                    "trusted_route_family",
                    family["query_id"],
                    None,
                    [],
                ),
                "closed_route_unclear": (
                    "No single registered relation is established.",
                    config["insufficientId"],
                    list(config["outputIds"]),
                    {"query_id": config["genericQueryId"], "selected_option_id": config["genericUnclearOptionId"]},
                    "trusted_route_unclear",
                    None,
                    None,
                    [],
                ),
                "closed_specific_left": (
                    family["left_answer"],
                    left,
                    [left],
                    {"query_id": family["query_id"], "selected_option_id": "LEFT"},
                    "trusted_specific_left",
                    family["query_id"],
                    None,
                    [],
                ),
                "closed_specific_right": (
                    family["right_answer"],
                    right,
                    [right],
                    {"query_id": family["query_id"], "selected_option_id": "RIGHT"},
                    "trusted_specific_right",
                    family["query_id"],
                    None,
                    [],
                ),
            }
            for stage in STAGES:
                text, truth, compatible, event, stratum, route_target, oracle_initial, relation_aliases = specs[stage]
                if stage in REQUEST_STAGES:
                    conversation = [{"role": "user", "text": text}]
                elif stage == "closed_route_family":
                    conversation = [
                        {"role": "user", "text": unique},
                        {"role": "assistant", "text": query_by_id[config["genericQueryId"]]["question"]},
                        {"role": "user", "text": text},
                    ]
                elif stage == "closed_route_unclear":
                    conversation = [
                        {"role": "user", "text": conflict},
                        {"role": "assistant", "text": query_by_id[config["genericQueryId"]]["question"]},
                        {"role": "user", "text": text},
                    ]
                else:
                    conversation = [
                        {"role": "user", "text": unique},
                        {"role": "assistant", "text": specific_query["question"]},
                        {"role": "user", "text": text},
                    ]
                fixture_id = _fixture_id(group_id, stage)
                public = {
                    "fixture_id": fixture_id,
                    "split": split,
                    "conversation": conversation,
                    "closed_answer_event": event,
                }
                route = route_from_answer_event(event, catalog)
                witness = witness_from_answer_event(event, catalog)
                grammar_query_ids = sorted(
                    {
                        row["query_id"]
                        for row in catalog["queries"]
                        if row["query_kind"] == "SPECIFIC_WITNESS"
                        and set(relation_aliases) & set(row["grammar_aliases"])
                    }
                )
                hidden = {
                    **public,
                    "group_id": group_id,
                    "family_id": family["family_id"],
                    "stage": stage,
                    "stratum": stratum,
                    "truth_state_id": truth,
                    "compatible_state_ids": compatible,
                    "oracle_specific_query_id": family["query_id"],
                    "oracle_initial_query_id": oracle_initial,
                    "route_target_query_id": route_target,
                    "oracle_route": route if route["route_valid"] else None,
                    "oracle_witness": witness,
                    "grammar_relation_aliases": relation_aliases,
                    "grammar_query_ids": grammar_query_ids,
                    "variant_index": variant_index,
                }
                public_rows.append(public)
                hidden_rows.append(hidden)
    public_rows.sort(key=lambda row: row["fixture_id"])
    hidden_rows.sort(key=lambda row: row["fixture_id"])
    summary = {
        "choice_count": len(catalog["choices"]),
        "query_count": len(catalog["queries"]),
        "specific_query_count": sum(row["query_kind"] == "SPECIFIC_WITNESS" for row in catalog["queries"]),
        "grammar_alias_count": sum(len(row.get("grammar_aliases", [])) for row in catalog["queries"]),
        "family_count": len(config["families"]),
        "group_count": len({row["group_id"] for row in hidden_rows}),
        "fixture_count": len(hidden_rows),
        "request_fixture_count": sum(row["stage"] in REQUEST_STAGES for row in hidden_rows),
        "route_fixture_count": sum(row["stage"] in ROUTE_STAGES for row in hidden_rows),
        "specific_answer_fixture_count": sum(row["stage"] in SPECIFIC_ANSWER_STAGES for row in hidden_rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in hidden_rows).items())),
        "stage_counts": dict(sorted(Counter(row["stage"] for row in hidden_rows).items())),
        "stratum_counts": dict(sorted(Counter(row["stratum"] for row in hidden_rows).items())),
    }
    return {
        "interaction_catalog": catalog,
        "public_fixtures": public_rows,
        "hidden_fixtures": hidden_rows,
        "population_summary": summary,
    }


def audit_population(
    population: dict[str, Any], config: dict[str, Any], prior_public_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    catalog = population["interaction_catalog"]
    public = population["public_fixtures"]
    hidden = population["hidden_fixtures"]
    summary = population["population_summary"]
    public_by_id = {row["fixture_id"]: row for row in public}
    hidden_by_id = {row["fixture_id"]: row for row in hidden}
    forbidden = {
        "group_id",
        "family_id",
        "stage",
        "stratum",
        "truth_state_id",
        "compatible_state_ids",
        "oracle_specific_query_id",
        "oracle_initial_query_id",
        "route_target_query_id",
        "oracle_route",
        "oracle_witness",
        "grammar_relation_aliases",
        "grammar_query_ids",
        "variant_index",
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in hidden:
        groups.setdefault(row["group_id"], []).append(row)
    current = {
        json.dumps(row["conversation"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in public
    }
    prior = {
        json.dumps(row["conversation"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in prior_public_rows
    }
    requests = [row for row in hidden if row["stage"] in REQUEST_STAGES]
    routes = [row for row in hidden if row["stage"] in ROUTE_STAGES]
    specifics = [row for row in hidden if row["stage"] in SPECIFIC_ANSWER_STAGES]
    request_outputs = [finalize_witness(None, None, config) for _ in requests]
    route_outputs = [
        finalize_witness(witness_from_answer_event(row["closed_answer_event"], catalog), None, config)
        for row in routes
    ]
    specific_outputs = [
        finalize_witness(witness_from_answer_event(row["closed_answer_event"], catalog), None, config)
        for row in specifics
    ]
    malformed = malformed_answer_events(catalog)
    malformed_outputs = [
        finalize_witness(witness_from_answer_event(event, catalog), None, config) for event in malformed
    ]
    malformed_routes = [route_from_answer_event(event, catalog) for event in malformed]
    route_records = [route_from_answer_event(row["closed_answer_event"], catalog) for row in routes]
    generic = next(row for row in catalog["queries"] if row["query_kind"] == "GENERIC_ROUTE")
    specific_queries = [row for row in catalog["queries"] if row["query_kind"] == "SPECIFIC_WITNESS"]
    aliases = [alias for row in specific_queries for alias in row["grammar_aliases"]]
    conflicts = [row for row in requests if row["stage"] == "request_grammar_conflict"]
    candidate_field_count = sum(len(CANDIDATE_FIELDS & set(row)) for row in public + hidden)
    kinds = Counter(row["kind"] for row in catalog["choices"])
    expected = config["gates"]
    checks = {
        "choice_query_and_generic_route_contract": bool(
            len(catalog["choices"]) == expected["requiredChoiceCount"]
            and kinds == Counter({"KNOWN": 4, "NOVEL_CANDIDATE": 1, "UNSUPPORTED": 1, "INSUFFICIENT_EVIDENCE": 1})
            and len(catalog["queries"]) == expected["requiredQueryCountIncludingGeneric"]
            and len(specific_queries) == expected["requiredSpecificQueryCount"]
            and generic["query_id"] == config["genericQueryId"]
            and len(generic["options"]) == expected["requiredSpecificQueryCount"] + 1
            and all("state_id" not in option and "witness" not in option for option in generic["options"])
            and all(len(query["options"]) == 2 for query in specific_queries)
        ),
        "grammar_registry_unique_and_state_free": bool(
            len(aliases) == len(set(aliases)) == 8
            and all(alias == alias.lower() and alias.strip() == alias for alias in aliases)
            and all(
                not ({"state_id", "choice_id", "witness", "truth_state_id"} & set(query))
                for query in specific_queries
            )
        ),
        "population_split_stage_and_stratum_counts": bool(
            summary["group_count"] == expected["requiredGroupCount"]
            and summary["fixture_count"] == expected["requiredFixtureCount"]
            and summary["request_fixture_count"] == expected["requiredRequestFixtureCount"]
            and summary["route_fixture_count"] == expected["requiredRouteFixtureCount"]
            and summary["specific_answer_fixture_count"] == expected["requiredSpecificAnswerFixtureCount"]
            and summary["split_counts"] == {"development": 128, "evaluation": 128}
            and all(count == 32 for count in summary["stage_counts"].values())
            and all(count == 32 for count in summary["stratum_counts"].values())
        ),
        "group_stage_completeness": all(
            len(rows) == 8
            and {row["stage"] for row in rows} == set(STAGES)
            and len({row["split"] for row in rows}) == 1
            for rows in groups.values()
        ),
        "public_hidden_alignment_without_truth_grammar_route_or_oracle_leakage": bool(
            set(public_by_id) == set(hidden_by_id)
            and all(not (forbidden & set(row)) for row in public)
            and all(
                all(public_by_id[key][field] == hidden_by_id[key][field] for field in public_by_id[key])
                for key in public_by_id
            )
        ),
        "conflicts_name_two_distinct_registered_queries": all(
            len(row["grammar_relation_aliases"]) == 2
            and len(row["grammar_query_ids"]) == 2
            and row["oracle_initial_query_id"] == config["genericQueryId"]
            for row in conflicts
        ),
        "generic_route_events_valid_without_semantic_witness": bool(
            all(route["route_valid"] for route in route_records)
            and all(witness_from_answer_event(row["closed_answer_event"], catalog) is None for row in routes)
            and all(not output["witness_valid"] and output["final_state_id"] == config["insufficientId"] for output in route_outputs)
        ),
        "specific_answers_route_exactly": all(
            output["witness_valid"] and output["final_state_id"] == specifics[index]["truth_state_id"]
            for index, output in enumerate(specific_outputs)
        ),
        "requests_routes_and_malformed_events_fail_closed": bool(
            all(not output["witness_valid"] and output["final_state_id"] == config["insufficientId"] for output in request_outputs + route_outputs + malformed_outputs)
            and all(not route["route_valid"] for route in malformed_routes)
        ),
        "no_candidate_state_proposal_or_pruning_surface": candidate_field_count == 0,
        "fresh_exact_conversations": not (current & prior),
        "complete_hypothesis_retention_and_zero_execution": all(
            output["authoritative_hypothesis_ids_retained"] == config["outputIds"]
            and not output["authoritative_hypothesis_universe_pruned"]
            and output["actual_execution_count"] == 0
            for output in request_outputs + route_outputs + specific_outputs + malformed_outputs
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "grammar_alias_uniqueness": len(set(aliases)) / len(aliases),
        "conflict_distinct_query_rate": sum(len(row["grammar_query_ids"]) == 2 for row in conflicts) / len(conflicts),
        "candidate_proposal_field_count": candidate_field_count,
        "exact_prior_conversation_overlap_count": len(current & prior),
        "specific_answer_witness_routing": sum(
            output["final_state_id"] == specifics[index]["truth_state_id"]
            for index, output in enumerate(specific_outputs)
        ) / len(specific_outputs),
        "generic_route_validity": sum(route["route_valid"] for route in route_records) / len(route_records),
        "generic_route_semantic_witness_count": sum(
            witness_from_answer_event(row["closed_answer_event"], catalog) is not None for row in routes
        ),
        "preanswer_and_generic_final_fail_closed_rate": sum(
            output["final_state_id"] == config["insufficientId"] for output in request_outputs + route_outputs
        ) / (len(request_outputs) + len(route_outputs)),
        "malformed_event_fail_closed_rate": sum(
            output["final_state_id"] == config["insufficientId"] and not route["route_valid"]
            for output, route in zip(malformed_outputs, malformed_routes)
        ) / len(malformed_outputs),
        "policy_score_count": 0,
        "model_load_count": 0,
        "model_generation_or_score_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "actual_execution_count": 0,
    }


__all__ = [
    "REQUEST_STAGES",
    "STAGES",
    "audit_population",
    "build_catalog",
    "build_population",
    "malformed_answer_events",
    "route_from_answer_event",
    "witness_from_answer_event",
]
