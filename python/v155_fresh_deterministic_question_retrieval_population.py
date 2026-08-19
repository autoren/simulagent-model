from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v148_typed_witness_firewall import finalize_witness


STAGES = (
    "request_left_anchor", "request_left_paraphrase", "request_right_clear", "request_ambiguous",
    "closed_answer_left", "closed_answer_right",
)
REQUEST_STAGES = set(STAGES[:4])
ANSWER_STAGES = set(STAGES[4:])
CANDIDATE_FIELDS = {
    "candidate_state_ids", "state_ranking", "compatible_state_proposal", "llm_proposal",
    "confidence", "presented_candidate_choice_id",
}


def _complete_witness(partial: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    return {"evidence_status": "SUFFICIENT", "source": config["trustedSource"], **partial}


def build_catalog(config: dict[str, Any]) -> dict[str, Any]:
    queries = []
    for family in config["families"]:
        queries.append({
            "query_id": family["query_id"],
            "title": family["query_title"],
            "question": family["question"],
            "retrieval_profile": deepcopy(family["retrieval_profile"]),
            "options": [
                {
                    "option_id": "LEFT", "text": family["left_option_text"],
                    "state_id": family["left_choice_id"],
                    "witness": _complete_witness(family["left_witness"], config),
                },
                {
                    "option_id": "RIGHT", "text": family["right_option_text"],
                    "state_id": family["right_choice_id"],
                    "witness": _complete_witness(family["right_witness"], config),
                },
            ],
        })
    payload = {
        "schema_version": "155-explicit-metadata-question-retrieval-catalog",
        "choices": deepcopy(config["catalog"]["choices"]),
        "queries": queries,
    }
    payload["catalog_sha256"] = payload_hash(payload)
    return payload


def witness_from_answer_event(event: Any, catalog: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(event, dict) or set(event) != {"query_id", "selected_option_id"}:
        return None
    query = next((row for row in catalog["queries"] if row["query_id"] == event["query_id"]), None)
    if query is None:
        return None
    option = next(
        (row for row in query["options"] if row["option_id"] == event["selected_option_id"]), None
    )
    return deepcopy(option["witness"]) if option is not None else None


def malformed_answer_events(catalog: dict[str, Any]) -> list[Any]:
    query_id = catalog["queries"][0]["query_id"]
    return [
        None, {}, {"query_id": query_id}, {"selected_option_id": "LEFT"},
        {"query_id": "UNKNOWN", "selected_option_id": "LEFT"},
        {"query_id": query_id, "selected_option_id": "UNKNOWN"},
        {"query_id": query_id, "selected_option_id": "LEFT", "extra": True},
        [query_id, "LEFT"],
    ]


def _slot_variants(family: dict[str, Any]) -> list[dict[str, str]]:
    columns = family["slot_columns"]
    if {len(values) for values in columns.values()} != {8}:
        raise ValueError(f"family {family['family_id']} must have eight values in every slot column")
    names = list(columns)
    return [{name: columns[name][index] for name in names} for index in range(8)]


def _fixture_id(group_id: str, stage: str) -> str:
    return "v155-" + hashlib.sha256(f"{group_id}|{stage}".encode()).hexdigest()[:16]


def build_population(config: dict[str, Any]) -> dict[str, Any]:
    catalog = build_catalog(config)
    queries = {row["query_id"]: row for row in catalog["queries"]}
    public_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(config["families"]):
        query = queries[family["query_id"]]
        left = family["left_choice_id"]
        right = family["right_choice_id"]
        for variant_index, slots in enumerate(_slot_variants(family)):
            split = "development" if variant_index < 4 else "evaluation"
            group_id = f"v155-g{family_index:02d}-{variant_index:02d}"
            render = lambda key: family[key].format(**slots)
            ambiguous = render("ambiguous")
            specs = {
                "request_left_anchor": (render("left_anchor"), left, [left], None),
                "request_left_paraphrase": (render("left_paraphrase"), left, [left], None),
                "request_right_clear": (render("right_clear"), right, [right], None),
                "request_ambiguous": (ambiguous, config["insufficientId"], sorted([left, right]), None),
                "closed_answer_left": (
                    render("left_answer"), left, [left],
                    {"query_id": family["query_id"], "selected_option_id": "LEFT"},
                ),
                "closed_answer_right": (
                    render("right_answer"), right, [right],
                    {"query_id": family["query_id"], "selected_option_id": "RIGHT"},
                ),
            }
            for stage in STAGES:
                text, truth, compatible, event = specs[stage]
                conversation = (
                    [{"role": "user", "text": text}]
                    if stage in REQUEST_STAGES else [
                        {"role": "user", "text": ambiguous},
                        {"role": "assistant", "text": query["question"]},
                        {"role": "user", "text": text},
                    ]
                )
                fixture_id = _fixture_id(group_id, stage)
                public = {
                    "fixture_id": fixture_id, "split": split,
                    "conversation": conversation, "closed_answer_event": event,
                }
                witness = witness_from_answer_event(event, catalog)
                hidden = {
                    **public, "group_id": group_id, "family_id": family["family_id"],
                    "stage": stage, "truth_state_id": truth,
                    "compatible_state_ids": compatible, "oracle_query_id": family["query_id"],
                    "oracle_witness": witness, "trusted_witness_available": witness is not None,
                    "variant_index": variant_index,
                }
                public_rows.append(public)
                hidden_rows.append(hidden)
    public_rows.sort(key=lambda row: row["fixture_id"])
    hidden_rows.sort(key=lambda row: row["fixture_id"])
    summary = {
        "choice_count": len(catalog["choices"]), "query_count": len(catalog["queries"]),
        "family_count": len(config["families"]),
        "group_count": len({row["group_id"] for row in hidden_rows}),
        "fixture_count": len(hidden_rows),
        "request_fixture_count": sum(row["stage"] in REQUEST_STAGES for row in hidden_rows),
        "closed_answer_fixture_count": sum(row["stage"] in ANSWER_STAGES for row in hidden_rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in hidden_rows).items())),
        "stage_counts": dict(sorted(Counter(row["stage"] for row in hidden_rows).items())),
        "truth_counts": dict(sorted(Counter(row["truth_state_id"] for row in hidden_rows).items())),
    }
    return {
        "interaction_catalog": catalog, "public_fixtures": public_rows,
        "hidden_fixtures": hidden_rows, "population_summary": summary,
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
        "group_id", "family_id", "stage", "truth_state_id", "compatible_state_ids",
        "oracle_query_id", "oracle_witness", "trusted_witness_available", "variant_index",
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in hidden:
        groups.setdefault(row["group_id"], []).append(row)
    current_conversations = {
        json.dumps(row["conversation"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in public
    }
    prior_conversations = {
        json.dumps(row["conversation"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in prior_public_rows
    }
    answered = [row for row in hidden if row["stage"] in ANSWER_STAGES]
    preanswer = [row for row in hidden if row["stage"] in REQUEST_STAGES]
    answered_outputs = [finalize_witness(row["oracle_witness"], None, config) for row in answered]
    preanswer_outputs = [finalize_witness(None, None, config) for _ in preanswer]
    malformed_outputs = [
        finalize_witness(witness_from_answer_event(event, catalog), None, config)
        for event in malformed_answer_events(catalog)
    ]
    candidate_field_count = sum(len(CANDIDATE_FIELDS & set(row)) for row in public + hidden)
    kinds = Counter(row["kind"] for row in catalog["choices"])
    profiles = [row["retrieval_profile"] for row in catalog["queries"]]
    profile_contract = all(
        set(profile) == {"anchor_phrases", "primary_terms", "secondary_terms"}
        and all(
            isinstance(profile[key], list)
            and len(profile[key]) == len(set(profile[key]))
            and all(isinstance(value, str) and value.strip() and value == value.lower() for value in profile[key])
            for key in profile
        )
        for profile in profiles
    )
    checks = {
        "choice_query_family_and_profile_counts": bool(
            len(catalog["choices"]) == config["gates"]["requiredChoiceCount"]
            and kinds == Counter({"KNOWN": 6, "NOVEL_CANDIDATE": 1, "UNSUPPORTED": 1, "INSUFFICIENT_EVIDENCE": 1})
            and len(catalog["queries"]) == config["gates"]["requiredQueryCount"]
            and len({row["query_id"] for row in catalog["queries"]}) == len(catalog["queries"])
            and all(len(row["options"]) == 2 for row in catalog["queries"])
            and profile_contract
        ),
        "population_split_and_stage_counts": bool(
            summary["family_count"] == 6 and summary["group_count"] == 48
            and summary["fixture_count"] == 288 and summary["request_fixture_count"] == 192
            and summary["closed_answer_fixture_count"] == 96
            and summary["split_counts"] == {"development": 144, "evaluation": 144}
            and all(count == 48 for count in summary["stage_counts"].values())
        ),
        "group_stage_completeness": all(
            len(rows) == 6 and {row["stage"] for row in rows} == set(STAGES)
            and len({row["split"] for row in rows}) == 1 for rows in groups.values()
        ),
        "public_hidden_alignment_without_truth_leakage": bool(
            set(public_by_id) == set(hidden_by_id)
            and all(not (forbidden & set(row)) for row in public)
            and all(
                all(public_by_id[key][field] == hidden_by_id[key][field] for field in public_by_id[key])
                for key in public_by_id
            )
        ),
        "no_candidate_state_proposal_or_pruning_surface": candidate_field_count == 0,
        "compatibility_exact": all(
            (row["truth_state_id"] == config["insufficientId"] and len(row["compatible_state_ids"]) == 2)
            or (row["truth_state_id"] != config["insufficientId"] and row["compatible_state_ids"] == [row["truth_state_id"]])
            for row in hidden
        ),
        "closed_answer_events_only_on_answer_stages": all(
            (row["closed_answer_event"] is not None) == (row["stage"] in ANSWER_STAGES) for row in hidden
        ),
        "closed_answers_route_exactly": all(
            output["witness_valid"] and output["final_state_id"] == answered[index]["truth_state_id"]
            and output["llm_proposal_non_authoritative"] and not output["executable"]
            for index, output in enumerate(answered_outputs)
        ),
        "preanswer_and_malformed_events_fail_closed": bool(
            all(not output["witness_valid"] and output["final_state_id"] == config["insufficientId"] for output in preanswer_outputs)
            and all(not output["witness_valid"] and output["final_state_id"] == config["insufficientId"] for output in malformed_outputs)
        ),
        "fresh_exact_conversations": not (current_conversations & prior_conversations),
        "complete_hypothesis_retention_and_zero_execution": all(
            output["authoritative_hypothesis_ids_retained"] == config["outputIds"]
            and not output["authoritative_hypothesis_universe_pruned"]
            and output["actual_execution_count"] == 0
            for output in answered_outputs + preanswer_outputs + malformed_outputs
        ),
    }
    return {
        "passed": all(checks.values()), "checks": checks, "summary": summary,
        "candidate_proposal_field_count": candidate_field_count,
        "exact_prior_conversation_overlap_count": len(current_conversations & prior_conversations),
        "closed_answer_witness_routing": sum(
            output["final_state_id"] == answered[index]["truth_state_id"]
            for index, output in enumerate(answered_outputs)
        ) / len(answered_outputs),
        "preanswer_fail_closed_rate": sum(
            output["final_state_id"] == config["insufficientId"] for output in preanswer_outputs
        ) / len(preanswer_outputs),
        "malformed_answer_event_fail_closed_rate": sum(
            output["final_state_id"] == config["insufficientId"] for output in malformed_outputs
        ) / len(malformed_outputs),
        "model_load_count": 0, "model_generation_or_score_count": 0,
        "evaluation_policy_score_count": 0, "API_call_count": 0,
        "training_run_count": 0, "actual_execution_count": 0,
    }
