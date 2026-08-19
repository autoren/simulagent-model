from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v148_typed_witness_firewall import finalize_witness


STAGES = (
    "request_known_familiar",
    "request_known_unfamiliar",
    "request_right",
    "request_ambiguous",
    "closed_answer_known",
    "closed_answer_right",
)
REQUEST_STAGES = set(STAGES[:4])
ANSWER_STAGES = set(STAGES[4:])
CANDIDATE_PROPOSAL_FIELDS = {
    "presented_candidate_choice_id",
    "candidate_state_ids",
    "compatible_state_proposal",
    "llm_proposal",
    "state_ranking",
}


def _complete_witness(partial: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    return {"evidence_status": "SUFFICIENT", "source": config["trustedSource"], **partial}


def build_catalog(config: dict[str, Any]) -> dict[str, Any]:
    queries = []
    for family in config["families"]:
        queries.append(
            {
                "query_id": family["query_id"],
                "question": family["question"],
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
        "schema_version": "152-question-order-only-interaction-catalog",
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
        (row for row in query["options"] if row["option_id"] == event["selected_option_id"]),
        None,
    )
    return deepcopy(option["witness"]) if option else None


def malformed_answer_events(catalog: dict[str, Any]) -> list[Any]:
    query_id = catalog["queries"][0]["query_id"]
    return [
        None,
        {},
        {"query_id": query_id},
        {"selected_option_id": "LEFT"},
        {"query_id": "UNKNOWN", "selected_option_id": "LEFT"},
        {"query_id": query_id, "selected_option_id": "UNKNOWN"},
        {"query_id": query_id, "selected_option_id": "LEFT", "extra": True},
        [query_id, "LEFT"],
    ]


def _slot_variants(family: dict[str, Any]) -> list[dict[str, str]]:
    columns = family["slot_columns"]
    lengths = {len(values) for values in columns.values()}
    if lengths != {8}:
        raise ValueError(f"family {family['family_id']} must have eight values in every slot column")
    names = list(columns)
    return [{name: columns[name][index] for name in names} for index in range(8)]


def _render(template: str, slots: dict[str, str]) -> str:
    return template.format(**slots)


def _fixture_id(group_id: str, stage: str) -> str:
    return "v152-" + hashlib.sha256(f"{group_id}|{stage}".encode()).hexdigest()[:16]


def build_population(config: dict[str, Any]) -> dict[str, Any]:
    catalog = build_catalog(config)
    queries = {row["query_id"]: row for row in catalog["queries"]}
    choice_kinds = {row["choice_id"]: row["kind"] for row in catalog["choices"]}
    public_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(config["families"]):
        left = family["left_choice_id"]
        right = family["right_choice_id"]
        compatible_pair = sorted([left, right])
        query = queries[family["query_id"]]
        for variant_index, slots in enumerate(_slot_variants(family)):
            split = "development" if variant_index < 4 else "evaluation"
            group_id = f"v152-g{family_index:02d}-{variant_index:02d}"
            ambiguous = _render(family["ambiguous"], slots)
            right_class = "novel_candidate" if choice_kinds[right] == "NOVEL_CANDIDATE" else "unsupported"
            specs = {
                "request_known_familiar": (
                    _render(family["known_familiar"], slots), left, [left], "known_familiar", None
                ),
                "request_known_unfamiliar": (
                    _render(family["known_unfamiliar"], slots), left, [left], "known_unfamiliar", None
                ),
                "request_right": (
                    _render(family["right_clear"], slots), right, [right], right_class, None
                ),
                "request_ambiguous": (
                    ambiguous, config["insufficientId"], compatible_pair, "insufficient_evidence", None
                ),
                "closed_answer_known": (
                    _render(family["left_answer"], slots), left, [left], "known_closed_answer",
                    {"query_id": family["query_id"], "selected_option_id": "LEFT"},
                ),
                "closed_answer_right": (
                    _render(family["right_answer"], slots), right, [right], f"{right_class}_closed_answer",
                    {"query_id": family["query_id"], "selected_option_id": "RIGHT"},
                ),
            }
            for stage in STAGES:
                text, truth, compatible, language_class, event = specs[stage]
                if stage in ANSWER_STAGES:
                    conversation = [
                        {"role": "user", "text": ambiguous},
                        {"role": "assistant", "text": query["question"]},
                        {"role": "user", "text": text},
                    ]
                else:
                    conversation = [{"role": "user", "text": text}]
                fixture_id = _fixture_id(group_id, stage)
                public = {
                    "fixture_id": fixture_id,
                    "split": split,
                    "conversation": conversation,
                    "closed_answer_event": event,
                }
                witness = witness_from_answer_event(event, catalog)
                hidden = {
                    **public,
                    "group_id": group_id,
                    "family_id": family["family_id"],
                    "stage": stage,
                    "language_class": language_class,
                    "truth_state_id": truth,
                    "compatible_state_ids": compatible,
                    "oracle_query_id": family["query_id"],
                    "oracle_witness": witness,
                    "trusted_witness_available": witness is not None,
                    "variant_index": variant_index,
                }
                public_rows.append(public)
                hidden_rows.append(hidden)
    public_rows.sort(key=lambda row: row["fixture_id"])
    hidden_rows.sort(key=lambda row: row["fixture_id"])
    summary = {
        "choice_count": len(catalog["choices"]),
        "query_count": len(catalog["queries"]),
        "family_count": len(config["families"]),
        "group_count": len({row["group_id"] for row in hidden_rows}),
        "fixture_count": len(hidden_rows),
        "request_fixture_count": sum(row["stage"] in REQUEST_STAGES for row in hidden_rows),
        "closed_answer_fixture_count": sum(row["stage"] in ANSWER_STAGES for row in hidden_rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in hidden_rows).items())),
        "stage_counts": dict(sorted(Counter(row["stage"] for row in hidden_rows).items())),
        "language_class_counts": dict(sorted(Counter(row["language_class"] for row in hidden_rows).items())),
        "truth_counts": dict(sorted(Counter(row["truth_state_id"] for row in hidden_rows).items())),
    }
    return {
        "interaction_catalog": catalog,
        "public_fixtures": public_rows,
        "hidden_fixtures": hidden_rows,
        "population_summary": summary,
    }


def audit_population(
    population: dict[str, Any],
    config: dict[str, Any],
    prior_public_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog = population["interaction_catalog"]
    public = population["public_fixtures"]
    hidden = population["hidden_fixtures"]
    summary = population["population_summary"]
    public_by_id = {row["fixture_id"]: row for row in public}
    hidden_by_id = {row["fixture_id"]: row for row in hidden}
    forbidden = {
        "group_id", "family_id", "stage", "language_class", "truth_state_id",
        "compatible_state_ids", "oracle_query_id", "oracle_witness",
        "trusted_witness_available", "variant_index",
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
    preanswer_outputs = [finalize_witness(None, None, config) for row in preanswer]
    malformed_outputs = [
        finalize_witness(witness_from_answer_event(event, catalog), None, config)
        for event in malformed_answer_events(catalog)
    ]
    option_outputs = []
    for query in catalog["queries"]:
        for option in query["options"]:
            option_outputs.append((option, finalize_witness(option["witness"], None, config)))
    kinds = Counter(row["kind"] for row in catalog["choices"])
    candidate_field_count = sum(
        len(CANDIDATE_PROPOSAL_FIELDS & set(row)) for row in public + hidden
    )
    checks = {
        "choice_query_family_counts": bool(
            len(catalog["choices"]) == config["gates"]["requiredChoiceCount"]
            and kinds == Counter({"KNOWN": 4, "NOVEL_CANDIDATE": 1, "UNSUPPORTED": 1, "INSUFFICIENT_EVIDENCE": 1})
            and len(catalog["queries"]) == config["gates"]["requiredQueryCount"]
            and len({row["query_id"] for row in catalog["queries"]}) == len(catalog["queries"])
            and all(len(row["options"]) == config["gates"]["requiredOptionsPerQuery"] for row in catalog["queries"])
        ),
        "population_split_stage_counts": bool(
            summary["family_count"] == config["gates"]["requiredFamilyCount"]
            and summary["group_count"] == config["gates"]["requiredGroupCount"]
            and summary["fixture_count"] == config["gates"]["requiredFixtureCount"]
            and summary["request_fixture_count"] == config["gates"]["requiredRequestFixtureCount"]
            and summary["closed_answer_fixture_count"] == config["gates"]["requiredClosedAnswerFixtureCount"]
            and summary["split_counts"] == {"development": 144, "evaluation": 144}
        ),
        "group_stage_completeness": all(
            len(rows) == config["gates"]["requiredStageCountPerGroup"]
            and {row["stage"] for row in rows} == set(STAGES)
            and len({row["split"] for row in rows}) == 1
            for rows in groups.values()
        ),
        "public_hidden_alignment_without_truth_leakage": bool(
            set(public_by_id) == set(hidden_by_id)
            and all(not (forbidden & set(row)) for row in public)
            and all(
                all(public_by_id[key][field] == hidden_by_id[key][field] for field in public_by_id[key])
                for key in public_by_id
            )
        ),
        "no_candidate_proposal_or_pruning_surface": candidate_field_count == config["gates"]["requiredCandidateProposalFieldCount"],
        "compatibility_exact": all(
            (row["truth_state_id"] == config["insufficientId"] and len(row["compatible_state_ids"]) == 2)
            or (row["truth_state_id"] != config["insufficientId"] and row["compatible_state_ids"] == [row["truth_state_id"]])
            for row in hidden
        ),
        "closed_answer_events_only_on_answer_stages": all(
            (row["closed_answer_event"] is not None) == (row["stage"] in ANSWER_STAGES)
            for row in hidden
        ),
        "closed_answer_witness_routing_exact": all(
            output["witness_valid"]
            and output["final_state_id"] == answered[index]["truth_state_id"]
            and output["llm_proposal_non_authoritative"]
            and not output["capability_defined_or_registered"]
            and not output["executable"]
            for index, output in enumerate(answered_outputs)
        ),
        "preanswer_and_malformed_fail_closed": bool(
            all(not output["witness_valid"] and output["final_state_id"] == config["insufficientId"] for output in preanswer_outputs)
            and all(not output["witness_valid"] and output["final_state_id"] == config["insufficientId"] for output in malformed_outputs)
        ),
        "registered_query_options_route_exactly": all(
            output["witness_valid"] and output["final_state_id"] == option["state_id"]
            for option, output in option_outputs
        ),
        "fresh_exact_conversations": not (current_conversations & prior_conversations),
        "complete_hypothesis_retention_zero_execution": all(
            output["authoritative_hypothesis_ids_retained"] == config["outputIds"]
            and not output["authoritative_hypothesis_universe_pruned"]
            and output["actual_execution_count"] == 0
            for output in answered_outputs + preanswer_outputs + malformed_outputs
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "candidate_proposal_field_count": candidate_field_count,
        "closed_answer_witness_routing": sum(
            output["final_state_id"] == answered[index]["truth_state_id"]
            for index, output in enumerate(answered_outputs)
        ) / len(answered_outputs),
        "preanswer_fail_closed_rate": sum(output["final_state_id"] == config["insufficientId"] for output in preanswer_outputs) / len(preanswer_outputs),
        "malformed_answer_event_fail_closed_rate": sum(output["final_state_id"] == config["insufficientId"] for output in malformed_outputs) / len(malformed_outputs),
        "exact_prior_conversation_overlap_count": len(current_conversations & prior_conversations),
        "true_hypothesis_retention": 1.0,
        "project_language_generation_count": len(public),
        "model_load_count": 0,
        "model_generation_or_score_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "actual_execution_count": 0,
    }


__all__ = [
    "ANSWER_STAGES", "CANDIDATE_PROPOSAL_FIELDS", "REQUEST_STAGES", "STAGES",
    "audit_population", "build_catalog", "build_population", "malformed_answer_events",
    "witness_from_answer_event",
]
