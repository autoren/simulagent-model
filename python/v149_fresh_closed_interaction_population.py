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
ANSWER_STAGES = {"closed_answer_known", "closed_answer_right"}


def _complete_witness(partial: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    return {
        "evidence_status": "SUFFICIENT",
        "source": config["trustedSource"],
        **partial,
    }


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
        "schema_version": "149-closed-interaction-catalog",
        "choices": config["catalog"]["choices"],
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


def _render(template: str, slots: dict[str, str]) -> str:
    return template.format(**slots)


def _fixture_id(group_id: str, stage: str) -> str:
    return "v149-" + hashlib.sha256(f"{group_id}|{stage}".encode()).hexdigest()[:16]


def build_population(config: dict[str, Any]) -> dict[str, Any]:
    catalog = build_catalog(config)
    query_by_id = {row["query_id"]: row for row in catalog["queries"]}
    public_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(config["families"]):
        left, right = family["left_choice_id"], family["right_choice_id"]
        pair = sorted([left, right])
        query = query_by_id[family["query_id"]]
        right_kind = next(row["kind"] for row in catalog["choices"] if row["choice_id"] == right)
        for variant_index, slots in enumerate(family["slot_variants"]):
            split = "development" if variant_index < 4 else "evaluation"
            group_id = f"v149-g{family_index:02d}-{variant_index:02d}"
            ambiguous = _render(family["ambiguous"], slots)
            specs = {
                "request_known_familiar": {
                    "conversation": [{"role": "user", "text": _render(family["known_familiar"], slots)}],
                    "truth": left,
                    "compatible": [left],
                    "language_class": "known_familiar",
                    "event": None,
                },
                "request_known_unfamiliar": {
                    "conversation": [{"role": "user", "text": _render(family["known_unfamiliar"], slots)}],
                    "truth": left,
                    "compatible": [left],
                    "language_class": "known_unfamiliar",
                    "event": None,
                },
                "request_right": {
                    "conversation": [{"role": "user", "text": _render(family["right_clear"], slots)}],
                    "truth": right,
                    "compatible": [right],
                    "language_class": "novel_candidate" if right_kind == "NOVEL_CANDIDATE" else "unsupported",
                    "event": None,
                },
                "request_ambiguous": {
                    "conversation": [{"role": "user", "text": ambiguous}],
                    "truth": config["insufficientId"],
                    "compatible": pair,
                    "language_class": "insufficient_evidence",
                    "event": None,
                },
                "closed_answer_known": {
                    "conversation": [
                        {"role": "user", "text": ambiguous},
                        {"role": "assistant", "text": query["question"]},
                        {"role": "user", "text": _render(family["left_answer"], slots)},
                    ],
                    "truth": left,
                    "compatible": [left],
                    "language_class": "known_closed_answer",
                    "event": {"query_id": family["query_id"], "selected_option_id": "LEFT"},
                },
                "closed_answer_right": {
                    "conversation": [
                        {"role": "user", "text": ambiguous},
                        {"role": "assistant", "text": query["question"]},
                        {"role": "user", "text": _render(family["right_answer"], slots)},
                    ],
                    "truth": right,
                    "compatible": [right],
                    "language_class": "novel_candidate_closed_answer" if right_kind == "NOVEL_CANDIDATE" else "unsupported_closed_answer",
                    "event": {"query_id": family["query_id"], "selected_option_id": "RIGHT"},
                },
            }
            for stage in STAGES:
                spec = specs[stage]
                fixture_id = _fixture_id(group_id, stage)
                public = {
                    "fixture_id": fixture_id,
                    "split": split,
                    "presented_candidate_choice_id": family["presented_candidate_choice_id"],
                    "conversation": spec["conversation"],
                    "closed_answer_event": spec["event"],
                }
                witness = witness_from_answer_event(spec["event"], catalog)
                hidden = {
                    **public,
                    "group_id": group_id,
                    "family_id": family["family_id"],
                    "stage": stage,
                    "language_class": spec["language_class"],
                    "truth_state_id": spec["truth"],
                    "compatible_state_ids": spec["compatible"],
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
        "closed_answer_fixture_count": sum(row["closed_answer_event"] is not None for row in public_rows),
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


def malformed_answer_events() -> list[Any]:
    return [
        None,
        {},
        {"query_id": "Q41"},
        {"selected_option_id": "LEFT"},
        {"query_id": "UNKNOWN", "selected_option_id": "LEFT"},
        {"query_id": "Q41", "selected_option_id": "UNKNOWN"},
        {"query_id": "Q41", "selected_option_id": "LEFT", "extra": True},
        ["Q41", "LEFT"],
    ]


def audit_population(
    population: dict[str, Any],
    config: dict[str, Any],
    prior_public_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog = population["interaction_catalog"]
    public, hidden = population["public_fixtures"], population["hidden_fixtures"]
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
    preanswer = [row for row in hidden if row["stage"] not in ANSWER_STAGES]
    answered_outputs = [
        finalize_witness(row["oracle_witness"], row["presented_candidate_choice_id"], config)
        for row in answered
    ]
    preanswer_outputs = [
        finalize_witness(None, row["presented_candidate_choice_id"], config)
        for row in preanswer
    ]
    malformed_outputs = [
        finalize_witness(witness_from_answer_event(event, catalog), "K41", config)
        for event in malformed_answer_events()
    ]
    option_outputs = []
    for query in catalog["queries"]:
        for option in query["options"]:
            output = finalize_witness(option["witness"], "A00", config)
            option_outputs.append((option, output))
    kinds = Counter(row["kind"] for row in catalog["choices"])
    checks = {
        "choice_and_query_counts": bool(
            len(catalog["choices"]) == config["gates"]["requiredChoiceCount"]
            and kinds == Counter({"KNOWN": 4, "NOVEL_CANDIDATE": 1, "UNSUPPORTED": 1, "INSUFFICIENT_EVIDENCE": 1})
            and len(catalog["queries"]) == config["gates"]["requiredQueryCount"]
            and all(len(query["options"]) == config["gates"]["requiredOptionsPerQuery"] for query in catalog["queries"])
        ),
        "family_group_fixture_split_counts": bool(
            summary["family_count"] == config["gates"]["requiredFamilyCount"]
            and summary["group_count"] == config["gates"]["requiredGroupCount"]
            and summary["fixture_count"] == config["gates"]["requiredFixtureCount"]
            and summary["split_counts"] == {"development": 144, "evaluation": 144}
            and summary["closed_answer_fixture_count"] == config["gates"]["requiredClosedAnswerFixtureCount"]
        ),
        "group_stage_completeness": all(
            len(rows) == config["gates"]["requiredStageCountPerGroup"]
            and {row["stage"] for row in rows} == set(STAGES)
            and len({row["split"] for row in rows}) == 1
            for rows in groups.values()
        ),
        "public_hidden_alignment_and_no_ground_truth_leakage": bool(
            set(public_by_id) == set(hidden_by_id)
            and all(not (forbidden & set(row)) for row in public)
            and all(all(public_by_id[key][field] == hidden_by_id[key][field] for field in public_by_id[key]) for key in public_by_id)
        ),
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
            and not output["capability_defined_or_registered"]
            and not output["executable"]
            for index, output in enumerate(answered_outputs)
        ),
        "preanswer_evidence_fails_closed": all(
            not output["witness_valid"] and output["final_state_id"] == config["insufficientId"]
            for output in preanswer_outputs
        ),
        "malformed_answer_events_fail_closed": all(
            not output["witness_valid"] and output["final_state_id"] == config["insufficientId"]
            for output in malformed_outputs
        ),
        "every_registered_query_option_maps_exactly": all(
            output["witness_valid"] and output["final_state_id"] == option["state_id"]
            for option, output in option_outputs
        ),
        "exact_conversation_nonoverlap": not (current_conversations & prior_conversations),
        "true_hypothesis_retention_and_zero_execution": all(
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
        "closed_answer_witness_routing": sum(output["final_state_id"] == answered[index]["truth_state_id"] for index, output in enumerate(answered_outputs)) / len(answered_outputs),
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
    "ANSWER_STAGES",
    "STAGES",
    "audit_population",
    "build_catalog",
    "build_population",
    "malformed_answer_events",
    "witness_from_answer_event",
]
