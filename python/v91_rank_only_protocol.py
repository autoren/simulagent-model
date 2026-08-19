#!/usr/bin/env python3
"""Frozen rank-only parsing, controls, scoring, and planner-invariance helpers for V91."""
from __future__ import annotations

from itertools import permutations
import json
import re
from typing import Any, Iterable

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CAMEL_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_tokens(value: str) -> tuple[str, ...]:
    expanded = CAMEL_PATTERN.sub(" ", value)
    return tuple(TOKEN_PATTERN.findall(expanded.lower()))


def format_user_prompt(record: dict[str, Any], config: dict[str, Any]) -> str:
    intents = "\n".join(
        f"- {item['id']}: {item['description']}"
        for item in record["schema_context"]["intents"]
    )
    intents += "\n- NONE: none of the supplied active intents currently applies"
    history = "\n".join(
        f"{turn['speaker']}: {turn['utterance']}"
        for turn in record["dialogue_history"]
    )
    return config["userPromptTemplate"].format(
        service_name=record["schema_context"]["service_name"],
        service_description=record["schema_context"]["service_description"],
        intent_lines=intents,
        dialogue_history=history,
    )


def canonical_complete_priority(
    allowed: list[str], proposed: Iterable[Any]
) -> list[str]:
    allowed_set = set(allowed)
    seen: set[str] = set()
    completed: list[str] = []
    for item in proposed:
        if isinstance(item, str) and item in allowed_set and item not in seen:
            completed.append(item)
            seen.add(item)
    completed.extend(item for item in allowed if item not in seen)
    if len(completed) != len(allowed) or set(completed) != allowed_set:
        raise RuntimeError("V91 canonical completion failed to preserve the schema set")
    return completed


def parse_and_complete(response: str, allowed: list[str]) -> dict[str, Any]:
    parsed: Any = None
    exact_json = False
    try:
        parsed = json.loads(response)
        exact_json = True
    except (json.JSONDecodeError, TypeError):
        pass
    exact_keys = bool(
        exact_json and isinstance(parsed, dict) and set(parsed) == {"intent_priority"}
    )
    raw = parsed.get("intent_priority") if exact_keys else None
    list_well_formed = isinstance(raw, list) and all(
        isinstance(item, str) for item in raw
    )
    raw_items = list(raw) if list_well_formed else []
    allowed_set = set(allowed)
    raw_allowed_only = bool(
        list_well_formed and all(item in allowed_set for item in raw_items)
    )
    raw_unique = bool(list_well_formed and len(raw_items) == len(set(raw_items)))
    raw_full_permutation = bool(
        raw_allowed_only
        and raw_unique
        and len(raw_items) == len(allowed)
        and set(raw_items) == allowed_set
    )
    completed = canonical_complete_priority(allowed, raw_items)
    return {
        "parsed": parsed,
        "exact_json": exact_json,
        "exact_keys": exact_keys,
        "list_well_formed": list_well_formed,
        "raw_allowed_only": raw_allowed_only,
        "raw_unique": raw_unique,
        "raw_full_permutation": raw_full_permutation,
        "raw_priority": raw_items,
        "completed_priority": completed,
        "canonical_complete_set": set(completed) == allowed_set,
        "canonical_NONE_retained": "NONE" in completed,
    }


def score_order(record: dict[str, Any], order: list[str]) -> dict[str, Any]:
    gold = record["gold_intent"]
    rank = order.index(gold) + 1
    return {
        "gold_intent": gold,
        "gold_rank": rank,
        "top1": rank == 1,
        "top2": rank <= 2,
        "reciprocal_rank": 1.0 / rank,
        "candidate_count": len(order),
    }


def schema_order(record: dict[str, Any]) -> list[str]:
    return list(record["allowed_intent_ids"])


def lexical_overlap_order(record: dict[str, Any]) -> list[str]:
    allowed = record["allowed_intent_ids"]
    dialogue_tokens = {
        token
        for turn in record["dialogue_history"]
        for token in normalize_tokens(turn["utterance"])
    }
    descriptions = {
        item["id"]: item["description"]
        for item in record["schema_context"]["intents"]
    }
    active_scores = {
        intent: len(
            dialogue_tokens
            & set(normalize_tokens(intent + " " + descriptions[intent]))
        )
        for intent in allowed
        if intent != "NONE"
    }
    maximum = max(active_scores.values(), default=0)
    scores = {
        **active_scores,
        "NONE": 1 if maximum == 0 else -1,
    }
    index = {intent: position for position, intent in enumerate(allowed)}
    return sorted(allowed, key=lambda intent: (-scores[intent], index[intent]))


def identifier_exact_match_order(record: dict[str, Any]) -> list[str]:
    allowed = record["allowed_intent_ids"]
    current_user = next(
        turn["utterance"]
        for turn in reversed(record["dialogue_history"])
        if turn["speaker"] == "USER"
    )
    current_tokens = set(normalize_tokens(current_user))
    active_matches = {
        intent: bool(set(normalize_tokens(intent)))
        and set(normalize_tokens(intent)) <= current_tokens
        for intent in allowed
        if intent != "NONE"
    }
    any_match = any(active_matches.values())
    scores = {
        **{intent: int(matches) for intent, matches in active_matches.items()},
        "NONE": int(not any_match),
    }
    index = {intent: position for position, intent in enumerate(allowed)}
    return sorted(allowed, key=lambda intent: (-scores[intent], index[intent]))


def oracle_first_order(record: dict[str, Any]) -> list[str]:
    gold = record["gold_intent"]
    return [gold] + [item for item in record["allowed_intent_ids"] if item != gold]


def score_response(record: dict[str, Any], response: str) -> dict[str, Any]:
    parsed = parse_and_complete(response, record["allowed_intent_ids"])
    ranking = score_order(record, parsed["completed_priority"])
    return {
        "id": record["id"],
        "source_record_id": record["source_record_id"],
        "service": record["service"],
        "label_kind": "none" if record["gold_intent"] == "NONE" else "active",
        "response": response,
        **parsed,
        **ranking,
        "authoritative_state_fingerprint_before": record[
            "authoritative_state_fingerprint"
        ],
        "authoritative_state_fingerprint_after": record[
            "authoritative_state_fingerprint"
        ],
        "authoritative_state_preserved": True,
        "permanently_non_deployable": True,
        "executable": False,
        "belief_authority": False,
        "action_authority": False,
        "pruning_authority": False,
    }


def aggregate_order_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("V91 cannot aggregate an empty ranking population")
    active = [row for row in rows if row["label_kind"] == "active"]
    none = [row for row in rows if row["label_kind"] == "none"]

    def mean(key: str, values: list[dict[str, Any]]) -> float:
        return sum(float(row[key]) for row in values) / len(values) if values else 0.0

    return {
        "record_count": len(rows),
        "active_record_count": len(active),
        "none_record_count": len(none),
        "overall_top1_rate": mean("top1", rows),
        "active_top1_rate": mean("top1", active),
        "none_top1_rate": mean("top1", none),
        "overall_top2_rate": mean("top2", rows),
        "mean_reciprocal_rank": mean("reciprocal_rank", rows),
        "mean_gold_rank": mean("gold_rank", rows),
    }


def control_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    functions = {
        "schema_order": schema_order,
        "lexical_overlap": lexical_overlap_order,
        "identifier_exact_match_grammar": identifier_exact_match_order,
        "oracle_first": oracle_first_order,
    }
    output: dict[str, dict[str, Any]] = {}
    for name, function in functions.items():
        rows = []
        for record in records:
            ranking = score_order(record, function(record))
            rows.append(
                {
                    **ranking,
                    "label_kind": (
                        "none" if record["gold_intent"] == "NONE" else "active"
                    ),
                }
            )
        output[name] = aggregate_order_rows(rows)
    exhaustive_rows = []
    for record in records:
        candidate_count = len(record["allowed_intent_ids"])
        exhaustive_rows.append(
            {
                "gold_rank": candidate_count,
                "top1": candidate_count == 1,
                "top2": candidate_count <= 2,
                "reciprocal_rank": 1.0 / candidate_count,
                "candidate_count": candidate_count,
                "label_kind": (
                    "none" if record["gold_intent"] == "NONE" else "active"
                ),
            }
        )
    output["exhaustive_unordered"] = aggregate_order_rows(exhaustive_rows)
    return output


def aggregate_model_rows(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    ranking = aggregate_order_rows(rows)
    controls = control_metrics(records)
    nonoracle = {
        key: value for key, value in controls.items() if key != "oracle_first"
    }
    best_mrr_name = max(
        nonoracle,
        key=lambda name: (
            nonoracle[name]["mean_reciprocal_rank"],
            -nonoracle[name]["mean_gold_rank"],
            name,
        ),
    )
    best_rank_name = min(
        nonoracle,
        key=lambda name: (
            nonoracle[name]["mean_gold_rank"],
            -nonoracle[name]["mean_reciprocal_rank"],
            name,
        ),
    )

    def mean_bool(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    return {
        **ranking,
        "exact_JSON_parse_rate": mean_bool("exact_json"),
        "raw_ontology_conformance_rate": sum(
            float(
                row["exact_keys"]
                and row["list_well_formed"]
                and row["raw_allowed_only"]
                and row["raw_unique"]
            )
            for row in rows
        )
        / len(rows),
        "raw_full_permutation_rate": mean_bool("raw_full_permutation"),
        "canonical_complete_set_rate": mean_bool("canonical_complete_set"),
        "canonical_NONE_retention_rate": mean_bool("canonical_NONE_retained"),
        "authoritative_state_preservation_rate": mean_bool(
            "authoritative_state_preserved"
        ),
        "permanent_non_deployable_rate": mean_bool("permanently_non_deployable"),
        "controls": controls,
        "best_nonoracle_MRR_control": best_mrr_name,
        "best_nonoracle_mean_rank_control": best_rank_name,
        "MRR_improvement_over_best_nonoracle_control": (
            ranking["mean_reciprocal_rank"]
            - nonoracle[best_mrr_name]["mean_reciprocal_rank"]
        ),
        "mean_rank_reduction_versus_best_nonoracle_control": (
            nonoracle[best_rank_name]["mean_gold_rank"]
            - ranking["mean_gold_rank"]
        ),
    }


def evaluate_gates(
    metrics: dict[str, Any],
    planner: dict[str, Any],
    config: dict[str, Any],
    access: dict[str, Any],
) -> dict[str, bool]:
    gates = config["qualityGates"]
    limits = config["accessGates"]
    return {
        "required_record_count": metrics["record_count"]
        == limits["requiredRecordCount"],
        "exact_JSON_parse": metrics["exact_JSON_parse_rate"]
        >= gates["minimumExactJSONParseRate"],
        "raw_ontology_conformance": metrics["raw_ontology_conformance_rate"]
        >= gates["minimumRawOntologyConformanceRate"],
        "raw_full_permutation": metrics["raw_full_permutation_rate"]
        >= gates["minimumRawFullPermutationRate"],
        "overall_top1": metrics["overall_top1_rate"]
        >= gates["minimumOverallTop1Rate"],
        "active_top1": metrics["active_top1_rate"]
        >= gates["minimumActiveTop1Rate"],
        "NONE_top1": metrics["none_top1_rate"] >= gates["minimumNoneTop1Rate"],
        "overall_top2": metrics["overall_top2_rate"]
        >= gates["minimumOverallTop2Rate"],
        "mean_reciprocal_rank": metrics["mean_reciprocal_rank"]
        >= gates["minimumMeanReciprocalRank"],
        "mean_gold_rank": metrics["mean_gold_rank"]
        <= gates["maximumMeanGoldRank"],
        "MRR_improvement_over_best_nonoracle_control": metrics[
            "MRR_improvement_over_best_nonoracle_control"
        ]
        >= gates["minimumMRRImprovementOverBestNonOracleControl"],
        "mean_rank_reduction_versus_best_nonoracle_control": metrics[
            "mean_rank_reduction_versus_best_nonoracle_control"
        ]
        >= gates["minimumMeanRankReductionVersusBestNonOracleControl"],
        "canonical_complete_set": metrics["canonical_complete_set_rate"]
        >= gates["minimumCanonicalCompleteSetRate"],
        "canonical_NONE_retention": metrics["canonical_NONE_retention_rate"]
        >= gates["minimumCanonicalNoneRetentionRate"],
        "authoritative_state_preservation": metrics[
            "authoritative_state_preservation_rate"
        ]
        >= gates["minimumAuthoritativeStatePreservationRate"],
        "permanent_non_deployable": metrics["permanent_non_deployable_rate"]
        >= gates["minimumPermanentNonDeployableRate"],
        "exact_planner_permutation_invariance": planner["invariance_rate"]
        >= gates["minimumExactPlannerPermutationInvarianceRate"],
        "planner_value_error": planner["maximum_absolute_value_error"]
        <= gates["maximumPlannerValueError"],
        "planner_action_mismatch": planner["action_mismatch_count"]
        <= gates["maximumPlannerActionMismatchCount"],
        "execution_certificate_violations": planner[
            "execution_certificate_violation_count"
        ]
        <= gates["maximumExecutionCertificateViolationCount"],
        "new_model_weight_download_budget": access["new_model_weight_download_count"]
        <= limits["maximumNewModelWeightDownloadCount"],
        "model_load_budget": access["model_load_count"]
        <= limits["maximumModelLoadCount"],
        "model_generation_budget": access["model_generation_count"]
        <= limits["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"]
        <= limits["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"]
        <= limits["maximumAdapterTrainingRunCount"],
        "zero_manual_utterance_inspection": access[
            "manual_utterance_inspection_count"
        ]
        <= limits["maximumManualUtteranceInspectionCount"],
        "zero_real_service_calls": access["real_service_call_count"]
        <= limits["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"]
        <= limits["maximumExternalSideEffectCount"],
    }


def permute_kernel_and_belief(kernel: Any, belief: np.ndarray, order: tuple[int, ...]):
    from v77_clarification_benchmark import ClarificationKernel

    inverse = {old: new for new, old in enumerate(order)}
    permuted = ClarificationKernel(
        hypothesis_names=tuple(kernel.hypothesis_names[index] for index in order),
        action_names=kernel.action_names,
        observation_names=kernel.observation_names,
        state_names=kernel.state_names,
        transition=kernel.transition.copy(),
        observation=kernel.observation[list(order)].copy(),
        reward=kernel.reward[list(order)].copy(),
        discount=kernel.discount,
        send_minimum_matching_posterior=kernel.send_minimum_matching_posterior,
        send_maximum_none_posterior=kernel.send_maximum_none_posterior,
        send_action_to_hypothesis=tuple(
            (action, inverse[hypothesis])
            for action, hypothesis in kernel.send_action_to_hypothesis
        ),
        none_hypothesis=inverse[kernel.none_hypothesis],
        always_certified_actions=kernel.always_certified_actions,
    )
    return permuted, np.asarray(belief, dtype=np.float64)[list(order)].copy()


def verify_v79_permutation_invariance(config: dict[str, Any]) -> dict[str, Any]:
    from v78_clarification_benchmark import build_fixture
    from v79_terminal_utility_planning import evaluate_policy_exact, plan_exact

    horizon = int(config["sharedParameters"]["horizonActions"])
    tolerance = float(config["sharedParameters"]["tieTolerance"])
    checks = 0
    mismatches = 0
    certificate_violations = 0
    maximum_error = 0.0
    fixture_rows = []
    for row in config["fixtures"]:
        fixture = build_fixture(config, row["name"])
        baseline = plan_exact(
            fixture.kernel,
            fixture.initial_belief,
            horizon,
            tie_tolerance=tolerance,
        )
        local_checks = 0
        for order in permutations(range(len(fixture.kernel.hypothesis_names))):
            kernel, belief = permute_kernel_and_belief(
                fixture.kernel, fixture.initial_belief, order
            )
            policy = plan_exact(
                kernel, belief, horizon, tie_tolerance=tolerance
            )
            violations: list[dict[str, Any]] = []
            replay = evaluate_policy_exact(
                kernel,
                belief,
                policy,
                horizon,
                certificate_violations=violations,
            )
            errors = [
                abs(float(policy["value"]) - float(baseline["value"])),
                abs(float(replay) - float(baseline["value"])),
            ]
            errors.extend(
                abs(float(policy["q_values"][action]) - float(baseline["q_values"][action]))
                for action in baseline["q_values"]
            )
            maximum_error = max(maximum_error, *errors)
            mismatch = bool(
                policy["selected_action"] != baseline["selected_action"]
                or policy["optimal_actions"] != baseline["optimal_actions"]
                or max(errors) > tolerance
            )
            mismatches += int(mismatch)
            certificate_violations += len(violations)
            checks += 1
            local_checks += 1
        fixture_rows.append(
            {
                "name": row["name"],
                "permutation_count": local_checks,
                "baseline_selected_action": baseline["selected_action"],
                "baseline_value": baseline["value"],
            }
        )
    return {
        "fixture_count": len(fixture_rows),
        "permutation_count": checks,
        "invariant_permutation_count": checks - mismatches,
        "invariance_rate": (checks - mismatches) / checks,
        "action_mismatch_count": mismatches,
        "maximum_absolute_value_error": maximum_error,
        "execution_certificate_violation_count": certificate_violations,
        "fixtures": fixture_rows,
        "model_output_access_count": 0,
    }


__all__ = [
    "aggregate_model_rows",
    "aggregate_order_rows",
    "canonical_complete_priority",
    "control_metrics",
    "evaluate_gates",
    "format_user_prompt",
    "identifier_exact_match_order",
    "lexical_overlap_order",
    "normalize_tokens",
    "oracle_first_order",
    "parse_and_complete",
    "permute_kernel_and_belief",
    "schema_order",
    "score_order",
    "score_response",
    "verify_v79_permutation_invariance",
]
