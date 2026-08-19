from __future__ import annotations

from collections import Counter
import math
from typing import Any


def _cost(
    proposal: list[str],
    target: str,
    fixed_cost: float,
    config: dict[str, Any],
) -> float:
    """Return the frozen trusted-controller cost for one proposal set."""
    if not proposal:
        return fixed_cost
    controller = config["trustedController"]
    question_cost = controller["bitCost"] * math.ceil(math.log2(len(proposal) + 1))
    miss_cost = 0.0 if target in proposal else controller["targetOutsideProposalAdditionalGenericCost"]
    return question_cost + miss_cost


def _plurality(sources: dict[str, list[str]], presentations: list[str]) -> list[str]:
    """Choose the plurality top-1 contract with the preregistered tie break."""
    top_contracts = [sources[presentation][0] for presentation in presentations if sources[presentation]]
    if not top_contracts:
        return []
    counts = Counter(top_contracts)
    maximum = max(counts.values())
    tied = {contract for contract, count in counts.items() if count == maximum}
    canonical = sources["CANONICAL"][0] if sources["CANONICAL"] else None
    return [canonical if canonical in tied else sorted(tied)[0]]


def _consensus(
    sources: dict[str, list[str]],
    presentations: list[str],
    minimum_votes: int,
    maximum_size: int,
) -> list[str]:
    """Return contracts included in enough top-3 sets, using fixed ordering."""
    votes = Counter(
        contract
        for presentation in presentations
        for contract in set(sources[presentation][:3])
    )
    canonical_rank = {
        contract: index for index, contract in enumerate(sources["CANONICAL"][:3])
    }
    chosen = [contract for contract, count in votes.items() if count >= minimum_votes]
    chosen.sort(
        key=lambda contract: (
            -votes[contract],
            canonical_rank.get(contract, 999),
            contract,
        )
    )
    return chosen[:maximum_size]


def _extract_model_sources(
    canonical_census: dict[str, Any],
    transformed_census: dict[str, Any],
    canonical_map: dict[str, Any],
    variant_maps: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    canonical_contract = {
        row["option_id"]: row["capability_contract_id"]
        for row in canonical_map["mappings"]
    }
    maps_by_record = {row["record_id"]: row for row in variant_maps["records"]}
    sources: dict[str, dict[str, list[str]]] = {}
    for record_id, fixture in canonical_census["fixtures"].items():
        proposal = fixture["normalized_proposal"]
        sources[record_id] = {
            "CANONICAL": (
                [canonical_contract[option_id] for option_id in proposal["ranked_option_ids"]]
                if proposal["status"] == "RANKED"
                else []
            )
        }
        for variant_id in ("ORDER_ONLY", "ORDER_AND_OPAQUE_ID"):
            transformed = transformed_census["fixtures"][
                f"{record_id}@@{variant_id}"
            ]["normalized_proposal"]
            variant = next(
                row
                for row in maps_by_record[record_id]["variants"]
                if row["variant_id"] == variant_id
            )
            by_option = {
                row["option_id"]: row["capability_contract_id"]
                for row in variant["mappings"]
            }
            sources[record_id][variant_id] = (
                [by_option[option_id] for option_id in transformed["ranked_option_ids"]]
                if transformed["status"] == "RANKED"
                else []
            )
    return sources


def _extract_char_last_sources(
    canonical_predictions: list[dict[str, Any]],
    transformed_records: list[dict[str, Any]],
    canonical_map: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    canonical_contract = {
        row["option_id"]: row["capability_contract_id"]
        for row in canonical_map["mappings"]
    }
    sources: dict[str, dict[str, list[str]]] = {}
    for row in canonical_predictions:
        if row["ranker_id"] != "CHAR_LAST":
            continue
        sources[row["record_id"]] = {
            "CANONICAL": [
                canonical_contract[option_id]
                for option_id in row["proposal"]["ranked_option_ids"]
            ]
        }
    for row in transformed_records:
        sources[row["record_id"]][row["variant_id"]] = list(
            row["ranked_contract_ids"]
        )
    return sources


def _proposal_for_policy(
    policy: dict[str, Any],
    sources: dict[str, list[str]],
    presentations: list[str],
) -> list[str]:
    if policy["kind"] == "top1_plurality":
        return _plurality(sources, presentations)
    if policy["kind"] == "top3_inclusion_consensus":
        return _consensus(
            sources,
            presentations,
            policy["minimumPresentationVotes"],
            policy["maximumMenuSize"],
        )
    raise ValueError(f"unsupported multi-presentation policy: {policy['kind']}")


def _condition_summary(
    condition: str,
    rows: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    return {
        "condition": condition,
        "primary_mean_cost": sum(
            weights[row["record_id"]] * row["model_cost"] for row in rows
        ),
        "macro_mean_cost": sum(row["model_cost"] for row in rows) / len(rows),
        "matched_CHAR_LAST_primary_mean_cost": sum(
            weights[row["record_id"]] * row["CHAR_LAST_cost"] for row in rows
        ),
        "matched_CHAR_LAST_macro_mean_cost": sum(
            row["CHAR_LAST_cost"] for row in rows
        )
        / len(rows),
        "target_hit_rate": sum(row["model_target_hit"] for row in rows) / len(rows),
        "mean_proposal_set_size": sum(len(row["model_proposal"]) for row in rows)
        / len(rows),
    }


def _evaluate_policy(
    policy: dict[str, Any],
    model_sources: dict[str, dict[str, list[str]]],
    char_sources: dict[str, dict[str, list[str]]],
    hidden: dict[str, dict[str, Any]],
    weights: dict[str, float],
    fixed_costs: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    presentations = config["presentations"]
    conditions: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    condition_ids = (
        presentations
        if policy["kind"] == "single_presentation_family"
        else [policy["policyId"]]
    )
    for condition in condition_ids:
        current_rows: list[dict[str, Any]] = []
        for record_id, truth in hidden.items():
            if policy["kind"] == "single_presentation_family":
                model_proposal = model_sources[record_id][condition][: policy["rankLimit"]]
                char_proposal = char_sources[record_id][condition][: policy["rankLimit"]]
            else:
                model_proposal = _proposal_for_policy(
                    policy, model_sources[record_id], presentations
                )
                char_proposal = _proposal_for_policy(
                    policy, char_sources[record_id], presentations
                )
            target = truth["target_contract_id"]
            row = {
                "policy_id": policy["policyId"],
                "condition": condition,
                "record_id": record_id,
                "target_contract_id": target,
                "model_proposal": model_proposal,
                "CHAR_LAST_proposal": char_proposal,
                "model_target_hit": target in model_proposal,
                "CHAR_LAST_target_hit": target in char_proposal,
                "model_cost": _cost(
                    model_proposal, target, fixed_costs[target], config
                ),
                "CHAR_LAST_cost": _cost(
                    char_proposal, target, fixed_costs[target], config
                ),
            }
            current_rows.append(row)
            scored_rows.append(row)
        conditions.append(_condition_summary(condition, current_rows, weights))

    costs_by_record: dict[str, list[float]] = {record_id: [] for record_id in hidden}
    hits_by_record: dict[str, list[bool]] = {record_id: [] for record_id in hidden}
    for row in scored_rows:
        costs_by_record[row["record_id"]].append(row["model_cost"])
        hits_by_record[row["record_id"]].append(row["model_target_hit"])

    result: dict[str, Any] = {
        "policy_id": policy["policyId"],
        "eligible_as_single_call": policy["eligibleAsSingleCall"],
        "model_calls_per_decision": policy["modelCallsPerDecision"],
        "conditions": conditions,
        "robust_primary_mean_cost": max(row["primary_mean_cost"] for row in conditions),
        "robust_macro_mean_cost": max(row["macro_mean_cost"] for row in conditions),
        "worst_condition_incremental_primary_improvement_over_matched_CHAR_LAST": min(
            row["matched_CHAR_LAST_primary_mean_cost"] - row["primary_mean_cost"]
            for row in conditions
        ),
        "target_hit_disagreement_rate_across_presentations": sum(
            len(set(values)) > 1 for values in hits_by_record.values()
        )
        / len(hits_by_record),
        "mean_per_record_cost_range_across_presentations": sum(
            max(values) - min(values) for values in costs_by_record.values()
        )
        / len(costs_by_record),
        "mean_proposal_set_size": sum(
            row["mean_proposal_set_size"] for row in conditions
        )
        / len(conditions),
        "target_retention_rate": 1.0,
        "final_exactness_after_trusted_answers": 1.0,
        "false_terminal_decisions": 0,
    }
    gates = config["qualificationGates"]
    checks = {
        "robust_primary_cost": result["robust_primary_mean_cost"]
        <= gates["maximumRobustPrimaryMeanCost"] + 1e-12,
        "robust_macro_cost": result["robust_macro_mean_cost"]
        <= gates["maximumRobustMacroMeanCost"] + 1e-12,
        "incremental_over_matched_CHAR_LAST": result[
            "worst_condition_incremental_primary_improvement_over_matched_CHAR_LAST"
        ]
        >= gates["minimumWorstConditionIncrementalPrimaryImprovementOverMatchedCHARLAST"]
        - 1e-12,
        "target_hit_disagreement": result[
            "target_hit_disagreement_rate_across_presentations"
        ]
        <= gates["maximumTargetHitDisagreementRateAcrossPresentations"] + 1e-12,
        "per_record_cost_range": result[
            "mean_per_record_cost_range_across_presentations"
        ]
        <= gates["maximumMeanPerRecordCostRangeAcrossPresentations"] + 1e-12,
        "proposal_set_size": result["mean_proposal_set_size"]
        <= gates["maximumMeanProposalSetSize"] + 1e-12,
        "target_retention": result["target_retention_rate"]
        == gates["requiredTargetRetentionRate"],
        "final_exactness": result["final_exactness_after_trusted_answers"]
        == gates["requiredFinalExactnessAfterTrustedAnswers"],
        "zero_false_terminal": result["false_terminal_decisions"]
        <= gates["maximumFalseTerminalDecisions"],
    }
    result["qualification_gates"] = checks
    result["qualified"] = all(checks.values())
    result["scored_rows"] = scored_rows
    return result


def evaluate_controllers(
    canonical_census: dict[str, Any],
    transformed_census: dict[str, Any],
    canonical_char: list[dict[str, Any]],
    transformed_char: list[dict[str, Any]],
    hidden_targets: dict[str, Any],
    canonical_map: dict[str, Any],
    variant_maps: dict[str, Any],
    prior: dict[str, float],
    fixed_costs: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    model_sources = _extract_model_sources(
        canonical_census, transformed_census, canonical_map, variant_maps
    )
    char_sources = _extract_char_last_sources(
        canonical_char, transformed_char, canonical_map
    )
    hidden = {
        row["record_id"]: row
        for row in hidden_targets["records"]
        if row["observation_available"]
    }
    per_contract = Counter(row["target_contract_id"] for row in hidden.values())
    weights = {
        record_id: prior[row["target_contract_id"]]
        / per_contract[row["target_contract_id"]]
        for record_id, row in hidden.items()
    }
    if set(model_sources) != set(hidden) or set(char_sources) != set(hidden):
        raise ValueError("normalized source record IDs do not equal observed hidden-target IDs")
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("primary weights do not sum to one")

    policy_results = [
        _evaluate_policy(
            policy,
            model_sources,
            char_sources,
            hidden,
            weights,
            fixed_costs,
            config,
        )
        for policy in config["controllerPolicies"]
    ]
    first_tier = [
        row
        for row in policy_results
        if row["qualified"] and row["eligible_as_single_call"]
    ]
    candidates = first_tier or [row for row in policy_results if row["qualified"]]
    selected = (
        min(
            candidates,
            key=lambda row: (
                row["robust_primary_mean_cost"],
                row["robust_macro_mean_cost"],
                row["mean_proposal_set_size"],
                row["policy_id"],
            ),
        )
        if candidates
        else None
    )
    scored_records = [
        scored
        for policy_result in policy_results
        for scored in policy_result.pop("scored_rows")
    ]
    summary = {
        "record_count": len(hidden),
        "presentation_count": len(config["presentations"]),
        "policy_count": len(policy_results),
        "qualified_policy_count": sum(row["qualified"] for row in policy_results),
        "selected_policy_id": selected["policy_id"] if selected else None,
        "selected_policy_uses_single_call_tier": bool(
            selected and selected["eligible_as_single_call"]
        ),
        "policies": policy_results,
        "normalized_model_fixture_read_count": len(canonical_census["fixtures"])
        + len(transformed_census["fixtures"]),
        "normalized_CHAR_LAST_prediction_read_count": len(
            [row for row in canonical_char if row["ranker_id"] == "CHAR_LAST"]
        )
        + len(transformed_char),
        "raw_model_response_read_count": 0,
        "utterance_or_dialogue_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "protected_language_read_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    return {"summary": summary, "scored_records": scored_records}


def audit_evaluation(
    evaluation: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    summary = evaluation["summary"]
    access = config["accessGates"]
    policy_ids = [row["policyId"] for row in config["controllerPolicies"]]
    result_by_id = {row["policy_id"]: row for row in summary["policies"]}
    single_qualified = [
        row
        for row in summary["policies"]
        if row["qualified"] and row["eligible_as_single_call"]
    ]
    all_qualified = [row for row in summary["policies"] if row["qualified"]]
    tier = single_qualified or all_qualified
    expected_selected = (
        min(
            tier,
            key=lambda row: (
                row["robust_primary_mean_cost"],
                row["robust_macro_mean_cost"],
                row["mean_proposal_set_size"],
                row["policy_id"],
            ),
        )["policy_id"]
        if tier
        else None
    )
    checks = {
        "exact_record_presentation_and_policy_counts": bool(
            summary["record_count"] == 84
            and summary["presentation_count"] == 3
            and summary["policy_count"] == len(policy_ids) == 4
        ),
        "all_fixed_policies_evaluated_once": bool(
            list(result_by_id) == policy_ids
            and len(evaluation["scored_records"]) == 84 * (3 + 3 + 1 + 1)
        ),
        "policy_qualification_is_conjunction_of_frozen_gates": all(
            row["qualified"] == all(row["qualification_gates"].values())
            for row in summary["policies"]
        ),
        "selection_rule_reconstructs_exactly": bool(
            summary["selected_policy_id"] == expected_selected
            and summary["selected_policy_uses_single_call_tier"]
            == bool(expected_selected and result_by_id[expected_selected]["eligible_as_single_call"])
        ),
        "normalized_model_read_count_exact": summary[
            "normalized_model_fixture_read_count"
        ]
        == access["requiredNormalizedModelFixtureReadCount"],
        "normalized_CHAR_LAST_read_count_exact": summary[
            "normalized_CHAR_LAST_prediction_read_count"
        ]
        == access["requiredNormalizedCHARLASTPredictionReadCount"],
        "forbidden_access_and_effects_zero": all(
            summary[key] <= access[gate]
            for key, gate in (
                ("raw_model_response_read_count", "maximumRawModelResponseReadCount"),
                ("utterance_or_dialogue_language_read_count", "maximumUtteranceOrDialogueLanguageReadCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("protected_language_read_count", "maximumProtectedLanguageReadCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ),
        "safety_invariants_exact": all(
            row["target_retention_rate"] == 1.0
            and row["final_exactness_after_trusted_answers"] == 1.0
            and row["false_terminal_decisions"] == 0
            for row in summary["policies"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = [
    "_consensus",
    "_cost",
    "_plurality",
    "audit_evaluation",
    "evaluate_controllers",
]
