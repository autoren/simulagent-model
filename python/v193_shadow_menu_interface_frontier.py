from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from typing import Any


def normalize_proposal(value: Any, valid_option_ids: set[str]) -> dict[str, Any]:
    insufficient = {"status": "INSUFFICIENT", "ranked_option_ids": []}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return insufficient
    if not isinstance(value, dict) or set(value) != {"status", "ranked_option_ids"}:
        return insufficient
    status = value.get("status")
    ranked = value.get("ranked_option_ids")
    if status == "INSUFFICIENT" and ranked == []:
        return insufficient
    if status != "RANKED" or not isinstance(ranked, list) or not 1 <= len(ranked) <= 3:
        return insufficient
    if any(not isinstance(item, str) or item not in valid_option_ids for item in ranked):
        return insufficient
    if len(set(ranked)) != len(ranked):
        return insufficient
    return {"status": "RANKED", "ranked_option_ids": list(ranked)}


def build_interface(
    contract_catalog: dict[str, Any],
    development_bindings: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    contracts = sorted(contract_catalog["contracts"], key=lambda row: row["capability_contract_id"])
    visible = []
    hidden = []
    for index, contract in enumerate(contracts, start=1):
        option_id = f"M{index:02d}"
        visible.append(
            {
                "option_id": option_id,
                "domain": contract["semantic_payload"]["domain"],
                "intent_concept": contract["normalized_intent_name"],
            }
        )
        hidden.append(
            {"option_id": option_id, "capability_contract_id": contract["capability_contract_id"]}
        )
    observed = [row for row in development_bindings["bindings"] if row["observation_available"]]
    prior_counts = Counter(row["target_contract_id"] for row in observed)
    total = sum(prior_counts.values())
    prior = {contract["capability_contract_id"]: prior_counts[contract["capability_contract_id"]] / total for contract in contracts}

    domains: dict[str, list[str]] = defaultdict(list)
    for contract in contracts:
        domains[contract["semantic_payload"]["domain"]].append(contract["capability_contract_id"])
    bit_cost = config["trustedController"]["bitCost"]
    domain_cost = bit_cost * math.ceil(math.log2(len(domains)))
    fixed_costs = {
        contract_id: domain_cost + (bit_cost if len(domains[domain]) > 1 else 0.0)
        for domain, contract_ids in domains.items()
        for contract_id in contract_ids
    }
    fixed_mean = sum(prior[key] * fixed_costs[key] for key in prior)
    generic = config["trustedController"]["outsideMenuGenericClarificationCost"]
    oracle_top1 = bit_cost * math.ceil(math.log2(2))
    oracle_top3 = bit_cost * math.ceil(math.log2(4))
    economics = config["economics"]
    denominator = economics["frontierRecallGridDenominator"]
    frontier = []
    for numerator in range(denominator + 1):
        recall = numerator / denominator
        frontier.append(
            {
                "recall": recall,
                "top1_mean_cost": oracle_top1 + (1.0 - recall) * generic,
                "top3_mean_cost": oracle_top3 + (1.0 - recall) * generic,
            }
        )
    first_top1_break_even = next(row["recall"] for row in frontier if row["top1_mean_cost"] < economics["fixedV190MeanCost"] - 1e-12)
    first_top1_material = next(row["recall"] for row in frontier if row["top1_mean_cost"] <= economics["maximumQualifyingMeanCost"] + 1e-12)
    first_top3_break_even = next(row["recall"] for row in frontier if row["top3_mean_cost"] < economics["fixedV190MeanCost"] - 1e-12)
    first_top3_material = next(row["recall"] for row in frontier if row["top3_mean_cost"] <= economics["maximumQualifyingMeanCost"] + 1e-12)

    valid_ids = {row["option_id"] for row in visible}
    invalid_cases = [
        None,
        "",
        "{",
        {"status": "RANKED"},
        {"status": "RANKED", "ranked_option_ids": []},
        {"status": "RANKED", "ranked_option_ids": ["UNKNOWN"]},
        {"status": "RANKED", "ranked_option_ids": ["M01", "M01"]},
        {"status": "RANKED", "ranked_option_ids": ["M01", "M02", "M03", "M04"]},
        {"status": "RANKED", "ranked_option_ids": ["M01"], "confidence": 1.0},
        {"status": "INSUFFICIENT", "ranked_option_ids": ["M01"]},
        {"status": "OTHER", "ranked_option_ids": []},
        ["M01"],
    ]
    invalid_rate = sum(normalize_proposal(value, valid_ids)["status"] == "INSUFFICIENT" for value in invalid_cases) / len(invalid_cases)
    summary = {
        "visible_option_count": len(visible),
        "hidden_mapping_count": len(hidden),
        "distinct_domain_count": len({row["domain"] for row in visible}),
        "distinct_intent_concept_count": len({row["intent_concept"] for row in visible}),
        "primary_prior_contract_count": sum(value > 0 for value in prior.values()),
        "primary_prior_observed_count": total,
        "primary_prior_counts": dict(sorted(prior_counts.items())),
        "fixed_hierarchy_mean_cost": fixed_mean,
        "always_generic_mean_cost": generic,
        "oracle_top1_mean_cost": oracle_top1,
        "oracle_top3_mean_cost": oracle_top3,
        "oracle_target_retention_rate": 1.0,
        "invalid_parser_case_count": len(invalid_cases),
        "invalid_parser_cases_insufficient_rate": invalid_rate,
        "missing_insufficient_rate": 1.0,
        "top1_first_grid_break_even_recall": first_top1_break_even,
        "top1_first_grid_material_recall": first_top1_material,
        "top3_first_grid_break_even_recall": first_top3_break_even,
        "top3_first_grid_material_recall": first_top3_material,
        "utterance_or_dialogue_language_read_count": 0,
        "protected_language_read_count": 0,
        "deterministic_language_score_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    return {
        "visible_menu": {
            "schema_version": "193-visible-finite-benchmark-menu",
            "options": visible,
            "option_count": len(visible),
            "finite_benchmark_menu_not_ontology_authority": True,
        },
        "hidden_option_map": {
            "schema_version": "193-hidden-menu-option-map",
            "mappings": hidden,
            "mapping_count": len(hidden),
        },
        "prior": prior,
        "fixed_costs": fixed_costs,
        "frontier": frontier,
        "summary": summary,
    }


def audit_interface(interface: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = interface["summary"]
    gates = config["interfaceGates"]
    checks = {
        "visible_menu_and_hidden_mapping_are_complete": bool(
            summary["visible_option_count"] == gates["requiredVisibleOptionCount"]
            and summary["hidden_mapping_count"] == gates["requiredHiddenMappingCount"]
            and summary["distinct_domain_count"] == gates["requiredDistinctDomainCount"]
            and summary["distinct_intent_concept_count"] == gates["requiredDistinctIntentConceptCount"]
        ),
        "primary_prior_and_baselines_reproduce_V190": bool(
            summary["primary_prior_contract_count"] == gates["requiredPrimaryPriorContractCount"]
            and summary["primary_prior_observed_count"] == gates["requiredPrimaryPriorObservedCount"]
            and abs(summary["fixed_hierarchy_mean_cost"] - gates["requiredFixedHierarchyMeanCost"]) <= 1e-12
            and abs(summary["always_generic_mean_cost"] - gates["requiredAlwaysGenericMeanCost"]) <= 1e-12
        ),
        "oracle_controls_and_retention_are_exact": bool(
            abs(summary["oracle_top1_mean_cost"] - gates["requiredOracleTop1MeanCost"]) <= 1e-12
            and abs(summary["oracle_top3_mean_cost"] - gates["requiredOracleTop3MeanCost"]) <= 1e-12
            and summary["oracle_target_retention_rate"] == gates["requiredOracleTargetRetentionRate"]
        ),
        "parser_and_missing_controls_fail_closed": bool(
            summary["invalid_parser_cases_insufficient_rate"] == gates["requiredInvalidParserCasesInsufficientRate"]
            and summary["missing_insufficient_rate"] == gates["requiredMissingInsufficientRate"]
        ),
        "material_recall_thresholds_are_exact": bool(
            summary["top1_first_grid_material_recall"] == gates["requiredTop1MaterialRecallThreshold"]
            and summary["top3_first_grid_material_recall"] == gates["requiredTop3MaterialRecallThreshold"]
        ),
        "language_model_authority_and_execution_access_is_zero": all(
            summary[key] == gates[gate]
            for key, gate in (
                ("utterance_or_dialogue_language_read_count", "maximumUtteranceOrDialogueLanguageReadCount"),
                ("protected_language_read_count", "maximumProtectedLanguageReadCount"),
                ("deterministic_language_score_count", "maximumDeterministicLanguageScoreCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = ["audit_interface", "build_interface", "normalize_proposal"]
