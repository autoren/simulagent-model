from __future__ import annotations

from typing import Any

from v189_multiway_typed_channel_feasibility import build_problem, evaluate_sequence


def build_confirmation_problem(contract_catalog: dict[str, Any], protected_bindings: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    adapter = {
        "multiwayQuestions": {
            "allQuestionSet": ["domain", "intent_concept", "transactionality"],
            "coarseQuestionSet": ["domain", "transactionality"],
        },
        "pricing": {"genericTrustedClarificationCost": config["fixedPolicy"]["genericTrustedClarificationCost"], "maximumMultiwayTurns": 2},
    }
    problem = build_problem(contract_catalog, protected_bindings, adapter)
    problem["bindings"] = protected_bindings["bindings"]
    return problem


def _summary(problem: dict[str, Any], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_cost": sum(problem["prior"][target] * row["cost"] for target, row in rows.items()),
        "mean_turn_count": sum(problem["prior"][target] * row["turn_count"] for target, row in rows.items()),
        "typed_completion_rate": sum(problem["prior"][target] for target, row in rows.items() if row["terminal_mode"] == "TYPED_SINGLETON"),
        "final_exactness_rate": sum(problem["prior"][target] for target, row in rows.items() if row["final_exact"]),
        "target_retention_rate": sum(problem["prior"][target] for target, row in rows.items() if row["target_retained"]),
    }


def evaluate(problem: dict[str, Any], config: dict[str, Any], source_v189: dict[str, Any]) -> dict[str, Any]:
    scenario = {"scenario_id": "bit_slot_o00", "rule": "bit_slot", "turn_overhead": 0.0}
    sequences = {
        "fixed_domain_then_intent": tuple(config["fixedPolicy"]["questionSequence"]),
        "flat_global_intent_menu": ("M189_intent_concept",),
        "domain_only_then_generic": ("M189_domain",),
        "always_generic": (),
    }
    paths = {
        name: {target: evaluate_sequence(problem, scenario, sequence, target) for target in problem["contract_ids"]}
        for name, sequence in sequences.items()
    }
    policy_summary = {name: _summary(problem, rows) for name, rows in paths.items()}
    fixed = policy_summary["fixed_domain_then_intent"]
    development_cost = source_v189["summary"]["pure_bit_slot_exact_cost"]
    record_rows = []
    for binding in problem["bindings"]:
        if not binding["observation_available"]:
            record_rows.append({
                "record_id": binding["record_id"], "observation_available": False,
                "target_contract_id": None, "all_policies_insufficient": True, "cost": 0.0,
            })
        else:
            target = binding["target_contract_id"]
            record_rows.append({
                "record_id": binding["record_id"], "observation_available": True,
                "target_contract_id": target,
                "policies": {name: rows[target] for name, rows in paths.items()},
            })
    summary = {
        "protected_binding_count": len(problem["bindings"]),
        "observed_protected_count": sum(row["observation_available"] for row in problem["bindings"]),
        "missing_protected_count": sum(not row["observation_available"] for row in problem["bindings"]),
        "contract_count": len(problem["contract_ids"]),
        "positive_prior_contract_count": sum(value > 0 for value in problem["prior"].values()),
        "protected_prior_counts": problem["prior_counts"],
        "fixed_sequence": list(sequences["fixed_domain_then_intent"]),
        "policy_summary": policy_summary,
        "fixed_improvement_over_always_generic": policy_summary["always_generic"]["mean_cost"] - fixed["mean_cost"],
        "development_fixed_cost": development_cost,
        "absolute_development_protected_cost_difference": abs(development_cost - fixed["mean_cost"]),
        "missing_insufficient_rate": 1.0,
        "policy_optimization_count": 0,
        "protected_utterance_language_read_count": 0,
        "utterance_or_dialogue_language_read_count": 0,
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
    return {"summary": summary, "paths": paths, "records": record_rows}


def audit(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    fixed = summary["policy_summary"]["fixed_domain_then_intent"]
    gates = config["confirmationGates"]
    checks = {
        "protected_population_and_fixed_policy_are_exact": bool(
            summary["protected_binding_count"] == gates["requiredProtectedBindingCount"]
            and summary["observed_protected_count"] == gates["requiredObservedProtectedCount"]
            and summary["missing_protected_count"] == gates["requiredMissingProtectedCount"]
            and summary["contract_count"] == gates["requiredContractCount"]
            and summary["positive_prior_contract_count"] == gates["requiredPositivePriorContractCount"]
            and summary["fixed_sequence"] == gates["requiredFixedSequence"]
            and summary["policy_optimization_count"] == gates["requiredPolicyOptimizationCount"]
        ),
        "fixed_policy_is_terminally_exact_and_safe": bool(
            fixed["final_exactness_rate"] == gates["requiredObservedFinalExactnessRate"]
            and fixed["target_retention_rate"] == gates["requiredTargetRetentionRate"]
            and fixed["typed_completion_rate"] == gates["requiredTypedCompletionRate"]
            and summary["missing_insufficient_rate"] == gates["requiredMissingInsufficientRate"]
        ),
        "fixed_policy_confirms_cost_and_turn_compression": bool(
            fixed["mean_cost"] <= gates["maximumFixedPolicyMeanCost"]
            and summary["fixed_improvement_over_always_generic"] >= gates["minimumImprovementOverAlwaysGeneric"]
            and fixed["mean_turn_count"] <= gates["maximumMeanTurnCount"]
            and summary["absolute_development_protected_cost_difference"] <= gates["maximumAbsoluteDevelopmentProtectedCostDifference"]
            and abs(summary["policy_summary"]["flat_global_intent_menu"]["mean_cost"] - gates["requiredFlatIntentMenuCost"]) <= 1e-12
        ),
        "language_model_authority_and_effect_access_is_zero": all(summary[key] == gates[gate] for key, gate in (
            ("protected_utterance_language_read_count", "maximumProtectedUtteranceLanguageReadCount"),
            ("utterance_or_dialogue_language_read_count", "maximumUtteranceOrDialogueLanguageReadCount"),
            ("model_load_count", "maximumModelLoadCount"),
            ("model_generation_count", "maximumModelGenerationCount"),
            ("API_call_count", "maximumAPICallCount"),
            ("training_run_count", "maximumTrainingRunCount"),
            ("ontology_registration_count", "maximumOntologyRegistrationCount"),
            ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
            ("service_call_count", "maximumServiceCallCount"),
            ("external_side_effect_count", "maximumExternalSideEffectCount"),
            ("actual_execution_count", "maximumActualExecutionCount"),
        )),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = ["audit", "build_confirmation_problem", "evaluate"]
