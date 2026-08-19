from __future__ import annotations

from collections import Counter
from typing import Any

from v93_open_set_source import canonical_sha256
from v106_open_world_benchmark import (
    ask_always_prediction, evaluate_predictions, oracle_prediction, prediction, retrieval_prediction,
)
from v112_open_world_full_policy_transfer import (
    novelty_evidence_metrics, policy_prediction, policy_quality_gates, population_gates,
    select_fresh_population,
)
from v113_known_disagreement_rescue import apply_rule, extract_rescue_features


def merged_excluded_population(populations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for population in populations for row in population["selected_population"]]
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("V114 excluded populations overlap unexpectedly")
    return {"selected_population": rows}


def select_v114_population(
    inventory: dict[str, Any], excluded_population: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    """Reuse V112's hash-only selector while assigning V114's distinct role and identifiers."""
    population = select_fresh_population(inventory, excluded_population, config)
    role = config["extraction"]["role"]
    for row in population["selected_population"]:
        row["role"] = role
        row["population_id"] = f"v114::{role}::{row['candidate_id']}"
    population["selected_population"].sort(key=lambda row: row["population_id"])
    population["selected_population_sha256"] = canonical_sha256(population["selected_population"])
    return population


def rescued_policy_predictions(
    records: list[dict[str, Any]], features: list[dict[str, Any]],
    direct: dict[str, dict[str, Any]], retrieval: dict[str, dict[str, Any]],
    v112_config: dict[str, Any], v114_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], int]:
    by_id = {row["record_id"]: row for row in features}
    actions, evidence, rescued = {}, {}, 0
    rule = v114_config["selectedRescueRule"]
    for record in records:
        identifier = record["record_id"]
        action, item_evidence = policy_prediction(direct[identifier], retrieval[identifier], v112_config)
        if apply_rule(rule, by_id[identifier]):
            action = prediction(
                "KNOWN", v114_config["rescueActionConfidence"],
                known_intent=direct[identifier]["known_intent"],
            )
            rescued += 1
        actions[identifier], evidence[identifier] = action, item_evidence
    return actions, evidence, rescued


def aggregate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "scored_rows"}


def rescue_mechanism_gates(diagnostics: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    spec = config["pairedRescueEvaluation"]
    return {
        "minimum_eligible_disagreement_opportunity": diagnostics["eligible_disagreement_count"] >= spec["minimumEligibleDisagreementCountForMechanismConclusion"],
        "minimum_triggered_rescue_opportunity": diagnostics["triggered_rescue_count"] >= spec["minimumTriggeredRescueCountForMechanismConclusion"],
        "minimum_rescue_precision": diagnostics["rescue_precision"] >= spec["minimumRescuePrecision"],
        "minimum_net_corrected_errors": diagnostics["net_corrected_errors"] >= spec["minimumNetCorrectedErrors"],
        "known_accuracy_not_worse": diagnostics["metric_deltas_rescued_minus_baseline"]["known_exact_intent_accuracy"] >= spec["minimumKnownExactIntentAccuracyDelta"],
        "selective_error_not_worse": diagnostics["metric_deltas_rescued_minus_baseline"]["top_confidence_80_percent_error"] <= spec["maximumTopConfidence80PercentErrorDelta"],
        "mean_regret_not_worse": diagnostics["metric_deltas_rescued_minus_baseline"]["mean_regret"] <= spec["maximumMeanRegretDelta"],
        "false_known_acceptance_not_worse": diagnostics["metric_deltas_rescued_minus_baseline"]["false_known_acceptance_rate"] <= spec["maximumFalseKnownAcceptanceRateDelta"],
        "unsupported_precision_not_worse": diagnostics["metric_deltas_rescued_minus_baseline"]["unsupported_precision"] >= spec["minimumUnsupportedPrecisionDelta"],
        "unsupported_recall_not_worse": diagnostics["metric_deltas_rescued_minus_baseline"]["unsupported_recall"] >= spec["minimumUnsupportedRecallDelta"],
        "novel_evidence_exactly_unchanged": diagnostics["novel_evidence_exactly_unchanged"] is spec["requireExactNovelEvidenceIdentity"],
    }


def classify_transfer(
    base_policy_pass: bool, rescued_policy_pass: bool, novel_evidence_pass: bool,
    access_pass: bool, diagnostics: dict[str, Any],
) -> dict[str, Any]:
    gates = diagnostics["mechanism_gates"]
    opportunity_names = {
        "minimum_eligible_disagreement_opportunity", "minimum_triggered_rescue_opportunity",
    }
    preservation_names = {
        "known_accuracy_not_worse", "selective_error_not_worse", "mean_regret_not_worse",
        "false_known_acceptance_not_worse", "unsupported_precision_not_worse",
        "unsupported_recall_not_worse", "novel_evidence_exactly_unchanged",
    }
    nonopportunity_pass = all(value for name, value in gates.items() if name not in opportunity_names)
    preservation_pass = all(gates[name] for name in preservation_names)
    opportunity = diagnostics["opportunity_sufficient"]
    mechanism_pass = diagnostics["mechanism_pass"]
    if not access_pass:
        decision = "invalid_due_to_access_gate_failure"
        mechanism_status = "invalid"
    elif not novel_evidence_pass:
        decision = "novel_evidence_nontransfer_close_abstention_signal_beyond_V112"
        mechanism_status = "not_interpretable"
    elif not preservation_pass:
        decision = "rescue_rejected_for_paired_safety_or_utility_harm"
        mechanism_status = "rejected"
    elif opportunity and mechanism_pass and rescued_policy_pass and not base_policy_pass:
        decision = "strong_rescue_transfer_rescued_passes_baseline_fails"
        mechanism_status = "transferred"
    elif opportunity and mechanism_pass and rescued_policy_pass and base_policy_pass:
        decision = "weaker_positive_rescue_transfer_both_policies_pass"
        mechanism_status = "transferred"
    elif not opportunity and nonopportunity_pass and rescued_policy_pass:
        decision = "full_policy_transfers_rescue_mechanism_inconclusive_for_insufficient_opportunity"
        mechanism_status = "inconclusive_insufficient_opportunity"
    elif opportunity and not mechanism_pass:
        decision = "rescue_mechanism_rejected_on_fresh_opportunities"
        mechanism_status = "rejected"
    elif base_policy_pass and not rescued_policy_pass:
        decision = "baseline_transfers_but_rescue_policy_fails"
        mechanism_status = "rejected"
    else:
        decision = "both_full_policies_fail_require_new_evidence_or_policy_structure"
        mechanism_status = "not_transferred"
    induction_design_authorized = bool(
        access_pass and novel_evidence_pass and rescued_policy_pass and preservation_pass
        and (mechanism_pass or (not opportunity and nonopportunity_pass))
    )
    return {
        "decision": decision,
        "mechanism_status": mechanism_status,
        "opportunity_sufficient": opportunity,
        "nonopportunity_mechanism_gates_pass": nonopportunity_pass,
        "paired_preservation_gates_pass": preservation_pass,
        "preregister_sandboxed_typed_induction_feasibility": induction_design_authorized,
    }


def paired_rescue_diagnostics(
    records: list[dict[str, Any]], features: list[dict[str, Any]],
    direct: dict[str, dict[str, Any]], base_actions: dict[str, dict[str, Any]],
    rescued_actions: dict[str, dict[str, Any]], base_scored: dict[str, Any],
    rescued_scored: dict[str, Any], base_novelty: dict[str, Any],
    rescued_novelty: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    base_by_id = {row["record_id"]: row for row in base_scored["scored_rows"]}
    rescued_by_id = {row["record_id"]: row for row in rescued_scored["scored_rows"]}
    feature_by_id = {row["record_id"]: row for row in features}
    transitions = Counter()
    changed_transitions = Counter()
    triggered_by_class = Counter()
    triggered_by_scenario = Counter()
    triggered_by_intent = Counter()
    eligible_by_class = Counter()
    for record in records:
        identifier = record["record_id"]
        baseline_correct = bool(base_by_id[identifier]["exact_decision"])
        rescued_correct = bool(rescued_by_id[identifier]["exact_decision"])
        transition = f"{'correct' if baseline_correct else 'wrong'}_to_{'correct' if rescued_correct else 'wrong'}"
        transitions[transition] += 1
        if feature_by_id[identifier]["eligible"]:
            eligible_by_class[record["class_label"]] += 1
        if base_actions[identifier] != rescued_actions[identifier]:
            changed_transitions[transition] += 1
            triggered_by_class[record["class_label"]] += 1
            triggered_by_scenario[record["scenario"]] += 1
            triggered_by_intent[direct[identifier].get("known_intent") or "NONE"] += 1
    changed_count = sum(changed_transitions.values())
    beneficial = changed_transitions["wrong_to_correct"]
    introduced = changed_transitions["correct_to_wrong"]
    base_aggregate = aggregate_metrics(base_scored)
    rescued_aggregate = aggregate_metrics(rescued_scored)
    deltas = {
        "observed_exact_decision_accuracy": rescued_aggregate["observed_exact_decision_accuracy"] - base_aggregate["observed_exact_decision_accuracy"],
        "known_exact_intent_accuracy": rescued_aggregate["known_exact_intent_accuracy"] - base_aggregate["known_exact_intent_accuracy"],
        "top_confidence_80_percent_error": rescued_aggregate["top_confidence_80_percent_error"] - base_aggregate["top_confidence_80_percent_error"],
        "mean_regret": rescued_aggregate["mean_regret"] - base_aggregate["mean_regret"],
        "false_known_acceptance_rate": rescued_aggregate["false_known_acceptance_rate"] - base_aggregate["false_known_acceptance_rate"],
        "unsupported_precision": rescued_aggregate["per_status"]["UNSUPPORTED"]["precision"] - base_aggregate["per_status"]["UNSUPPORTED"]["precision"],
        "unsupported_recall": rescued_aggregate["per_status"]["UNSUPPORTED"]["recall"] - base_aggregate["per_status"]["UNSUPPORTED"]["recall"],
    }
    class_deltas = {
        label: {
            "exact_decision_accuracy": rescued_aggregate["per_class"][label]["exact_decision_accuracy"] - base_aggregate["per_class"][label]["exact_decision_accuracy"],
            "mean_regret": rescued_aggregate["per_class"][label]["mean_regret"] - base_aggregate["per_class"][label]["mean_regret"],
        }
        for label in sorted(base_aggregate["per_class"])
    }
    diagnostics = {
        "eligible_disagreement_count": sum(row["eligible"] for row in features),
        "triggered_rescue_count": changed_count,
        "beneficial_correction_count": beneficial,
        "introduced_error_count": introduced,
        "net_corrected_errors": beneficial - introduced,
        "rescue_precision": beneficial / changed_count if changed_count else 0.0,
        "all_record_transition_counts": dict(sorted(transitions.items())),
        "changed_record_transition_counts": dict(sorted(changed_transitions.items())),
        "eligible_by_class": dict(sorted(eligible_by_class.items())),
        "triggered_by_class": dict(sorted(triggered_by_class.items())),
        "triggered_by_scenario": dict(sorted(triggered_by_scenario.items())),
        "triggered_by_proposed_intent": dict(sorted(triggered_by_intent.items())),
        "metric_deltas_rescued_minus_baseline": deltas,
        "per_class_deltas_rescued_minus_baseline": class_deltas,
        "novel_evidence_exactly_unchanged": base_novelty == rescued_novelty,
    }
    diagnostics["mechanism_gates"] = rescue_mechanism_gates(diagnostics, config)
    opportunity_names = (
        "minimum_eligible_disagreement_opportunity", "minimum_triggered_rescue_opportunity",
    )
    diagnostics["opportunity_sufficient"] = all(
        diagnostics["mechanism_gates"][name] for name in opportunity_names
    )
    diagnostics["mechanism_pass"] = all(diagnostics["mechanism_gates"].values())
    diagnostics["individual_record_emission_count"] = 0
    return diagnostics


def evaluate_transfer(
    records: list[dict[str, Any]], fixtures: dict[str, dict[str, Any]],
    fitted: dict[str, Any], retrieval: dict[str, dict[str, Any]],
    access: dict[str, Any], v112_config: dict[str, Any],
    v114_config: dict[str, Any], baseline_config: dict[str, Any],
) -> dict[str, Any]:
    observed = {row["record_id"]: fixtures[row["record_id"]] for row in records}
    controls = [row for row in fixtures.values() if row["kind"] == "controlled_missing_observation"]
    direct = {identifier: row["parsed_response"] for identifier, row in observed.items()}
    features = extract_rescue_features(fitted, records, direct)
    base_actions, evidence, _ = rescued_policy_predictions(
        records, features, direct, retrieval, v112_config,
        {**v114_config, "selectedRescueRule": {"family": "no_rescue", "complexity": 0}},
    )
    rescued_actions, rescued_evidence, rescued_count = rescued_policy_predictions(
        records, features, direct, retrieval, v112_config, v114_config,
    )
    fixed = v112_config["fixedRetrievalThresholds"]
    predictions = {
        "ask_always": {row["record_id"]: ask_always_prediction(row) for row in records},
        "direct_llm": direct,
        "fixed_v110_character_retrieval": {
            row["record_id"]: retrieval_prediction(
                retrieval[row["record_id"]], fixed["known"], fixed["unsupported"],
            ) for row in records
        },
        "V112_validated_novelty_evidence_policy": base_actions,
        "V114_rescued_policy": rescued_actions,
        "oracle": {row["record_id"]: oracle_prediction(row) for row in records},
    }
    scored_metrics = {
        name: evaluate_predictions(records, values, baseline_config)
        for name, values in predictions.items()
    }
    metrics = {name: aggregate_metrics(value) for name, value in scored_metrics.items()}
    base_novelty = novelty_evidence_metrics(records, evidence)
    novelty = novelty_evidence_metrics(records, rescued_evidence)
    if novelty != base_novelty:
        raise RuntimeError("V114 rescue changed novelty evidence")
    validity = sum(row["response_valid"] for row in fixtures.values()) / len(fixtures)
    control_accuracy = sum(
        row["response_valid"] and row["parsed_response"]["status"] == "ABSTAIN"
        for row in controls
    ) / len(controls)
    rescued_quality = policy_quality_gates(
        validity, control_accuracy, novelty, metrics["V114_rescued_policy"], 0, 1.0, v112_config,
    )
    base_quality = policy_quality_gates(
        validity, control_accuracy, novelty, metrics["V112_validated_novelty_evidence_policy"], 0, 1.0, v112_config,
    )
    paired = paired_rescue_diagnostics(
        records, features, direct, base_actions, rescued_actions,
        scored_metrics["V112_validated_novelty_evidence_policy"],
        scored_metrics["V114_rescued_policy"], base_novelty, novelty, v114_config,
    )
    if paired["triggered_rescue_count"] != rescued_count:
        raise RuntimeError("V114 rescue count does not match paired action changes")
    limits = v112_config["accessGates"]
    access_gates = {
        "fresh_development_language_read_budget": access["fresh_development_language_read_count"] <= limits["maximumFreshDevelopmentLanguageReadCount"],
        "zero_protected_test_language_reads": access["protected_test_language_read_count"] <= limits["maximumProtectedTestLanguageReadCount"],
        "zero_manual_language_or_raw_response_inspection": access["manual_language_or_raw_response_inspection_count"] <= limits["maximumManualLanguageOrRawResponseInspectionCount"],
        "model_load_budget": access["model_load_count"] <= limits["maximumModelLoadCount"],
        "model_generation_budget": access["model_generation_count"] <= limits["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= limits["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= limits["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= limits["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= limits["maximumExternalSideEffectCount"],
    }
    return {
        "interface_validity": validity,
        "controlled_missing_observation_abstention_accuracy": control_accuracy,
        "novel_evidence_metrics": novelty,
        "policy_metrics": metrics,
        "base_quality_gates": base_quality,
        "rescued_quality_gates": rescued_quality,
        "paired_rescue_diagnostics": paired,
        "access_gates": access_gates,
        "actual_execution_count": 0,
        "true_hypothesis_retention": 1.0,
        "individual_evidence_emission_count": 0,
    }


__all__ = [
    "classify_transfer", "evaluate_transfer", "merged_excluded_population", "paired_rescue_diagnostics",
    "rescue_mechanism_gates", "rescued_policy_predictions", "select_v114_population",
    "population_gates",
]
