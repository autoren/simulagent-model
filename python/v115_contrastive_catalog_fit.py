from __future__ import annotations

import json
from typing import Any

from v93_open_set_source import canonical_sha256
from v106_open_world_benchmark import (
    ask_always_prediction, evaluate_predictions, oracle_prediction, prediction, retrieval_prediction,
)
from v109_open_world_typed_choice import validate_and_expand_choice
from v112_open_world_full_policy_transfer import (
    novelty_evidence_metrics, policy_prediction, policy_quality_gates,
    population_gates, select_fresh_population,
)


def merged_excluded_population(populations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for population in populations for row in population["selected_population"]]
    identifiers = [row["candidate_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("V115 excluded populations overlap unexpectedly")
    return {"selected_population": rows}


def select_v115_population(
    inventory: dict[str, Any], excluded_population: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    population = select_fresh_population(inventory, excluded_population, config)
    role = config["extraction"]["role"]
    for row in population["selected_population"]:
        row["role"] = role
        row["population_id"] = f"v115::{role}::{row['candidate_id']}"
    population["selected_population"].sort(key=lambda row: row["population_id"])
    population["selected_population_sha256"] = canonical_sha256(population["selected_population"])
    return population


def known_choices(choice_catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in choice_catalog["choices"] if row["kind"] == "KNOWN"]


def choice_for_intent(choice_catalog: dict[str, Any], intent_id: str) -> dict[str, Any]:
    matches = [row for row in known_choices(choice_catalog) if row["intent_id"] == intent_id]
    if len(matches) != 1:
        raise ValueError("reviewed intent does not map to one exact known choice")
    return matches[0]


def reviewed_choice(
    pass_one: dict[str, Any], nearest_intent: str | None,
    choice_catalog: dict[str, Any], observation_available: bool,
) -> dict[str, Any]:
    if not observation_available:
        return min(known_choices(choice_catalog), key=lambda row: row["choice_id"])
    intent = pass_one["known_intent"] if pass_one["status"] == "KNOWN" else nearest_intent
    if not isinstance(intent, str):
        raise ValueError("observed contrastive review requires a deterministic candidate")
    return choice_for_intent(choice_catalog, intent)


def render_contrastive_prompt(
    choice_catalog: dict[str, Any], candidate: dict[str, Any], utterance: str | None,
    observation_available: bool, config: dict[str, Any],
) -> str:
    if observation_available and not isinstance(utterance, str):
        raise ValueError("observed fixture requires an utterance")
    if not observation_available and utterance is not None:
        raise ValueError("missing fixture cannot expose an utterance")
    spec = config["contrastiveInterface"]
    payload = {
        "instruction": (
            "Challenge candidate_under_review against every declared capability. C00 only if that exact "
            "candidate fully covers the request. O00 only if a different declared K choice fully covers it. "
            "N00 only for a coherent request in a visible scenario that needs a valid capability absent "
            "from the declared catalog. U00 only outside every visible scenario. A00 when evidence is "
            "insufficient. Return exactly one JSON object and no explanation."
        ),
        "candidate_under_review": candidate,
        "complete_declared_catalog": known_choices(choice_catalog),
        "visible_scenarios": sorted({row["scenario"] for row in known_choices(choice_catalog)}),
        "verdicts": spec["verdicts"],
        "allowed_selected_choices": [
            {key: row[key] for key in row if key in {"choice_id", "kind", "scenario", "meaning"}}
            for row in choice_catalog["choices"]
        ],
        "response_contract": {
            "required_keys": spec["outputKeys"],
            "verdict_id": "exactly one of C00,O00,N00,U00,A00",
            "selected_choice_id": "one supplied choice_id consistent with the verdict",
            "verdict_confidence": "number from 0 through 1",
            "novel_probability": "probability from 0 through 1 that N00 is true",
            "extra_keys_allowed": False,
        },
        "observation_available": observation_available,
        "user_utterance": utterance if observation_available else config["prompt"]["missingObservationSentinel"],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fallback() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {"status": "ABSTAIN", "known_intent": None, "novel_scenario": None, "confidence": 0.0},
        {
            "verdict_id": "A00", "selected_choice_id": "A00",
            "novel_candidate": False, "novel_evidence_probability": 0.0,
            "complete_safe_hypothesis_universe_retained": True,
            "capability_defined": False, "executable": False,
        },
    )


def validate_and_expand_contrastive(
    response: str | dict[str, Any], candidate: dict[str, Any],
    choice_catalog: dict[str, Any], config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool, str]:
    try:
        value = json.loads(response) if isinstance(response, str) else response
    except (json.JSONDecodeError, TypeError):
        prediction_value, evidence = _fallback()
        return prediction_value, evidence, False, "invalid_json"
    required = set(config["contrastiveInterface"]["outputKeys"])
    if not isinstance(value, dict) or set(value) != required:
        prediction_value, evidence = _fallback()
        return prediction_value, evidence, False, "invalid_keys"
    verdict = value.get("verdict_id")
    selected = value.get("selected_choice_id")
    confidence = value.get("verdict_confidence")
    novel_probability = value.get("novel_probability")
    if (
        verdict not in config["contrastiveInterface"]["verdicts"]
        or not isinstance(selected, str)
        or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
        or isinstance(novel_probability, bool) or not isinstance(novel_probability, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
        or not 0.0 <= float(novel_probability) <= 1.0
    ):
        prediction_value, evidence = _fallback()
        return prediction_value, evidence, False, "invalid_types"
    by_id = {row["choice_id"]: row for row in choice_catalog["choices"]}
    if selected not in by_id:
        prediction_value, evidence = _fallback()
        return prediction_value, evidence, False, "unknown_selected_choice"
    chosen = by_id[selected]
    consistent = bool(
        (verdict == "C00" and selected == candidate["choice_id"] and chosen["kind"] == "KNOWN")
        or (verdict == "O00" and selected != candidate["choice_id"] and chosen["kind"] == "KNOWN")
        or (verdict == "N00" and chosen["kind"] == "NOVEL")
        or (verdict == "U00" and chosen["kind"] == "UNSUPPORTED")
        or (verdict == "A00" and chosen["kind"] == "ABSTAIN")
    )
    if not consistent:
        prediction_value, evidence = _fallback()
        return prediction_value, evidence, False, "inconsistent_verdict_and_choice"
    prediction_value = {
        "status": chosen["kind"], "known_intent": None,
        "novel_scenario": None, "confidence": float(confidence),
    }
    if chosen["kind"] == "KNOWN":
        prediction_value["known_intent"] = chosen["intent_id"]
    elif chosen["kind"] == "NOVEL":
        prediction_value["novel_scenario"] = chosen["scenario"]
    evidence = {
        "verdict_id": verdict, "selected_choice_id": selected,
        "novel_candidate": verdict == "N00",
        "novel_evidence_probability": float(novel_probability),
        "complete_safe_hypothesis_universe_retained": True,
        "capability_defined": False, "executable": False,
    }
    return prediction_value, evidence, True, "valid"


def contrastive_policy_prediction(
    pass_one: dict[str, Any], pass_two: dict[str, Any], pass_two_evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        pass_one["status"] == "KNOWN" and pass_two_evidence["verdict_id"] == "C00"
        and pass_two["status"] == "KNOWN"
        and pass_one["known_intent"] == pass_two["known_intent"]
    ):
        action = prediction(
            "KNOWN", min(pass_one["confidence"], pass_two["confidence"]),
            known_intent=pass_one["known_intent"],
        )
        state = "two_pass_confirmed_known"
    elif pass_one["status"] == "UNSUPPORTED" and pass_two["status"] == "UNSUPPORTED":
        action = prediction("UNSUPPORTED", min(pass_one["confidence"], pass_two["confidence"]))
        state = "two_pass_confirmed_unsupported"
    else:
        action = prediction("ABSTAIN", 0.0)
        state = "ask_for_clarification"
    evidence = {**pass_two_evidence, "policy_state": state}
    return action, evidence


def _aggregate(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "scored_rows"}


def contrastive_evidence_gates(
    pass_one_validity: float, pass_two_validity: float, both_control_accuracy: float,
    novelty: dict[str, Any], pass_two_metrics: dict[str, Any], config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["contrastiveEvidenceGates"]
    return {
        "pass_one_structured_validity": pass_one_validity >= gates["minimumPassOneStructuredValidity"],
        "pass_two_structured_validity": pass_two_validity >= gates["minimumPassTwoStructuredValidity"],
        "both_pass_missing_observation_abstention": both_control_accuracy >= gates["minimumBothPassMissingObservationAbstentionAccuracy"],
        "explicit_novel_precision": novelty["precision"] >= gates["minimumExplicitNovelPrecision"],
        "explicit_novel_recall": novelty["recall"] >= gates["minimumExplicitNovelRecall"],
        "explicit_novel_non_novel_false_positive_rate": novelty["non_novel_false_positive_rate"] <= gates["maximumExplicitNovelNonNovelFalsePositiveRate"],
        "explicit_novel_ECE": novelty["ECE_10_bin"] <= gates["maximumExplicitNovelECE"],
        "pass_two_observed_exact_decisions": pass_two_metrics["observed_exact_decision_accuracy"] >= gates["minimumPassTwoObservedExactDecisionAccuracy"],
        "pass_two_observed_status_macro_F1": pass_two_metrics["observed_status_macro_f1"] >= gates["minimumPassTwoObservedStatusMacroF1"],
        "pass_two_known_exact_intent": pass_two_metrics["known_exact_intent_accuracy"] >= gates["minimumPassTwoKnownExactIntentAccuracy"],
        "pass_two_novel_exact_scenario": pass_two_metrics["novel_exact_scenario_accuracy"] >= gates["minimumPassTwoNovelExactScenarioAccuracy"],
        "pass_two_unsupported_recall": pass_two_metrics["per_status"]["UNSUPPORTED"]["recall"] >= gates["minimumPassTwoUnsupportedRecall"],
        "pass_two_unsupported_precision": pass_two_metrics["per_status"]["UNSUPPORTED"]["precision"] >= gates["minimumPassTwoUnsupportedPrecision"],
        "pass_two_false_known_acceptance": pass_two_metrics["false_known_acceptance_rate"] <= gates["maximumPassTwoFalseKnownAcceptanceRate"],
    }


def evaluate_v115(
    records: list[dict[str, Any]], fixtures: dict[str, dict[str, Any]],
    retrieval: dict[str, dict[str, Any]], access: dict[str, Any],
    v112_config: dict[str, Any], v115_config: dict[str, Any], baseline_config: dict[str, Any],
) -> dict[str, Any]:
    observed = {row["record_id"]: fixtures[row["record_id"]] for row in records}
    controls = [row for row in fixtures.values() if row["kind"] == "controlled_missing_observation"]
    pass_one = {identifier: row["pass_one"]["parsed_response"] for identifier, row in observed.items()}
    pass_two = {identifier: row["pass_two"]["parsed_response"] for identifier, row in observed.items()}
    pass_two_evidence = {identifier: row["pass_two"]["evidence"] for identifier, row in observed.items()}
    base_actions, base_evidence = {}, {}
    contrastive_actions, contrastive_evidence = {}, {}
    for record in records:
        identifier = record["record_id"]
        base_actions[identifier], base_evidence[identifier] = policy_prediction(
            pass_one[identifier], retrieval[identifier], v112_config,
        )
        contrastive_actions[identifier], contrastive_evidence[identifier] = contrastive_policy_prediction(
            pass_one[identifier], pass_two[identifier], pass_two_evidence[identifier],
        )
    fixed = v112_config["fixedRetrievalThresholds"]
    predictions = {
        "ask_always": {row["record_id"]: ask_always_prediction(row) for row in records},
        "pass_one_direct": pass_one,
        "pass_two_semantic_review": pass_two,
        "V112_policy_from_pass_one": base_actions,
        "V115_contrastive_policy": contrastive_actions,
        "fixed_character_retrieval": {
            row["record_id"]: retrieval_prediction(
                retrieval[row["record_id"]], fixed["known"], fixed["unsupported"],
            ) for row in records
        },
        "oracle": {row["record_id"]: oracle_prediction(row) for row in records},
    }
    metrics = {
        name: _aggregate(evaluate_predictions(records, values, baseline_config))
        for name, values in predictions.items()
    }
    pass_one_validity = sum(row["pass_one"]["response_valid"] for row in fixtures.values()) / len(fixtures)
    pass_two_validity = sum(row["pass_two"]["response_valid"] for row in fixtures.values()) / len(fixtures)
    pass_one_control_accuracy = sum(
        row["pass_one"]["response_valid"] and row["pass_one"]["parsed_response"]["status"] == "ABSTAIN"
        for row in controls
    ) / len(controls)
    both_control_accuracy = sum(
        row["pass_one"]["response_valid"] and row["pass_one"]["parsed_response"]["status"] == "ABSTAIN"
        and row["pass_two"]["response_valid"] and row["pass_two"]["evidence"]["verdict_id"] == "A00"
        for row in controls
    ) / len(controls)
    base_novelty = novelty_evidence_metrics(records, base_evidence)
    contrastive_novelty = novelty_evidence_metrics(records, contrastive_evidence)
    evidence_gates = contrastive_evidence_gates(
        pass_one_validity, pass_two_validity, both_control_accuracy,
        contrastive_novelty, metrics["pass_two_semantic_review"], v115_config,
    )
    base_quality = policy_quality_gates(
        pass_one_validity, pass_one_control_accuracy, base_novelty,
        metrics["V112_policy_from_pass_one"], 0, 1.0, v112_config,
    )
    combined_quality = policy_quality_gates(
        min(pass_one_validity, pass_two_validity), both_control_accuracy, contrastive_novelty,
        metrics["V115_contrastive_policy"], 0, 1.0, v112_config,
    )
    limits = v115_config["accessGates"]
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
        "pass_one_structured_validity": pass_one_validity,
        "pass_two_structured_validity": pass_two_validity,
        "pass_one_missing_observation_abstention_accuracy": pass_one_control_accuracy,
        "both_pass_missing_observation_abstention_accuracy": both_control_accuracy,
        "base_novel_evidence_metrics": base_novelty,
        "contrastive_novel_evidence_metrics": contrastive_novelty,
        "policy_metrics": metrics,
        "contrastive_evidence_gates": evidence_gates,
        "base_quality_gates": base_quality,
        "combined_quality_gates": combined_quality,
        "access_gates": access_gates,
        "true_hypothesis_retention": 1.0, "actual_execution_count": 0,
        "individual_evidence_emission_count": 0,
    }


def classify_v115(summary: dict[str, Any]) -> dict[str, Any]:
    evidence_pass = all(summary["contrastive_evidence_gates"].values())
    policy_pass = all(summary["combined_quality_gates"].values())
    access_pass = all(summary["access_gates"].values())
    if not access_pass:
        decision = "invalid_due_to_access_gate_failure"
    elif evidence_pass and policy_pass:
        decision = "positive_contrastive_development_seek_independent_source_transfer"
    elif evidence_pass:
        decision = "contrastive_evidence_positive_policy_negative_require_new_population_policy"
    else:
        decision = "contrastive_evidence_negative_close_two_pass_single_model_branch"
    return {
        "contrastive_evidence_pass": evidence_pass,
        "combined_policy_pass": policy_pass,
        "base_policy_pass": all(summary["base_quality_gates"].values()),
        "access_pass": access_pass, "decision": decision,
        "seek_independent_source_transfer": bool(access_pass and evidence_pass and policy_pass),
        "schema_induction_authorized": False,
    }


__all__ = [
    "choice_for_intent", "classify_v115", "contrastive_evidence_gates",
    "contrastive_policy_prediction", "evaluate_v115", "known_choices",
    "merged_excluded_population", "population_gates", "render_contrastive_prompt",
    "reviewed_choice", "select_v115_population", "validate_and_expand_contrastive",
    "validate_and_expand_choice",
]
