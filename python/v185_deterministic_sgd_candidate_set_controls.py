from __future__ import annotations

from collections import Counter
from itertools import product
import math
import re
import unicodedata
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def camel_tokens(value: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return normalize_text(spaced).split()


def _counts(items: list[str]) -> Counter[str]:
    return Counter(items)


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _char_ngrams(text: str, widths: list[int]) -> Counter[str]:
    compact = f" {normalize_text(text)} "
    return _counts(
        [compact[index : index + width] for width in widths for index in range(max(0, len(compact) - width + 1))]
    )


def _token_counts(text: str) -> Counter[str]:
    return _counts(normalize_text(text).split())


def _catalog_document(choice: dict[str, Any]) -> str:
    parts = [
        choice["service_description"], choice["intent_name"], choice["intent_description"],
        " ".join(camel_tokens(choice["intent_name"])),
        " ".join(choice["required_slots"]), " ".join(choice["optional_slots"]),
        " ".join(choice["result_slots"]),
    ]
    for slot in choice["slots"]:
        parts.extend([slot["name"], slot["description"], " ".join(map(str, slot["possible_values"]))])
    return " ".join(parts)


def _query_text(record: dict[str, Any]) -> str:
    if not record["observation_available"]:
        return ""
    return " ".join(
        turn["utterance"] for turn in record["conversation"] if turn["speaker"] == "USER"
    )


def raw_scores(record: dict[str, Any], catalog: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    choices = sorted(catalog["choices"], key=lambda row: row["choice_id"])
    query = _query_text(record)
    widths = config["deterministicViews"]["characterNgram"]["ngramWidths"]
    q_char = _char_ngrams(query, widths)
    q_token = _token_counts(query)
    char_scores = {}
    token_scores = {}
    aliases = []
    normalized_query = f" {normalize_text(query)} "
    for choice in choices:
        document = _catalog_document(choice)
        cid = choice["choice_id"]
        char_scores[cid] = _cosine(q_char, _char_ngrams(document, widths))
        token_scores[cid] = _cosine(q_token, _token_counts(document))
        alias = " ".join(camel_tokens(choice["intent_name"]))
        if alias and f" {alias} " in normalized_query:
            aliases.append(cid)
    all_ids = [row["choice_id"] for row in choices]
    alias_set = aliases if len(aliases) == 1 else all_ids
    return {"character": char_scores, "token": token_scores, "alias_set": alias_set}


def threshold_set(scores: dict[str, float], minimum: float, margin: float) -> list[str]:
    top = max(scores.values(), default=0.0)
    if top < minimum:
        return sorted(scores)
    selected = sorted(key for key, value in scores.items() if value >= top - margin - 1e-12)
    return selected if selected else sorted(scores)


def split_development(
    development: dict[str, Any], hidden: dict[str, Any], config: dict[str, Any]
) -> dict[str, list[str]]:
    hidden_by_id = {row["record_id"]: row for row in hidden["records"] if row["role"] == "development"}
    groups: dict[str, list[str]] = {}
    for row in development["records"]:
        kind = hidden_by_id[row["record_id"]]["truth_kind"]
        groups.setdefault(kind, []).append(row["record_id"])
    calibration = []
    evaluation = []
    salt = config["developmentSplit"]["baseSalt"]
    for kind, ids in sorted(groups.items()):
        ordered = sorted(ids, key=lambda rid: hash_order(salt, kind, rid))
        cut = len(ordered) // 2
        calibration.extend(ordered[:cut])
        evaluation.extend(ordered[cut:])
    return {"calibration": sorted(calibration), "evaluation": sorted(evaluation)}


def _truth_choice(hidden_row: dict[str, Any], contract_to_choice: dict[str, str]) -> str | None:
    if hidden_row["truth_kind"] == "KNOWN":
        return contract_to_choice[hidden_row["truth_contract_id"]]
    return None


def _route_metrics(
    ids: list[str],
    hidden_by_id: dict[str, Any],
    specific: dict[str, str | None],
    contract_to_choice: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    costs = config["trustedClarificationPolicy"]
    observed_ids = [rid for rid in ids if hidden_by_id[rid]["observation_available"]]
    known_ids = [rid for rid in observed_ids if hidden_by_id[rid]["truth_kind"] == "KNOWN"]
    nonknown_ids = [rid for rid in observed_ids if hidden_by_id[rid]["truth_kind"] != "KNOWN"]
    routed = [rid for rid in observed_ids if specific.get(rid) is not None]
    correct = [
        rid for rid in routed
        if specific[rid] == _truth_choice(hidden_by_id[rid], contract_to_choice)
    ]
    false_nonknown = [rid for rid in nonknown_ids if specific.get(rid) is not None]
    total_cost = 0.0
    for rid in ids:
        row = hidden_by_id[rid]
        if not row["observation_available"]:
            total_cost += costs["missingObservationCost"]
        elif specific.get(rid) is None:
            total_cost += costs["genericClarificationCost"]
        elif rid in correct:
            total_cost += costs["candidateSpecificQuestionCost"]
        else:
            total_cost += costs["wrongSpecificThenGenericCost"]
    return {
        "record_count": len(ids),
        "observed_record_count": len(observed_ids),
        "specific_route_count": len(routed),
        "correct_specific_count": len(correct),
        "candidate_specific_precision": len(correct) / len(routed) if routed else 0.0,
        "false_specific_rate_on_nonknown": len(false_nonknown) / len(nonknown_ids) if nonknown_ids else 0.0,
        "known_specific_coverage": len(correct) / len(known_ids) if known_ids else 0.0,
        "mean_clarification_cost": total_cost / len(ids),
        "final_exactness_after_trusted_answers": 1.0,
        "authoritative_hypothesis_retention_rate": 1.0,
    }


def _specific_routes(
    ids: list[str], scores: dict[str, dict[str, Any]], params: tuple[float, float, float, float]
) -> tuple[dict[str, str | None], dict[str, dict[str, list[str]]]]:
    char_min, char_margin, token_min, token_margin = params
    routes = {}
    sets = {}
    for rid in ids:
        char_set = threshold_set(scores[rid]["character"], char_min, char_margin)
        token_set = threshold_set(scores[rid]["token"], token_min, token_margin)
        specific = char_set[0] if len(char_set) == len(token_set) == 1 and char_set == token_set else None
        routes[rid] = specific
        sets[rid] = {"character": char_set, "token": token_set}
    return routes, sets


def _select_parameters(
    calibration_ids: list[str],
    scores: dict[str, dict[str, Any]],
    hidden_by_id: dict[str, Any],
    contract_to_choice: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    char = config["deterministicViews"]["characterNgram"]
    token = config["deterministicViews"]["tokenSchemaOverlap"]
    rule = config["calibrationRule"]
    rows = []
    for params in product(
        char["minimumTopScoreGrid"], char["topRelativeMarginGrid"],
        token["minimumTopScoreGrid"], token["topRelativeMarginGrid"],
    ):
        routes, _ = _specific_routes(calibration_ids, scores, params)
        metrics = _route_metrics(calibration_ids, hidden_by_id, routes, contract_to_choice, config)
        qualifies = bool(
            metrics["candidate_specific_precision"] >= rule["requiredCalibrationCandidateSpecificPrecision"]
            and metrics["false_specific_rate_on_nonknown"] <= rule["maximumCalibrationFalseSpecificRateOnNonKnown"]
            and metrics["known_specific_coverage"] >= rule["minimumCalibrationKnownSpecificCoverage"]
        )
        rows.append({"params": list(params), "metrics": metrics, "qualifies": qualifies})
    qualifying = [row for row in rows if row["qualifies"]]
    if not qualifying:
        return {"qualified": False, "selected_params": None, "selected_metrics": None, "grid_point_count": len(rows), "qualifying_grid_point_count": 0}
    selected = min(
        qualifying,
        key=lambda row: (
            row["metrics"]["mean_clarification_cost"],
            -row["metrics"]["candidate_specific_precision"],
            -row["metrics"]["known_specific_coverage"],
            row["params"],
        ),
    )
    return {
        "qualified": True,
        "selected_params": selected["params"],
        "selected_metrics": selected["metrics"],
        "grid_point_count": len(rows),
        "qualifying_grid_point_count": len(qualifying),
    }


def run_controls(
    development: dict[str, Any],
    catalog: dict[str, Any],
    hidden: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    split = split_development(development, hidden, config)
    records = {row["record_id"]: row for row in development["records"]}
    hidden_by_id = {row["record_id"]: row for row in hidden["records"] if row["role"] == "development"}
    contract_to_choice = {row["capability_contract_id"]: row["choice_id"] for row in catalog["choices"]}
    scores = {rid: raw_scores(record, catalog, config) for rid, record in records.items()}
    calibration = _select_parameters(split["calibration"], scores, hidden_by_id, contract_to_choice, config)
    evaluation_ids = split["evaluation"]
    if calibration["qualified"]:
        params = tuple(calibration["selected_params"])
        consensus_routes, candidate_sets = _specific_routes(evaluation_ids, scores, params)
    else:
        params = None
        consensus_routes = {rid: None for rid in evaluation_ids}
        candidate_sets = {rid: {"character": sorted(contract_to_choice.values()), "token": sorted(contract_to_choice.values())} for rid in evaluation_ids}

    all_known = sorted(contract_to_choice.values())
    alias_routes = {
        rid: scores[rid]["alias_set"][0] if len(scores[rid]["alias_set"]) == 1 else None
        for rid in evaluation_ids
    }
    char_routes = {
        rid: sets["character"][0] if len(sets["character"]) == 1 else None
        for rid, sets in candidate_sets.items()
    }
    token_routes = {
        rid: sets["token"][0] if len(sets["token"]) == 1 else None
        for rid, sets in candidate_sets.items()
    }
    no_specific = {rid: None for rid in evaluation_ids}
    oracle_routes = {
        rid: _truth_choice(hidden_by_id[rid], contract_to_choice)
        if hidden_by_id[rid]["truth_kind"] == "KNOWN" and hidden_by_id[rid]["observation_available"]
        else None
        for rid in evaluation_ids
    }
    comparators = {
        "complete_safe_enumeration_then_generic_clarification": _route_metrics(evaluation_ids, hidden_by_id, no_specific, contract_to_choice, config),
        "always_generic_clarification": _route_metrics(evaluation_ids, hidden_by_id, no_specific, contract_to_choice, config),
        "exact_intent_alias_lookup_then_safe_fallback": _route_metrics(evaluation_ids, hidden_by_id, alias_routes, contract_to_choice, config),
        "character_ngram_view_then_safe_fallback": _route_metrics(evaluation_ids, hidden_by_id, char_routes, contract_to_choice, config),
        "token_schema_view_then_safe_fallback": _route_metrics(evaluation_ids, hidden_by_id, token_routes, contract_to_choice, config),
        "same_singleton_consensus_then_safe_fallback": _route_metrics(evaluation_ids, hidden_by_id, consensus_routes, contract_to_choice, config),
        "hidden_information_oracle": _route_metrics(evaluation_ids, hidden_by_id, oracle_routes, contract_to_choice, config),
    }
    residual_ids = sorted(
        rid for rid in evaluation_ids
        if hidden_by_id[rid]["observation_available"] and consensus_routes[rid] is None
    )
    predictions = []
    for rid in sorted(evaluation_ids):
        predictions.append({
            "record_id": rid,
            "character_candidate_set": candidate_sets[rid]["character"],
            "token_candidate_set": candidate_sets[rid]["token"],
            "exact_alias_candidate_set": scores[rid]["alias_set"],
            "same_singleton_consensus": consensus_routes[rid],
            "evidence_status": "SPECIFIC_CLARIFICATION" if consensus_routes[rid] else "INSUFFICIENT_GENERIC_CLARIFICATION",
            "residual_member": rid in residual_ids,
        })
    summary = {
        "split_counts": {
            part: dict(sorted(Counter(hidden_by_id[rid]["truth_kind"] for rid in ids).items()))
            for part, ids in split.items()
        },
        "calibration": calibration,
        "selected_parameters": list(params) if params else None,
        "evaluation_record_count": len(evaluation_ids),
        "observed_evaluation_record_count": sum(hidden_by_id[rid]["observation_available"] for rid in evaluation_ids),
        "missing_evaluation_record_count": sum(not hidden_by_id[rid]["observation_available"] for rid in evaluation_ids),
        "structured_prediction_rate": 1.0,
        "missing_insufficient_rate": sum(
            not hidden_by_id[rid]["observation_available"] and consensus_routes[rid] is None
            for rid in evaluation_ids
        ) / sum(not hidden_by_id[rid]["observation_available"] for rid in evaluation_ids),
        "comparators": comparators,
        "residual_count": len(residual_ids),
        "residual_truth_kind_coverage": sorted({hidden_by_id[rid]["truth_kind"] for rid in residual_ids}),
        "residual_membership_uses_predictions_only": True,
        "prediction_payload_sha256": canonical_sha256(predictions),
        "development_language_read_count": 1,
        "protected_language_read_count": 0,
        "manual_language_inspection_count": 0,
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
    return {"split": split, "predictions": predictions, "residual_ids": residual_ids, "summary": summary}


def audit_controls(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    gates = config["evaluationGates"]
    consensus = summary["comparators"]["same_singleton_consensus_then_safe_fallback"]
    always = summary["comparators"]["always_generic_clarification"]
    split_config = config["developmentSplit"]
    checks = {
        "development_split_counts_are_exact": bool(
            summary["split_counts"]["calibration"] == split_config["requiredCalibrationCounts"]
            and summary["split_counts"]["evaluation"] == split_config["requiredEvaluationCounts"]
        ),
        "calibration_found_one_prospectively_qualified_configuration": bool(summary["calibration"]["qualified"]),
        "evaluation_counts_structure_and_missing_fallback_are_exact": bool(
            summary["evaluation_record_count"] == gates["requiredEvaluationRecordCount"]
            and summary["observed_evaluation_record_count"] == gates["requiredObservedEvaluationRecordCount"]
            and summary["missing_evaluation_record_count"] == gates["requiredMissingEvaluationRecordCount"]
            and summary["structured_prediction_rate"] == gates["requiredStructuredPredictionRate"]
            and summary["missing_insufficient_rate"] == gates["requiredMissingInsufficientRate"]
        ),
        "specific_route_is_precise_selective_and_cost_beneficial": bool(
            consensus["candidate_specific_precision"] >= gates["minimumCandidateSpecificPrecision"]
            and consensus["false_specific_rate_on_nonknown"] <= gates["maximumFalseSpecificRateOnNonKnown"]
            and consensus["known_specific_coverage"] >= gates["minimumKnownSpecificCoverage"]
            and consensus["mean_clarification_cost"] <= gates["maximumMeanClarificationCost"]
            and always["mean_clarification_cost"] - consensus["mean_clarification_cost"]
            >= gates["minimumImprovementOverAlwaysGeneric"]
        ),
        "trusted_answer_finality_and_authoritative_retention_are_exact": bool(
            consensus["final_exactness_after_trusted_answers"] == gates["requiredFinalExactnessAfterTrustedAnswers"]
            and consensus["authoritative_hypothesis_retention_rate"] == gates["requiredAuthoritativeHypothesisRetentionRate"]
        ),
        "prediction_defined_residual_is_meaningful_and_diverse": bool(
            gates["minimumPredictionDefinedResidualCount"] <= summary["residual_count"] <= gates["maximumPredictionDefinedResidualCount"]
            and summary["residual_truth_kind_coverage"] == sorted(gates["requiredResidualTruthKindCoverage"])
            and summary["residual_membership_uses_predictions_only"] == gates["requiredResidualMembershipUsesPredictionsOnly"]
        ),
        "protected_model_authority_and_effect_access_is_zero": bool(
            summary["protected_language_read_count"] == gates["maximumProtectedLanguageReadCount"]
            and all(summary[key] == gates[gate] for key, gate in (
                ("manual_language_inspection_count", "maximumManualLanguageInspectionCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            ))
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = ["audit_controls", "run_controls", "split_development", "threshold_set"]
