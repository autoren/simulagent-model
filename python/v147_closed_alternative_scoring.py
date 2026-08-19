from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from v144_local_certificate_realization import evaluate as evaluate_certificate_policy
from v145_finite_certificate_codebook import finalize_code, oracle_code


def alias_mapping(fixture_id: str, codebook: list[dict[str, Any]], aliases: list[str]) -> dict[str, str]:
    codes = [row["certificate_code"] for row in codebook]
    ordered = sorted(codes, key=lambda code: hashlib.sha256(f"{fixture_id}|{code}".encode()).hexdigest())
    if len(ordered) != len(aliases) or len(set(aliases)) != len(aliases):
        raise ValueError("alias and codebook cardinality mismatch")
    return dict(zip(aliases, ordered))


def render_prompt(catalog: dict[str, Any], codebook: list[dict[str, Any]], fixture: dict[str, Any], config: dict[str, Any]) -> str:
    by_code = {row["certificate_code"]: row for row in codebook}
    mapping = alias_mapping(fixture["fixture_id"], codebook, config["scoring"]["aliases"])
    alternatives = []
    for alias in config["scoring"]["aliases"]:
        certificate = by_code[mapping[alias]]["certificate"]
        alternatives.append({"alias": alias, **certificate})
    return json.dumps({
        "instruction": config["prompt"]["instruction"],
        "presented_candidate_under_review": fixture["presented_candidate_choice_id"],
        "choices": catalog["choices"],
        "registered_certificate_alternatives": alternatives,
        "conversation": fixture["conversation"],
        "scored_continuation_contract": "one opaque alias",
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _relative_probabilities(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {key: math.exp(value - maximum) for key, value in scores.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def select_scored_code(
    fixture_id: str,
    scores: Any,
    codebook: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    aliases = config["scoring"]["aliases"]
    mapping = alias_mapping(fixture_id, codebook, aliases)
    invalid = lambda reason: {
        **finalize_code(None, codebook),
        "selection_valid": False,
        "selection_reason": reason,
        "selected_alias": None,
        "selected_certificate_code": None,
        "relative_probabilities_by_code": {},
        "top_relative_probability": 0.0,
        "score_margin": 0.0,
    }
    if not isinstance(scores, dict) or set(scores) != set(aliases):
        return invalid("invalid_score_keys")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in scores.values()
    ):
        return invalid("nonfinite_or_non_numeric_score")
    ordered = sorted(scores.items(), key=lambda row: (-row[1], row[0]))
    if ordered[0][1] == ordered[1][1]:
        return invalid("tied_maximum_score")
    selected_alias = ordered[0][0]
    selected_code = mapping[selected_alias]
    finalized = finalize_code(selected_code, codebook)
    relative_by_alias = _relative_probabilities({key: float(value) for key, value in scores.items()})
    relative_by_code = {mapping[alias]: probability for alias, probability in relative_by_alias.items()}
    return {
        **finalized,
        "selection_valid": True,
        "selection_reason": "unique_maximum_registered_alias",
        "selected_alias": selected_alias,
        "selected_certificate_code": selected_code,
        "relative_probabilities_by_code": relative_by_code,
        "top_relative_probability": relative_by_alias[selected_alias],
        "score_margin": ordered[0][1] - ordered[1][1],
    }


def _calibration_diagnostics(
    completed: dict[str, dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    hidden_by_id = {row["fixture_id"]: row for row in hidden_rows}
    observations = []
    for fixture_id, output in completed.items():
        truth_code = oracle_code(hidden_by_id[fixture_id])
        probabilities = output["relative_probabilities_by_code"]
        top_confidence = output["top_relative_probability"]
        correct = output["selected_certificate_code"] == truth_code
        brier = sum((probability - (1.0 if code == truth_code else 0.0)) ** 2 for code, probability in probabilities.items()) if probabilities else 2.0
        observations.append({"confidence": top_confidence, "correct": correct, "brier": brier})
    bins = []
    ece = 0.0
    for lower_index in range(10):
        lower, upper = lower_index / 10, (lower_index + 1) / 10
        subset = [row for row in observations if lower <= row["confidence"] < upper or (upper == 1.0 and row["confidence"] == 1.0)]
        if not subset:
            continue
        accuracy = sum(row["correct"] for row in subset) / len(subset)
        confidence = sum(row["confidence"] for row in subset) / len(subset)
        ece += len(subset) / len(observations) * abs(accuracy - confidence)
        bins.append({"lower": lower, "upper": upper, "count": len(subset), "accuracy": accuracy, "mean_relative_confidence": confidence})
    risk_coverage = {}
    for threshold in (0.5, 0.7, 0.9):
        covered = [row for row in observations if row["confidence"] >= threshold]
        risk_coverage[f"{threshold:.1f}"] = {"coverage": len(covered) / len(observations), "selective_accuracy": sum(row["correct"] for row in covered) / len(covered) if covered else None}
    return {"candidate_relative_ECE_10_bin": ece, "multiclass_brier_score": sum(row["brier"] for row in observations) / len(observations), "calibration_bins": bins, "risk_coverage": risk_coverage, "relative_scores_are_not_claimed_calibrated": True}


def evaluate(
    completed: dict[str, dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    catalog: dict[str, Any],
    v136_config: dict[str, Any],
    access: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    adapted = {}
    for fixture_id, row in completed.items():
        adapted[fixture_id] = {
            **row,
            "certificate_valid": row["code_valid"],
            "thinking_trace_present": False,
            "maximum_new_tokens_hit": False,
            "generated_token_count": 0,
            "generation_seconds": row["scoring_seconds"],
            "validation_reason": row["selection_reason"],
        }
    base_access = {**access, "model_generation_count": 0, "maximum_generation_count_per_fixture": 0, "test_fixture_model_generation_count": 0}
    base = evaluate_certificate_policy(adapted, hidden_rows, catalog, v136_config, base_access, config)
    extra_access = config["accessGates"]
    scoring_access = {
        "tokenizer_load_budget": access["tokenizer_load_count"] <= extra_access["maximumTokenizerLoadCount"],
        "scoring_fixture_budget": access["model_scoring_fixture_count"] <= extra_access["maximumModelScoringFixtureCount"],
        "candidate_sequence_budget": access["candidate_sequence_score_count"] <= extra_access["maximumCandidateSequenceScoreCount"],
        "zero_test_scores": access["test_fixture_score_count"] <= extra_access["maximumTestFixtureScoreCount"],
    }
    diagnostics = _calibration_diagnostics(completed, hidden_rows)
    qualified = all(base["qualification_gates"].values()) and all(base["access_gates"].values()) and all(scoring_access.values())
    return {
        **base,
        "access_gates": {**base["access_gates"], **scoring_access},
        "calibration_diagnostics": diagnostics,
        "qualified": qualified,
        "decision": config["decisionRule"]["ifEveryQualificationAndAccessGatePasses"] if qualified else config["decisionRule"]["otherwise"],
    }


__all__ = ["alias_mapping", "evaluate", "render_prompt", "select_scored_code"]
