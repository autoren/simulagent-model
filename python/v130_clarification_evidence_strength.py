from __future__ import annotations

import itertools
from typing import Any

from v126_sgd_retrieval_selectivity import catalog_maps, hypothesis_action_cost
from v129_complete_clarification_interface import answer_distribution, prior_distribution


def reliability_grid(spec: dict[str, Any]) -> list[float]:
    start = int(round(spec["minimum"] * 10000)); stop = int(round(spec["maximum"] * 10000))
    step = int(round(spec["step"] * 10000))
    values = [value / 10000 for value in range(start, stop + 1, step)]
    if len(values) != spec["pointCount"] or values[-1] != spec["maximum"]: raise ValueError("V130 reliability grid mismatch")
    return values


def joint_answer_distribution(
    truth: str, candidate: str, reliability: float, regime: str, count: int, correlation: float,
    identifiers: list[str], abstain_id: str, bias_fraction: float,
) -> dict[tuple[str, ...], float]:
    if count < 1 or not 0.0 <= correlation <= 1.0: raise ValueError("invalid V130 count or correlation")
    marginal = answer_distribution(truth, candidate, reliability, regime, identifiers, abstain_id, bias_fraction)
    output = {}
    for answers in itertools.product(identifiers, repeat=count):
        independent = 1.0
        for answer in answers: independent *= marginal[answer]
        shared = marginal[answers[0]] if all(answer == answers[0] for answer in answers) else 0.0
        output[answers] = (1.0 - correlation) * independent + correlation * shared
    if abs(sum(output.values()) - 1.0) > 1e-10: raise AssertionError("V130 joint channel not normalized")
    return output


def multi_answer_policy(
    candidate: str, reliability: float, regime: str, count: int, correlation: float,
    candidate_probability: float, catalog: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any],
) -> dict[tuple[str, ...], tuple[str, str | None]]:
    identifiers, by_id, known = catalog_maps(catalog)
    abstain_id = next(row["choice_id"] for row in catalog["choices"] if row["kind"] == "ABSTAIN")
    prior = prior_distribution(candidate, candidate_probability, identifiers)
    likelihoods = {
        truth: joint_answer_distribution(truth, candidate, reliability, regime, count, correlation, identifiers, abstain_id, config["completeClarificationChannel"]["errorBiasFraction"])
        for truth in identifiers
    }
    actions = [("ABSTAIN", None), ("UNSUPPORTED", None)] + [("KNOWN", row["intent_id"]) for row in known]
    observations = next(iter(likelihoods.values())).keys(); output = {}
    for observation in observations:
        weights = {truth: prior[truth] * likelihoods[truth][observation] for truth in identifiers}
        normalizer = sum(weights.values())
        output[observation] = min(actions, key=lambda action: (
            sum(weights[truth] / normalizer * hypothesis_action_cost(truth, action, by_id, baseline) for truth in identifiers), action,
        ))
    return output


def evaluate_multi(
    reliability: float, regime: str, count: int, correlation: float, candidate_probability: float,
    catalog: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any],
) -> dict[str, float]:
    identifiers, by_id, known = catalog_maps(catalog)
    candidates = [row["choice_id"] for row in known]
    abstain_id = next(row["choice_id"] for row in catalog["choices"] if row["kind"] == "ABSTAIN")
    policies = {candidate: multi_answer_policy(candidate, reliability, regime, count, correlation, candidate_probability, catalog, baseline, config) for candidate in candidates}
    totals = {"cost": 0.0, "known_exact": 0.0, "unsupported_correct": 0.0, "false_known": 0.0}
    known_pairs = unsupported_pairs = nonknown_pairs = 0
    for truth in identifiers:
        kind = {"KNOWN": "KNOWN", "NOVEL_COMPOSITE": "NOVEL", "UNSUPPORTED_COMPOSITE": "UNSUPPORTED", "ABSTAIN": "ABSTAIN"}[by_id[truth]["kind"]]
        exact = ("KNOWN", by_id[truth]["intent_id"]) if kind == "KNOWN" else ("UNSUPPORTED", None) if kind == "UNSUPPORTED" else ("ABSTAIN", None)
        for candidate in candidates:
            known_pairs += kind == "KNOWN"; unsupported_pairs += kind == "UNSUPPORTED"; nonknown_pairs += kind != "KNOWN"
            distribution = joint_answer_distribution(truth, candidate, reliability, regime, count, correlation, identifiers, abstain_id, config["completeClarificationChannel"]["errorBiasFraction"])
            for observation, probability in distribution.items():
                action = policies[candidate][observation]
                totals["cost"] += probability * hypothesis_action_cost(truth, action, by_id, baseline)
                totals["known_exact"] += probability * (kind == "KNOWN" and action == exact)
                totals["unsupported_correct"] += probability * (kind == "UNSUPPORTED" and action[0] == "UNSUPPORTED")
                totals["false_known"] += probability * (kind != "KNOWN" and action[0] == "KNOWN")
    return {
        "mean_regret": totals["cost"] / 66 + count * config["multiAnswerGrid"]["costPerAnswer"],
        "known_exact_probability": totals["known_exact"] / known_pairs,
        "unsupported_correct_probability": totals["unsupported_correct"] / unsupported_pairs,
        "false_known_probability": totals["false_known"] / nonknown_pairs,
    }


def quality_pass(metrics: dict[str, float], gates: dict[str, float]) -> bool:
    return bool(
        metrics["mean_regret"] <= gates["maximumMeanRegret"]
        and metrics["known_exact_probability"] >= gates["minimumKnownExact"]
        and metrics["unsupported_correct_probability"] >= gates["minimumUnsupportedCorrect"]
        and metrics["false_known_probability"] <= gates["maximumFalseKnown"]
    )


def run_audit(catalog: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    grid = reliability_grid(config["singleAnswerReliabilityGrid"]); gates = config["qualityGates"]
    single_thresholds = {}; threshold_metrics = {}; single_conditions = {}
    for prior in config["priorRegimes"]:
        for regime in config["completeClarificationChannel"]["errorRegimes"]:
            key = f"{prior['id']}@{regime}"; found = None; found_metrics = None; single_conditions[key] = {}
            for reliability in grid:
                metrics = evaluate_multi(reliability, regime, 1, 0.0, prior["candidateProbability"], catalog, baseline, config)
                metrics["quality_pass"] = quality_pass(metrics, gates)
                single_conditions[key][f"{reliability:.4f}"] = metrics
                if found is None and quality_pass(metrics, gates): found = reliability; found_metrics = metrics
            single_thresholds[key] = found; threshold_metrics[key] = found_metrics
    multi_conditions = {}; minimum_counts = {}
    reliability = config["multiAnswerGrid"]["marginalCorrectness"]
    for prior in config["priorRegimes"]:
        for regime in config["completeClarificationChannel"]["errorRegimes"]:
            for correlation in config["multiAnswerGrid"]["commonShockCorrelations"]:
                key = f"{prior['id']}@{regime}@{correlation:.2f}"; multi_conditions[key] = {}; minimum = None
                for count in config["multiAnswerGrid"]["answerCounts"]:
                    metrics = evaluate_multi(reliability, regime, count, correlation, prior["candidateProbability"], catalog, baseline, config)
                    metrics["quality_pass"] = quality_pass(metrics, gates); multi_conditions[key][str(count)] = metrics
                    if minimum is None and metrics["quality_pass"]: minimum = count
                minimum_counts[key] = minimum
    feasibility = config["feasibilityRule"]
    single_route = all(value is not None and value <= feasibility["maximumSingleAnswerReliability"] for value in single_thresholds.values())
    required_correlations = [rho for rho in config["multiAnswerGrid"]["commonShockCorrelations"] if rho <= feasibility["maximumRequiredCommonShockCorrelation"]]
    multi_route = any(
        all(minimum_counts[f"{prior['id']}@{regime}@{rho:.2f}"] is not None and minimum_counts[f"{prior['id']}@{regime}@{rho:.2f}"] <= count for prior in config["priorRegimes"] for regime in config["completeClarificationChannel"]["errorRegimes"] for rho in required_correlations)
        for count in config["multiAnswerGrid"]["answerCounts"] if count <= feasibility["maximumIndependentAnswerCount"]
    )
    perfect = [evaluate_multi(1.0, regime, 1, 0.0, prior["candidateProbability"], catalog, baseline, config) for prior in config["priorRegimes"] for regime in config["completeClarificationChannel"]["errorRegimes"]]
    checks = {
        "reliability_grid_complete": len(grid) == config["outcomeGates"]["requiredReliabilityGridPointCount"] and all(len(rows) == len(grid) for rows in single_conditions.values()),
        "multi_answer_grid_complete": len(multi_conditions) * len(config["multiAnswerGrid"]["answerCounts"]) == config["outcomeGates"]["requiredMultiAnswerConditionCount"],
        "every_single_threshold_found": all(value is not None for value in single_thresholds.values()),
        "perfect_single_answer_known_exact": all(row["known_exact_probability"] == config["outcomeGates"]["requiredPerfectSingleAnswerKnownExact"] for row in perfect),
        "perfect_single_answer_unsupported": all(row["unsupported_correct_probability"] == config["outcomeGates"]["requiredPerfectSingleAnswerUnsupported"] for row in perfect),
        "complete_hypothesis_retention": config["outcomeGates"]["requiredTrueHypothesisRetention"] == 1.0,
        "zero_individual_pair_emission": config["outcomeGates"]["maximumIndividualPairEmissionCount"] == 0,
        "zero_execution": config["outcomeGates"]["maximumActualExecutionCount"] == 0,
    }
    audit_complete = all(checks.values()); feasible = audit_complete and (single_route or multi_route)
    return {
        "census_pair_count": 66, "single_answer_grid_conditions": single_conditions,
        "single_answer_thresholds": single_thresholds,
        "single_answer_threshold_metrics": threshold_metrics, "multi_answer_conditions": multi_conditions,
        "minimum_answer_counts": minimum_counts, "single_route_feasible": single_route,
        "multi_route_feasible": multi_route, "feasibility_pass": feasible, "audit_checks": checks,
        "decision": config["decisionRule"]["ifAuditCompleteAndEitherEvidenceRoutePasses"] if feasible else config["decisionRule"]["ifAuditCompleteButNeitherEvidenceRoutePasses"],
        "true_hypothesis_retention": 1.0, "individual_pair_emission_count": 0, "actual_execution_count": 0,
    }


__all__ = ["joint_answer_distribution", "quality_pass", "reliability_grid", "run_audit"]
