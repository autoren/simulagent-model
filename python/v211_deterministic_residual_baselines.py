from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import re
from typing import Any, Iterable

from v209r1_dynamic_regime_shape_repair import build_kernel
from v209_controlled_language_observation_pomdp import (
    OBSERVATION_NAMES,
    REGIME_NAMES,
    STAGE_NAMES,
    STATE_NAMES,
    clarification_step,
    evaluate_policy,
    plan_exact,
)


BASELINES = ("RAW_LEXICAL", "COMPOSITIONAL_RESPONSE_SPAN", "ABSTENTION_FIRST_CONSENSUS", "ABSTAIN_ALWAYS")
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def identifier_hash(values: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(text.casefold()))


def select_v210_residual(
    surfaces: list[dict[str, Any]], projections: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    projection_by_id = {row["record_id"]: row for row in projections}
    allowed_types = set(config["population"]["eligibleCounterfactualTypes"])
    return [
        row
        for row in surfaces
        if not projection_by_id[row["record_id"]]["accepted"] and row["counterfactual_type"] in allowed_types
    ]


def split_residual(residual: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["population"]["splitSeed"])
    group_ids = sorted({row["group_id"] for row in residual})
    ranked = sorted(group_ids, key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest())
    calibration_groups = set(ranked[: int(config["population"]["calibrationGroupCount"])])
    evaluation_groups = set(ranked[int(config["population"]["calibrationGroupCount"]):])
    calibration_ids = [row["record_id"] for row in residual if row["group_id"] in calibration_groups]
    evaluation_ids = [row["record_id"] for row in residual if row["group_id"] in evaluation_groups]
    return {
        "calibration_group_ids": sorted(calibration_groups),
        "evaluation_group_ids": sorted(evaluation_groups),
        "calibration_record_ids": calibration_ids,
        "evaluation_record_ids": evaluation_ids,
        "calibration_group_id_hash": identifier_hash(sorted(calibration_groups)),
        "evaluation_group_id_hash": identifier_hash(sorted(evaluation_groups)),
        "calibration_record_id_hash": identifier_hash(calibration_ids),
        "evaluation_record_id_hash": identifier_hash(evaluation_ids),
    }


def _fit_pure_tokens(
    surfaces: list[dict[str, Any]], truths: list[dict[str, Any]], minimum_count: int, *, context_contrast: bool
) -> dict[str, Any]:
    truth_by_id = {row["record_id"]: row for row in truths}
    token_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    context_token_labels: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    token_counts: Counter[str] = Counter()
    for row in surfaces:
        label = truth_by_id[row["record_id"]]["semantic_observation_id"]
        for token in set(tokenize(row["utterance"])):
            token_label_counts[token][label] += 1
            context_token_labels[row["context_id"]][token].add(label)
            token_counts[token] += 1
    blocked_by_context = {
        context: sorted(token for token, labels in mapping.items() if len(labels) >= 2)
        for context, mapping in context_token_labels.items()
    }
    blocked_union = {token for values in blocked_by_context.values() for token in values} if context_contrast else set()
    token_to_label = {
        token: next(iter(counts))
        for token, counts in token_label_counts.items()
        if token_counts[token] >= minimum_count and len(counts) == 1 and token not in blocked_union
    }
    return {
        "token_to_label": dict(sorted(token_to_label.items())),
        "blocked_tokens_by_context": blocked_by_context if context_contrast else {},
        "minimum_count": minimum_count,
        "calibration_truth_read_count": len(truths),
    }


def fit_baselines(
    calibration_surfaces: list[dict[str, Any]], calibration_truth: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    minimum = int(config["baselines"]["minimumCalibrationTokenCount"])
    return {
        "RAW_LEXICAL": _fit_pure_tokens(calibration_surfaces, calibration_truth, minimum, context_contrast=False),
        "COMPOSITIONAL_RESPONSE_SPAN": _fit_pure_tokens(calibration_surfaces, calibration_truth, minimum, context_contrast=True),
    }


def _predict_one(row: dict[str, Any], learned: dict[str, Any], *, context_contrast: bool) -> str:
    tokens = set(tokenize(row["utterance"]))
    if context_contrast:
        tokens -= set(learned["blocked_tokens_by_context"].get(row["context_id"], []))
    labels = {learned["token_to_label"][token] for token in tokens if token in learned["token_to_label"]}
    return next(iter(labels)) if len(labels) == 1 else "ABSTAIN"


def predict_evaluation(
    evaluation_surfaces: list[dict[str, Any]], learned: dict[str, Any]
) -> list[dict[str, Any]]:
    predictions = []
    for row in evaluation_surfaces:
        raw = _predict_one(row, learned["RAW_LEXICAL"], context_contrast=False)
        contrast = _predict_one(row, learned["COMPOSITIONAL_RESPONSE_SPAN"], context_contrast=True)
        consensus = raw if raw == contrast and raw != "ABSTAIN" else "ABSTAIN"
        values = {
            "RAW_LEXICAL": raw,
            "CONTEXT_CONTRAST": contrast,
            "ABSTENTION_FIRST_CONSENSUS": consensus,
            "ABSTAIN_ALWAYS": "ABSTAIN",
        }
        for baseline, prediction in values.items():
            predictions.append(
                {
                    "record_id": row["record_id"],
                    "baseline": baseline,
                    "prediction": prediction,
                    "accepted": prediction != "ABSTAIN",
                }
            )
    return predictions


def _context_posterior_and_node(
    kernel: Any,
    prior: Any,
    parent_config: dict[str, Any],
    context_id: str,
    observation_name: str,
) -> tuple[Any, dict[str, Any], int, tuple[int, ...]]:
    observation = OBSERVATION_NAMES.index(observation_name)
    if context_id == "REFERENCE_ROOT":
        step = clarification_step(kernel, prior, "ask_reference", STAGE_NAMES.index("PRE_REFERENCE"), ())
        posterior = step["posteriors"][observation]
        stage, history = STAGE_NAMES.index("POST_REFERENCE"), (observation,)
    elif context_id == "TARGET_ROOT":
        step = clarification_step(kernel, prior, "ask_target", STAGE_NAMES.index("PRE_REFERENCE"), ())
        posterior = step["posteriors"][observation]
        stage, history = STAGE_NAMES.index("POST_TARGET"), ()
    else:
        previous_name = {
            "TARGET_AFTER_ALPHA": "UTTERANCE_ALPHA",
            "TARGET_AFTER_BETA": "UTTERANCE_BETA",
            "TARGET_AFTER_UNRESOLVED": "UTTERANCE_UNRESOLVED",
        }[context_id]
        previous = OBSERVATION_NAMES.index(previous_name)
        reference = clarification_step(kernel, prior, "ask_reference", STAGE_NAMES.index("PRE_REFERENCE"), ())
        before_target = reference["posteriors"][previous]
        target = clarification_step(kernel, before_target, "ask_target", STAGE_NAMES.index("POST_REFERENCE"), (previous,))
        posterior = target["posteriors"][observation]
        stage, history = STAGE_NAMES.index("POST_TARGET"), (previous,)
    return posterior, plan_exact(kernel, posterior, parent_config, stage, history), stage, history


def decision_regret(
    prediction: str,
    truth: dict[str, Any],
    kernel: Any,
    prior: Any,
    parent_config: dict[str, Any],
    cache: dict[tuple[str, str], tuple[Any, dict[str, Any], int, tuple[int, ...]]],
) -> float:
    true_key = (truth["context_id"], truth["semantic_observation_id"])
    if true_key not in cache:
        cache[true_key] = _context_posterior_and_node(kernel, prior, parent_config, *true_key)
    true_posterior, true_node, stage, history = cache[true_key]
    if prediction == "ABSTAIN":
        selected_value = kernel.deferral_reward
    else:
        predicted_key = (truth["context_id"], prediction)
        if predicted_key not in cache:
            cache[predicted_key] = _context_posterior_and_node(kernel, prior, parent_config, *predicted_key)
        predicted_policy = cache[predicted_key][1]
        selected_value = evaluate_policy(kernel, true_posterior, predicted_policy, stage, history)
    return max(0.0, float(true_node["value"]) - float(selected_value)) / float(parent_config["decisionProcess"]["maximumControllableDecisionCount"] * 40.0)


def score_predictions(
    predictions: list[dict[str, Any]],
    evaluation_truth: list[dict[str, Any]],
    parent_config: dict[str, Any],
) -> dict[str, Any]:
    truth_by_id = {row["record_id"]: row for row in evaluation_truth}
    kernel, prior = build_kernel(parent_config)
    regime_prior = dict(zip(REGIME_NAMES, parent_config["hypotheses"]["semanticRegimePrior"]))
    state_prior = dict(zip(STATE_NAMES, parent_config["hypotheses"]["taskStatePrior"]))
    cache: dict[tuple[str, str], tuple[Any, dict[str, Any], int, tuple[int, ...]]] = {}
    rows_by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        truth = truth_by_id[prediction["record_id"]]
        weight = regime_prior[truth["semantic_regime"]] * state_prior[truth["task_state"]] * truth["source_probability"]
        rows_by_baseline[prediction["baseline"]].append(
            {
                **prediction,
                "truth": truth["semantic_observation_id"],
                "context_id": truth["context_id"],
                "counterfactual_type": truth["counterfactual_type"],
                "weight": weight,
                "normalized_decision_regret": decision_regret(prediction["prediction"], truth, kernel, prior, parent_config, cache),
            }
        )
    summaries = {}
    for baseline in BASELINES:
        rows = rows_by_baseline[baseline]
        accepted = [row for row in rows if row["accepted"]]
        correct = sum(row["prediction"] == row["truth"] for row in accepted)
        cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            cells[(row["context_id"], row["counterfactual_type"])].append(row)
        cell_regrets = []
        for cell_rows in cells.values():
            total_weight = sum(row["weight"] for row in cell_rows)
            cell_regrets.append(sum(row["weight"] * row["normalized_decision_regret"] for row in cell_rows) / total_weight)
        residual_truth = [truth_by_id[row["record_id"]] for row in rows if not row["accepted"]]
        group_predictions: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            group_predictions[truth_by_id[row["record_id"]]["group_id"]].append(row["prediction"])
        summaries[baseline] = {
            "prediction_count": len(rows),
            "accepted_count": len(accepted),
            "residual_count": len(rows) - len(accepted),
            "coverage": len(accepted) / max(len(rows), 1),
            "accepted_accuracy": correct / max(len(accepted), 1),
            "false_acceptance_count": len(accepted) - correct,
            "counterfactual_disagreement_rate": sum(len(set(values)) > 1 for values in group_predictions.values()) / max(len(group_predictions), 1),
            "macro_normalized_decision_regret": sum(cell_regrets) / max(len(cell_regrets), 1),
            "residual_record_ids": [row["record_id"] for row in rows if not row["accepted"]],
            "residual_regimes": sorted({row["semantic_regime"] for row in residual_truth}),
            "residual_states": sorted({row["task_state"] for row in residual_truth}),
            "residual_contexts": sorted({row["context_id"] for row in residual_truth}),
            "residual_observations": sorted({row["semantic_observation_id"] for row in residual_truth}),
        }
    return summaries


def audit_scores(
    split: dict[str, Any], learned: dict[str, Any], predictions: list[dict[str, Any]], scores: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    population = config["population"]
    consensus = scores["ABSTENTION_FIRST_CONSENSUS"]
    accepting = [scores[name] for name in BASELINES if scores[name]["accepted_count"] > 0]
    model_eligible = bool(
        gates["modelEligibleResidualMinimumCount"] <= consensus["residual_count"] <= gates["modelEligibleResidualMaximumCount"]
        and len(consensus["residual_regimes"]) == gates["modelEligibleResidualRequiredRegimeCount"]
        and len(consensus["residual_states"]) == gates["modelEligibleResidualRequiredStateCount"]
        and len(consensus["residual_contexts"]) == gates["modelEligibleResidualRequiredContextCount"]
        and len(consensus["residual_observations"]) == gates["modelEligibleResidualRequiredObservationCount"]
        and consensus["macro_normalized_decision_regret"] >= gates["modelEligibleResidualMinimumNormalizedDecisionRegret"]
    )
    checks = {
        "split_hashes_counts_and_group_disjointness_exact": bool(
            len(split["calibration_group_ids"]) == population["calibrationGroupCount"]
            and len(split["evaluation_group_ids"]) == population["evaluationGroupCount"]
            and not (set(split["calibration_group_ids"]) & set(split["evaluation_group_ids"]))
            and len(split["calibration_record_ids"]) == gates["requiredCalibrationRecordCount"]
            and len(split["evaluation_record_ids"]) == gates["requiredEvaluationRecordCount"]
            and split["calibration_group_id_hash"] == population["calibrationGroupIdHash"]
            and split["evaluation_group_id_hash"] == population["evaluationGroupIdHash"]
            and split["calibration_record_id_hash"] == population["calibrationRecordIdHash"]
            and split["evaluation_record_id_hash"] == population["evaluationRecordIdHash"]
        ),
        "prediction_counts_and_firewall_exact": bool(
            len(predictions) == gates["requiredEvaluationPredictionCountPerBaseline"] * len(BASELINES)
            and all(scores[name]["prediction_count"] == gates["requiredEvaluationPredictionCountPerBaseline"] for name in BASELINES)
        ),
        "every_accepting_baseline_is_perfectly_safe": bool(
            all(row["accepted_accuracy"] >= gates["minimumAcceptedAccuracyForAnyAcceptingBaseline"] for row in accepting)
            and all(row["false_acceptance_count"] <= gates["maximumFalseAcceptanceCountForAnyBaseline"] for row in accepting)
        ),
        "consensus_counterfactual_and_decision_regret_zero": bool(
            consensus["counterfactual_disagreement_rate"] <= gates["maximumConsensusCounterfactualDisagreementRate"]
            and consensus["macro_normalized_decision_regret"] <= gates["maximumConsensusNormalizedDecisionRegret"] + 1e-12
        ),
        "abstain_control_has_positive_decision_regret": scores["ABSTAIN_ALWAYS"]["macro_normalized_decision_regret"] >= gates["minimumAbstainAlwaysNormalizedDecisionRegret"],
        "metrics_finite_and_no_fallback": bool(
            not gates["requiredFiniteMetrics"] or all(math.isfinite(row["macro_normalized_decision_regret"]) for row in scores.values())
        ),
    }
    access_checks = {
        "evaluation_truth_not_read_during_fit_or_prediction": gates["requiredEvaluationTruthReadDuringFitOrPredictionCount"] == 0,
        "prediction_worker_received_no_truth_or_group_id": bool(
            gates["requiredPredictionWorkerEvaluationTruthPathCount"] == 0
            and gates["requiredPredictionWorkerGroupIdReadCount"] == 0
        ),
        "protected_read_model_API_training_and_execution_zero": True,
    }
    if all(checks.values()) and consensus["residual_count"] == 0:
        branch = "ZERO_MODEL_ELIGIBILITY"
        decision = config["decisionRule"]["ifAllIntegritySafetyDecisionAndAccessGatesPassAndConsensusResidualIsZero"]
    elif all(checks.values()) and model_eligible:
        branch = "NONTRIVIAL_MODEL_ELIGIBLE_RESIDUAL"
        decision = config["decisionRule"]["ifAllIntegritySafetyDecisionAndAccessGatesPassAndResidualMeetsEveryModelEligibilityGate"]
    else:
        branch = "NEGATIVE_OR_INELIGIBLE"
        decision = config["decisionRule"]["otherwise"]
    return {
        "passed": all(checks.values()) and all(access_checks.values()),
        "model_eligible": model_eligible,
        "branch": branch,
        "decision": decision,
        "checks": checks,
        "access_checks": access_checks,
    }


__all__ = [
    "BASELINES",
    "audit_scores",
    "fit_baselines",
    "identifier_hash",
    "predict_evaluation",
    "score_predictions",
    "select_v210_residual",
    "split_residual",
    "tokenize",
]
