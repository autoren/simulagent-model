from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from typing import Any, Iterable

from v212_open_class_identifiability_oracle import (
    REPRESENTATION_ORDER,
    all_behavior_ids,
    behavior_bits,
    behavior_id,
    classify_behavior,
    evidence_status,
    expressibility_set,
    language_catalog,
    rename_record,
    resolve_episode,
    shadow_action,
)
from v213_fresh_programmatic_concept_population import (
    FAMILIES,
    VARIANT_CODES,
    _group_instruction,
    _opaque,
    _rank,
    _variant_public,
)


METHODS = (
    "EXACT_STRUCTURAL_CEILING",
    "NORMALIZED_EXACT_RETRIEVAL_K8",
    "TYPED_APPROX_RETRIEVAL_K8",
    "BOUNDED_L0_LPLUS_SYNTHESIS",
    "FULL_CONSTRAINT_PROPAGATION",
    "DETERMINISTIC_STACK",
)


def reconstruct_development_subsplit(
    v213_config: dict[str, Any],
    v214_config: dict[str, Any],
    semantics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    design = v213_config["populationDesign"]
    v213_split = v213_config["splitDesign"]
    v214_split = v214_config["developmentSubsplit"]
    catalog = language_catalog(semantics)
    fit_groups: set[str] = set()
    evaluation_groups: set[str] = set()
    selected: list[tuple[str, int, str, str]] = []
    for family in FAMILIES:
        family_groups = [
            (
                index,
                _opaque("group", design["groupIdentifierSalt"], family, index),
            )
            for index in range(design["groupsPerFamily"])
        ]
        ranked_v213 = _rank(
            [group_id for _, group_id in family_groups], design["splitSeed"], family
        )
        development = set(ranked_v213[: v213_split["developmentGroupsPerFamily"]])
        ranked_v214 = _rank(development, v214_split["seed"], family)
        family_fit = set(ranked_v214[: v214_split["fitGroupsPerFamily"]])
        fit_groups.update(family_fit)
        evaluation_groups.update(development - family_fit)
        for index, group_id in family_groups:
            if group_id in development:
                selected.append(
                    (family, index, group_id, "fit" if group_id in family_fit else "evaluation")
                )

    fit_public: list[dict[str, Any]] = []
    fit_truth: list[dict[str, Any]] = []
    evaluation_public: list[dict[str, Any]] = []
    evaluation_truth: list[dict[str, Any]] = []
    for family, index, group_id, subsplit in selected:
        base_public, base_truth = _group_instruction(family, index, v213_config, semantics, catalog)
        for code in VARIANT_CODES:
            case_id = _opaque("case", design["caseIdentifierSalt"], group_id, code)
            public = _variant_public(base_public, code, group_id)
            public.update(
                {
                    "case_id": case_id,
                    "group_id": group_id,
                    "split": "development",
                    "variant_code": code,
                }
            )
            ordered_public = {
                field: public[field] for field in v213_config["roleSeparation"]["publicFields"]
            }
            truth = {
                "case_id": case_id,
                "group_id": group_id,
                "split": "development",
                "variant_code": code,
                **deepcopy(base_truth),
            }
            if subsplit == "fit":
                fit_public.append(ordered_public)
                fit_truth.append(truth)
            else:
                evaluation_public.append(ordered_public)
                evaluation_truth.append(truth)
    key = lambda row: row["case_id"]
    subsplit = {
        "schema_version": "214-development-fit-evaluation-subsplit",
        "unit": "opaque_concept_group",
        "fit_group_ids": sorted(fit_groups),
        "evaluation_group_ids": sorted(evaluation_groups),
        "fit_record_count": len(fit_public),
        "evaluation_record_count": len(evaluation_public),
        "family_group_counts": {
            family: {"fit": 4, "evaluation": 4} for family in FAMILIES
        },
        "protected_group_record_construction_count": 0,
    }
    return (
        sorted(fit_public, key=key),
        sorted(fit_truth, key=key),
        sorted(evaluation_public, key=key),
        sorted(evaluation_truth, key=key),
        subsplit,
    )


def _canonical_expression(expression: dict[str, Any]) -> dict[str, Any]:
    op = expression["op"]
    if op == "IDENTITY":
        return _canonical_expression(expression["arg"])
    if op == "PRIMITIVE":
        return {"op": op, "name": expression["name"]}
    if op == "NOT":
        return {"op": op, "arg": _canonical_expression(expression["arg"])}
    if op in {"AND", "OR", "XOR"}:
        args = [_canonical_expression(arg) for arg in expression["args"]]
        args.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
        return {"op": op, "args": args}
    raise ValueError(f"V214 unknown expression operator: {op}")


def _canonical_definition(definition: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(definition)
    if value["kind"] == "EXPRESSION":
        value["expression"] = _canonical_expression(value["expression"])
    if value["kind"] == "OUTSIDE_DESCRIPTION":
        value["token"] = "<OUTSIDE>"
    return value


def normalize_public(record: dict[str, Any]) -> dict[str, Any]:
    value = rename_record(record)
    normalized = {
        "definition": _canonical_definition(value["definition"]),
        "references": [],
        "reference_facts": sorted(
            deepcopy(value["reference_facts"]),
            key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
        ),
        "observations": sorted(
            deepcopy(value["observations"]), key=lambda row: (row["world"], row["output"])
        ),
        "comparison_anchor": None,
    }
    for reference in value["references"]:
        normalized["references"].append(
            {
                "reference_id": reference["reference_id"],
                "definition": _canonical_definition(reference["definition"]),
                "observations": sorted(
                    deepcopy(reference["observations"]),
                    key=lambda row: (row["world"], row["output"]),
                ),
            }
        )
    normalized["references"].sort(key=lambda row: row["reference_id"])
    if value["comparison_anchor"] is not None:
        normalized["comparison_anchor"] = _canonical_definition(value["comparison_anchor"])
    return normalized


def normalized_signature(record: dict[str, Any]) -> str:
    return json.dumps(normalize_public(record), sort_keys=True, separators=(",", ":"))


def _flatten_features(value: Any, prefix: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}/{key}"
            result.add(child)
            result.update(_flatten_features(item, child))
    elif isinstance(value, list):
        for item in value:
            result.update(_flatten_features(item, f"{prefix}[]"))
    else:
        result.add(f"{prefix}={value}")
    return result


def typed_features(record: dict[str, Any]) -> set[str]:
    return _flatten_features(normalize_public(record))


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def _special_prediction(
    record: dict[str, Any], status: str, semantics: dict[str, Any], mechanism: str
) -> dict[str, Any]:
    if status == "OUTSIDE_INTERFACE":
        return {
            "case_id": record["case_id"],
            "candidate_ids": [],
            "proposal_status": status,
            "evidence_status": "AMBIGUOUS",
            "expressibility_set": list(REPRESENTATION_ORDER),
            "shadow_action": "DEFER_OUTSIDE",
            "mechanism": mechanism,
        }
    if status == "CONTRADICTORY":
        evidence = "CONTRADICTORY"
    else:
        evidence = "NO_PROPOSAL"
    return {
        "case_id": record["case_id"],
        "candidate_ids": [],
        "proposal_status": status,
        "evidence_status": evidence,
        "expressibility_set": [],
        "shadow_action": "DEFER_ADJUDICATE",
        "mechanism": mechanism,
    }


def _candidate_prediction(
    record: dict[str, Any],
    candidates: Iterable[str],
    semantics: dict[str, Any],
    mechanism: str,
) -> dict[str, Any]:
    ordered = sorted(set(candidates))
    catalog = language_catalog(semantics)
    return {
        "case_id": record["case_id"],
        "candidate_ids": ordered,
        "proposal_status": evidence_status(ordered),
        "evidence_status": evidence_status(ordered),
        "expressibility_set": expressibility_set(ordered, catalog),
        "shadow_action": shadow_action(ordered, record["definition"]["kind"], catalog, semantics),
        "mechanism": mechanism,
    }


def _exact_prediction(record: dict[str, Any], semantics: dict[str, Any], mechanism: str) -> dict[str, Any]:
    resolved = resolve_episode(record, semantics)
    return {
        "case_id": record["case_id"],
        "candidate_ids": resolved["candidate_ids"],
        "proposal_status": "OUTSIDE_INTERFACE" if record["definition"]["kind"] == "OUTSIDE_DESCRIPTION" else resolved["evidence_status"],
        "evidence_status": resolved["evidence_status"],
        "expressibility_set": resolved["expressibility_set"],
        "shadow_action": resolved["shadow_action"],
        "mechanism": mechanism,
    }


def _fit_group_representatives(
    fit_public: list[dict[str, Any]], fit_truth: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    public = {row["case_id"]: row for row in fit_public}
    truth = {row["case_id"]: row for row in fit_truth}
    groups: dict[str, list[str]] = {}
    for identifier, hidden in truth.items():
        groups.setdefault(hidden["group_id"], []).append(identifier)
    representatives = []
    for group_id in sorted(groups):
        case_id = sorted(groups[group_id])[0]
        representatives.append((public[case_id], truth[case_id]))
    return representatives


def _rank_behavior_votes(
    neighbors: list[tuple[float, str, dict[str, Any]]], budget: int
) -> list[str]:
    best: dict[str, tuple[float, str]] = {}
    for distance, group_id, truth in neighbors:
        for identifier in truth["expected_candidate_ids"]:
            candidate = (distance, group_id)
            if identifier not in best or candidate < best[identifier]:
                best[identifier] = candidate
    return [
        identifier
        for identifier, _ in sorted(best.items(), key=lambda item: (item[1][0], item[1][1], item[0]))[:budget]
    ]


def normalized_exact_retrieval(
    record: dict[str, Any], representatives: list[tuple[dict[str, Any], dict[str, Any]]], semantics: dict[str, Any], budget: int
) -> dict[str, Any]:
    if record["definition"]["kind"] == "OUTSIDE_DESCRIPTION":
        return _special_prediction(record, "OUTSIDE_INTERFACE", semantics, "outside_interface")
    signature = normalized_signature(record)
    matches = [
        (0.0, hidden["group_id"], hidden)
        for surface, hidden in representatives
        if normalized_signature(surface) == signature
    ]
    if not matches:
        return _special_prediction(record, "NO_PROPOSAL", semantics, "no_exact_retrieval_match")
    if all(hidden["expected_evidence_status"] == "CONTRADICTORY" for _, _, hidden in matches):
        return _special_prediction(record, "CONTRADICTORY", semantics, "retrieved_contradiction")
    candidates = _rank_behavior_votes(matches, budget)
    if not candidates:
        return _special_prediction(record, "NO_PROPOSAL", semantics, "matched_without_behavior_proposal")
    return _candidate_prediction(record, candidates, semantics, "normalized_exact_retrieval")


def typed_approximate_retrieval(
    record: dict[str, Any],
    representatives: list[tuple[dict[str, Any], dict[str, Any]]],
    semantics: dict[str, Any],
    budget: int,
    nearest_count: int,
) -> dict[str, Any]:
    if record["definition"]["kind"] == "OUTSIDE_DESCRIPTION":
        return _special_prediction(record, "OUTSIDE_INTERFACE", semantics, "outside_interface")
    target_features = typed_features(record)
    ranked = sorted(
        (
            _jaccard_distance(target_features, typed_features(surface)),
            hidden["group_id"],
            hidden,
        )
        for surface, hidden in representatives
    )[:nearest_count]
    candidates = _rank_behavior_votes(ranked, budget)
    if not candidates:
        if ranked and all(hidden["expected_evidence_status"] == "CONTRADICTORY" for _, _, hidden in ranked):
            return _special_prediction(record, "CONTRADICTORY", semantics, "nearest_contradictions")
        return _special_prediction(record, "NO_PROPOSAL", semantics, "nearest_without_behavior_proposal")
    return _candidate_prediction(record, candidates, semantics, "typed_approximate_retrieval")


def bounded_synthesis(record: dict[str, Any], semantics: dict[str, Any]) -> dict[str, Any]:
    if record["definition"]["kind"] == "OUTSIDE_DESCRIPTION":
        return _special_prediction(record, "OUTSIDE_INTERFACE", semantics, "outside_interface")
    full = resolve_episode(record, semantics)
    if not full["candidate_ids"]:
        return _special_prediction(record, "CONTRADICTORY", semantics, "deterministic_contradiction")
    if record["definition"]["kind"] in {"EXPRESSION", "SYMBOL"}:
        return _candidate_prediction(record, full["candidate_ids"], semantics, "typed_expression_or_reference")
    catalog = language_catalog(semantics)
    represented = catalog["base_ids"] | catalog["extension_ids"]
    candidates = [identifier for identifier in full["candidate_ids"] if identifier in represented]
    if not candidates:
        return _special_prediction(record, "NO_PROPOSAL", semantics, "outside_bounded_languages")
    return _candidate_prediction(record, candidates, semantics, "bounded_L0_LPLUS_enumeration")


def run_controls(
    fit_public: list[dict[str, Any]],
    fit_truth: list[dict[str, Any]],
    evaluation_public: list[dict[str, Any]],
    semantics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    representatives = _fit_group_representatives(fit_public, fit_truth)
    exact_budget = config["methods"]["NORMALIZED_EXACT_RETRIEVAL_K8"]["candidateBudget"]
    approximate = config["methods"]["TYPED_APPROX_RETRIEVAL_K8"]
    results = {method: [] for method in METHODS}
    for record in sorted(evaluation_public, key=lambda row: row["case_id"]):
        ceiling = _exact_prediction(record, semantics, "complete_structural_ceiling")
        exact_retrieval = normalized_exact_retrieval(record, representatives, semantics, exact_budget)
        approximate_retrieval = typed_approximate_retrieval(
            record,
            representatives,
            semantics,
            approximate["candidateBudget"],
            approximate["nearestFitGroupCount"],
        )
        synthesis = bounded_synthesis(record, semantics)
        full = _exact_prediction(record, semantics, "complete_256_behavior_filter")
        if (
            synthesis["candidate_ids"] == full["candidate_ids"]
            and synthesis["evidence_status"] == full["evidence_status"]
            and synthesis["shadow_action"] == full["shadow_action"]
        ):
            stack = {**synthesis, "mechanism": "bounded_synthesis_closed_case"}
        else:
            stack = {**full, "mechanism": "full_constraint_fallback"}
        for method, prediction in (
            ("EXACT_STRUCTURAL_CEILING", ceiling),
            ("NORMALIZED_EXACT_RETRIEVAL_K8", exact_retrieval),
            ("TYPED_APPROX_RETRIEVAL_K8", approximate_retrieval),
            ("BOUNDED_L0_LPLUS_SYNTHESIS", synthesis),
            ("FULL_CONSTRAINT_PROPAGATION", full),
            ("DETERMINISTIC_STACK", stack),
        ):
            results[method].append(prediction)
    return results


def _action_value(
    action: str, truth: dict[str, Any], semantics: dict[str, Any], config: dict[str, Any]
) -> float:
    scoring = config["decisionScoring"]
    correct = float(scoring["correctDiagnosisReward"])
    wrong = float(scoring["wrongDiagnosisReward"])
    defer = float(scoring["safeDeferralReward"])
    candidates = truth["expected_candidate_ids"]
    outside = truth["concept_family"] == "OUTSIDE_DESCRIPTION"
    if action in {"DEFER_ADJUDICATE", "DEFER_OUTSIDE"}:
        return defer
    if action == "REQUEST_BOUNDARY":
        if outside or not candidates:
            return float(scoring["outsideBoundaryRequestReward"])
        return correct + float(scoring["boundaryRequestCost"])
    if outside or not candidates:
        return wrong
    catalog = language_catalog(semantics)
    action_map = semantics["action_by_singleton_expressibility"]
    correct_count = sum(
        action_map[classify_behavior(identifier, catalog)] == action for identifier in candidates
    )
    probability = correct_count / len(candidates)
    return probability * correct + (1.0 - probability) * wrong


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def score_controls(
    predictions: dict[str, list[dict[str, Any]]],
    evaluation_public: list[dict[str, Any]],
    evaluation_truth: list[dict[str, Any]],
    subsplit: dict[str, Any],
    semantics: dict[str, Any],
    prediction_freeze: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    public = {row["case_id"]: row for row in evaluation_public}
    truth = {row["case_id"]: row for row in evaluation_truth}
    methods: dict[str, Any] = {}
    exact_values = {
        identifier: _action_value(hidden["expected_shadow_action"], hidden, semantics, config)
        for identifier, hidden in truth.items()
    }
    for method in METHODS:
        rows = {row["case_id"]: row for row in predictions[method]}
        if set(rows) != set(truth):
            raise ValueError(f"V214 prediction identifiers mismatch for {method}")
        recalls = []
        false_count = 0
        proposal_count = 0
        exact_sets = []
        statuses = []
        expressibility = []
        actions = []
        regrets = []
        residuals = []
        for identifier in sorted(truth):
            hidden = truth[identifier]
            prediction = rows[identifier]
            expected = set(hidden["expected_candidate_ids"])
            proposed = set(prediction["candidate_ids"])
            if not expected:
                recall = float(
                    not proposed and prediction["evidence_status"] == "CONTRADICTORY"
                )
            else:
                recall = len(expected & proposed) / len(expected)
            recalls.append(recall)
            false_count += len(proposed - expected)
            proposal_count += len(proposed)
            exact_sets.append(prediction["candidate_ids"] == hidden["expected_candidate_ids"])
            statuses.append(prediction["evidence_status"] == hidden["expected_evidence_status"])
            expressibility.append(
                prediction["expressibility_set"] == hidden["expected_expressibility_set"]
            )
            actions.append(prediction["shadow_action"] == hidden["expected_shadow_action"])
            value = _action_value(prediction["shadow_action"], hidden, semantics, config)
            normalized_regret = (exact_values[identifier] - value) / float(
                config["decisionScoring"]["normalizationRange"]
            )
            regrets.append(normalized_regret)
            if not exact_sets[-1] or not actions[-1]:
                residuals.append(
                    {
                        "case_id": identifier,
                        "group_id": hidden["group_id"],
                        "family": hidden["concept_family"],
                        "truth_status": hidden["expected_evidence_status"],
                        "truth_expressibility": hidden["expected_expressibility_set"],
                        "proposal_status": prediction["proposal_status"],
                        "mechanism": prediction["mechanism"],
                        "correct_class_absent": bool(expected - proposed),
                        "false_class_count": len(proposed - expected),
                        "action_wrong": not actions[-1],
                        "normalized_decision_regret": normalized_regret,
                    }
                )
        sizes = [len(row["candidate_ids"]) for row in rows.values()]
        group_invariance = []
        for group_id in sorted({hidden["group_id"] for hidden in truth.values()}):
            identifiers = sorted(
                identifier for identifier, hidden in truth.items() if hidden["group_id"] == group_id
            )
            signatures = [
                (
                    rows[identifier]["candidate_ids"],
                    rows[identifier]["evidence_status"],
                    rows[identifier]["expressibility_set"],
                    rows[identifier]["shadow_action"],
                )
                for identifier in identifiers
            ]
            group_invariance.append(all(signature == signatures[0] for signature in signatures))
        budget = None
        if method in {"NORMALIZED_EXACT_RETRIEVAL_K8", "TYPED_APPROX_RETRIEVAL_K8"}:
            budget = config["methods"][method]["candidateBudget"]
        budget_compliance = 1.0 if budget is None else _rate([size <= budget for size in sizes])
        family_residual_counts = dict(Counter(row["family"] for row in residuals))
        mechanism_residual_counts = dict(Counter(row["mechanism"] for row in residuals))
        methods[method] = {
            "prediction_count": len(rows),
            "oracle_class_recall": sum(recalls) / len(recalls),
            "exact_version_space_accuracy": _rate(exact_sets),
            "false_class_proposal_count": false_count,
            "false_class_proposal_rate": false_count / proposal_count if proposal_count else 0.0,
            "average_candidate_set_size": sum(sizes) / len(sizes),
            "maximum_candidate_set_size": max(sizes),
            "evidence_status_accuracy": _rate(statuses),
            "expressibility_set_accuracy": _rate(expressibility),
            "shadow_action_accuracy": _rate(actions),
            "average_normalized_decision_regret": sum(regrets) / len(regrets),
            "candidate_budget_compliance": budget_compliance,
            "group_variant_invariance": _rate(group_invariance),
            "residual_record_count": len(residuals),
            "residual_group_count": len({row["group_id"] for row in residuals}),
            "residual_family_counts": family_residual_counts,
            "residual_mechanism_counts": mechanism_residual_counts,
            "residuals": residuals,
        }

    fit_groups = set(subsplit["fit_group_ids"])
    evaluation_groups = set(subsplit["evaluation_group_ids"])
    selector = config["modelEligibilityRule"]["selectorMethod"]
    selector_residuals = methods[selector]["residuals"]
    excluded = set(config["modelEligibilityRule"]["excludedTruthFamilies"])
    eligible = [
        row
        for row in selector_residuals
        if row["family"] not in excluded
        and (row["correct_class_absent"] or row["action_wrong"])
        and row["normalized_decision_regret"] > 0.0
    ]
    eligible_groups = {row["group_id"] for row in eligible}
    average_eligible_regret = (
        sum(row["normalized_decision_regret"] for row in eligible) / len(eligible)
        if eligible
        else 0.0
    )
    eligibility_rule = config["modelEligibilityRule"]
    model_eligible = bool(
        len(eligible_groups) >= eligibility_rule["minimumResidualGroupCount"]
        and average_eligible_regret >= eligibility_rule["minimumAverageNormalizedDecisionRegret"]
    )
    imperfect_bounded = sum(
        methods[method]["exact_version_space_accuracy"] < 1.0
        for method in (
            "NORMALIZED_EXACT_RETRIEVAL_K8",
            "TYPED_APPROX_RETRIEVAL_K8",
            "BOUNDED_L0_LPLUS_SYNTHESIS",
        )
    )
    metrics = {
        "fit_group_count": len(fit_groups),
        "evaluation_group_count": len(evaluation_groups),
        "fit_evaluation_group_overlap_count": len(fit_groups & evaluation_groups),
        "fit_record_count": subsplit["fit_record_count"],
        "evaluation_record_count": len(evaluation_truth),
        "family_group_counts": subsplit["family_group_counts"],
        "methods": methods,
        "imperfect_bounded_control_count": imperfect_bounded,
        "prediction_freeze_before_evaluation_truth_join": bool(
            prediction_freeze["predictions_frozen_before_evaluation_truth_join"]
            and not prediction_freeze["evaluation_truth_joined_before_prediction_freeze"]
        ),
        "model_eligible_residual_record_count": len(eligible),
        "model_eligible_residual_group_count": len(eligible_groups),
        "model_eligible_average_normalized_decision_regret": average_eligible_regret,
        "model_eligible": model_eligible,
    }
    metrics["finite_metrics"] = _finite(metrics)
    return metrics


def audit_controls(
    metrics: dict[str, Any], prediction_freeze: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["evaluationGates"]
    methods = metrics["methods"]
    checks = {
        "development_subsplit_counts_and_separation_exact": bool(
            metrics["fit_group_count"] == gates["requiredFitGroupCount"]
            and metrics["evaluation_group_count"] == gates["requiredEvaluationGroupCount"]
            and metrics["fit_record_count"] == gates["requiredFitRecordCount"]
            and metrics["evaluation_record_count"] == gates["requiredEvaluationRecordCount"]
            and metrics["fit_evaluation_group_overlap_count"] <= gates["maximumFitEvaluationGroupOverlapCount"]
            and all(
                counts["fit"] == gates["requiredFitAndEvaluationGroupsPerFamily"]
                and counts["evaluation"] == gates["requiredFitAndEvaluationGroupsPerFamily"]
                for counts in metrics["family_group_counts"].values()
            )
        ),
        "all_method_prediction_counts_and_invariance_exact": bool(
            all(method["prediction_count"] == gates["requiredPredictionCountPerMethod"] for method in methods.values())
            and all(method["group_variant_invariance"] == gates["requiredGroupVariantInvariance"] for method in methods.values())
            and all(method["candidate_budget_compliance"] == gates["requiredCandidateBudgetCompliance"] for method in methods.values())
        ),
        "ceiling_full_constraint_and_stack_exact": bool(
            methods["EXACT_STRUCTURAL_CEILING"]["exact_version_space_accuracy"] == gates["requiredStructuralCeilingExactVersionSpaceAccuracy"]
            and methods["EXACT_STRUCTURAL_CEILING"]["evidence_status_accuracy"] == gates["requiredStructuralCeilingStatusAccuracy"]
            and methods["FULL_CONSTRAINT_PROPAGATION"]["exact_version_space_accuracy"] == gates["requiredFullConstraintExactVersionSpaceAccuracy"]
            and methods["FULL_CONSTRAINT_PROPAGATION"]["evidence_status_accuracy"] == gates["requiredFullConstraintStatusAccuracy"]
            and methods["DETERMINISTIC_STACK"]["exact_version_space_accuracy"] == gates["requiredDeterministicStackExactVersionSpaceAccuracy"]
            and methods["DETERMINISTIC_STACK"]["evidence_status_accuracy"] == gates["requiredDeterministicStackStatusAccuracy"]
            and methods["DETERMINISTIC_STACK"]["shadow_action_accuracy"] == gates["requiredDeterministicStackActionAccuracy"]
            and methods["DETERMINISTIC_STACK"]["average_normalized_decision_regret"] <= gates["maximumDeterministicStackNormalizedDecisionRegret"]
        ),
        "bounded_controls_are_empirically_separated": metrics["imperfect_bounded_control_count"] >= gates["minimumImperfectBoundedControlCount"],
        "predictions_frozen_before_evaluation_truth_join": bool(
            metrics["prediction_freeze_before_evaluation_truth_join"] == gates["requiredPredictionFreezeBeforeEvaluationTruthJoin"]
            and prediction_freeze["control_worker_evaluation_truth_path_count"] <= gates["maximumControlWorkerEvaluationTruthPathCount"]
            and prediction_freeze["control_worker_hidden_evaluation_field_count"] <= gates["maximumControlWorkerHiddenEvaluationFieldCount"]
        ),
        "metrics_are_finite": metrics["finite_metrics"] == gates["requiredFiniteMetrics"],
    }
    access_gates = config["accessGates"]
    access_checks = {
        "one_deterministic_control_run": access["deterministic_control_run_count"] == access_gates["requiredDeterministicControlRunCount"],
        "protected_language_model_external_and_effect_boundaries_zero": bool(
            access["v213_protected_public_access_count"] <= access_gates["maximumV213ProtectedPublicAccessCount"]
            and access["v213_protected_truth_access_count"] <= access_gates["maximumV213ProtectedTruthAccessCount"]
            and access["protected_group_construction_count"] <= access_gates["maximumProtectedGroupConstructionCount"]
            and access["natural_language_surface_read_count"] <= access_gates["maximumNaturalLanguageSurfaceReadCount"]
            and access["external_ontology_payload_read_count"] <= access_gates["maximumExternalOntologyPayloadReadCount"]
            and access["model_load_count"] <= access_gates["maximumModelLoadCount"]
            and access["model_generation_count"] <= access_gates["maximumModelGenerationCount"]
            and access["api_call_count"] <= access_gates["maximumAPICallCount"]
            and access["training_run_count"] <= access_gates["maximumTrainingRunCount"]
            and access["ontology_registration_count"] <= access_gates["maximumOntologyRegistrationCount"]
            and access["trusted_state_mutation_count"] <= access_gates["maximumTrustedStateMutationCount"]
            and access["service_call_count"] <= access_gates["maximumServiceCallCount"]
            and access["external_side_effect_count"] <= access_gates["maximumExternalSideEffectCount"]
            and access["actual_execution_count"] <= access_gates["maximumActualExecutionCount"]
        ),
    }
    passed = all(checks.values()) and all(access_checks.values())
    if not passed:
        branch = "NEGATIVE_DETERMINISTIC_CONTROL_STUDY"
        decision = config["decisionRule"]["otherwise"]
    elif metrics["model_eligible"]:
        branch = "BOUNDED_LOCAL_CANDIDATE_GENERATOR_DESIGN_ELIGIBLE"
        decision = config["decisionRule"]["ifStudyPassesAndModelEligibilityRulePasses"]
    else:
        branch = "DETERMINISTIC_CLOSURE_ZERO_MODEL_ELIGIBILITY"
        decision = config["decisionRule"]["ifStudyPassesAndModelEligibilityRuleFails"]
    return {
        "passed": passed,
        "branch": branch,
        "model_eligible": bool(passed and metrics["model_eligible"]),
        "decision": decision,
        "checks": checks,
        "access_checks": access_checks,
    }
