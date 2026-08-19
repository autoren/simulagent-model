from __future__ import annotations

from collections import defaultdict
from typing import Any

from v165_factored_ontology_identifiability_population import (
    candidate_universe,
    enumerate_version_space,
    parse_definition,
)


BASELINE_NAMES = (
    "complete_safe_enumeration",
    "definition_parser_only",
    "ontology_retrieval_only",
    "observation_filter_only",
    "exact_parser_plus_version_space",
    "oracle_hidden_contract",
)


def evidence_status(candidate_ids: list[str]) -> str:
    if not candidate_ids:
        return "contradictory"
    if len(candidate_ids) == 1:
        return "sufficient"
    return "ambiguous"


def _namespace(record: dict[str, Any], ontology: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in ontology["namespaces"]
        if row["entity_noun"] == record["entity_noun"]
        and [item["name"] for item in row["primitives"]]
        == record["registered_primitives"]
    ]
    if len(matches) != 1:
        raise ValueError(f"record does not identify one namespace: {record['record_id']}")
    return {
        "namespace_id": matches[0]["namespace_id"],
        "entity_noun": matches[0]["entity_noun"],
        "primitive_names": [item["name"] for item in matches[0]["primitives"]],
        "registered_formulas": matches[0]["registered_formulas"],
    }


def canonical_registered_definitions(
    record: dict[str, Any], namespace: dict[str, Any]
) -> dict[str, str]:
    result: dict[str, str] = {}
    entity = record["entity_noun"]
    concept = record["concept_name"]
    for formula in namespace["registered_formulas"]:
        atoms = formula["atoms"]
        if formula["form"] == "ATOM":
            definition = f"A {entity} is {concept} exactly when it is {atoms[0]}."
        elif formula["form"] == "AND":
            definition = (
                f"A {entity} is {concept} exactly when it is both "
                f"{atoms[0]} and {atoms[1]}."
            )
        elif formula["form"] == "OR":
            definition = (
                f"A {entity} is {concept} exactly when it is either "
                f"{atoms[0]} or {atoms[1]} or both."
            )
        else:
            raise ValueError(f"unexpected registered form: {formula['form']}")
        if definition in result:
            raise ValueError("canonical registered definitions are not unique")
        result[definition] = formula["candidate_id"]
    return result


def _prediction(candidate_ids: list[str]) -> dict[str, Any]:
    ordered = sorted(set(candidate_ids))
    return {
        "candidate_ids": ordered,
        "candidate_count": len(ordered),
        "evidence_status": evidence_status(ordered),
    }


def predict_record(
    record: dict[str, Any],
    hidden: dict[str, Any],
    ontology: dict[str, Any],
    universe: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    candidates = candidate_universe() if universe is None else universe
    all_ids = [row["candidate_id"] for row in candidates]
    namespace = _namespace(record, ontology)
    parsed = parse_definition(record["definition"], namespace, record["concept_name"])
    parser_ids = [parsed["candidate_id"]] if parsed.get("candidate_id") else all_ids
    canonical = canonical_registered_definitions(record, namespace)
    retrieved = canonical.get(record["definition"])
    retrieval_ids = [retrieved] if retrieved else all_ids
    unconstrained = {"parse_kind": "underspecified", "candidate_id": None}
    observation_ids = [
        row["candidate_id"]
        for row in enumerate_version_space(
            unconstrained, record["observations"], namespace, candidates
        )
    ]
    combined_ids = [
        row["candidate_id"]
        for row in enumerate_version_space(
            parsed, record["observations"], namespace, candidates
        )
    ]
    return {
        "complete_safe_enumeration": _prediction(all_ids),
        "definition_parser_only": _prediction(parser_ids),
        "ontology_retrieval_only": _prediction(retrieval_ids),
        "observation_filter_only": _prediction(observation_ids),
        "exact_parser_plus_version_space": _prediction(combined_ids),
        "oracle_hidden_contract": _prediction(
            hidden["version_space_candidate_ids"]
        ),
    }


def build_predictions(
    public_records: list[dict[str, Any]],
    hidden_records: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> list[dict[str, Any]]:
    hidden_by_id = {row["record_id"]: row for row in hidden_records}
    if len(hidden_by_id) != len(hidden_records):
        raise ValueError("hidden record identifiers are not unique")
    universe = candidate_universe()
    rows = []
    for public in sorted(public_records, key=lambda row: row["record_id"]):
        hidden = hidden_by_id[public["record_id"]]
        rows.append(
            {
                "record_id": public["record_id"],
                "predictions": predict_record(public, hidden, ontology, universe),
            }
        )
    return rows


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _binary_metrics(predicted: list[bool], actual: list[bool]) -> dict[str, float]:
    true_positive = sum(p and a for p, a in zip(predicted, actual))
    predicted_positive = sum(predicted)
    actual_positive = sum(actual)
    return {
        "precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "recall": true_positive / actual_positive if actual_positive else 0.0,
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    hidden_records: list[dict[str, Any]],
) -> dict[str, Any]:
    hidden_by_id = {row["record_id"]: row for row in hidden_records}
    universe_by_id = {row["candidate_id"]: row for row in candidate_universe()}
    metrics: dict[str, Any] = {}
    for baseline in BASELINE_NAMES:
        joined = [
            (hidden_by_id[row["record_id"]], row["predictions"][baseline])
            for row in predictions
        ]
        noncontradictory = [
            (hidden, prediction)
            for hidden, prediction in joined
            if hidden["evidence_status"] != "contradictory"
        ]
        sufficient = [
            (hidden, prediction)
            for hidden, prediction in joined
            if hidden["evidence_status"] == "sufficient"
        ]
        known_sufficient = [
            (hidden, prediction)
            for hidden, prediction in sufficient
            if hidden["generative_expressibility"] in {"alias", "composition"}
        ]
        unresolved = [
            (hidden, prediction)
            for hidden, prediction in joined
            if hidden["evidence_status"] in {"ambiguous", "contradictory"}
        ]
        ambiguity = _binary_metrics(
            [prediction["evidence_status"] == "ambiguous" for _, prediction in joined],
            [hidden["evidence_status"] == "ambiguous" for hidden, _ in joined],
        )
        contradiction = _binary_metrics(
            [
                prediction["evidence_status"] == "contradictory"
                for _, prediction in joined
            ],
            [hidden["evidence_status"] == "contradictory" for hidden, _ in joined],
        )
        renaming: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for hidden, prediction in joined:
            renaming[hidden["logical_target_group"]].append(
                tuple(prediction["candidate_ids"])
            )
        metrics[baseline] = {
            "record_count": len(joined),
            "exact_version_space_accuracy": _rate(
                [
                    prediction["candidate_ids"]
                    == hidden["version_space_candidate_ids"]
                    for hidden, prediction in joined
                ]
            ),
            "evidence_status_accuracy": _rate(
                [
                    prediction["evidence_status"] == hidden["evidence_status"]
                    for hidden, prediction in joined
                ]
            ),
            "noncontradictory_target_retention": _rate(
                [
                    hidden["target_candidate_id"] in prediction["candidate_ids"]
                    for hidden, prediction in noncontradictory
                ]
            ),
            "sufficient_exact_candidate_accuracy": _rate(
                [
                    prediction["candidate_ids"] == [hidden["target_candidate_id"]]
                    for hidden, prediction in sufficient
                ]
            ),
            "sufficient_expressibility_class_accuracy": _rate(
                [
                    len(prediction["candidate_ids"]) == 1
                    and universe_by_id[prediction["candidate_ids"][0]][
                        "expressibility_class"
                    ]
                    == hidden["generative_expressibility"]
                    for hidden, prediction in sufficient
                ]
            ),
            "ambiguity_precision": ambiguity["precision"],
            "ambiguity_recall": ambiguity["recall"],
            "contradiction_precision": contradiction["precision"],
            "contradiction_recall": contradiction["recall"],
            "false_provisional_creation_rate": _rate(
                [
                    len(prediction["candidate_ids"]) == 1
                    and universe_by_id[prediction["candidate_ids"][0]][
                        "expressibility_class"
                    ]
                    == "provisional_primitive"
                    for _, prediction in known_sufficient
                ]
            ),
            "false_resolution_rate": _rate(
                [len(prediction["candidate_ids"]) == 1 for _, prediction in unresolved]
            ),
            "mean_candidate_set_size": sum(
                prediction["candidate_count"] for _, prediction in joined
            )
            / len(joined),
            "renaming_invariance": _rate(
                [len(set(group)) == 1 for group in renaming.values()]
            ),
        }
    combined_by_id = {
        row["record_id"]: row["predictions"]["exact_parser_plus_version_space"]
        for row in predictions
    }
    residual_ids = sorted(
        hidden["record_id"]
        for hidden in hidden_records
        if combined_by_id[hidden["record_id"]]["candidate_ids"]
        != hidden["version_space_candidate_ids"]
    )
    ambiguous_counts = sorted(
        combined_by_id[hidden["record_id"]]["candidate_count"]
        for hidden in hidden_records
        if hidden["evidence_status"] == "ambiguous"
    )
    return {
        "baseline_metrics": metrics,
        "model_eligible_residual_record_ids": residual_ids,
        "model_eligible_residual_count": len(residual_ids),
        "intentionally_ambiguous_record_count": len(ambiguous_counts),
        "intentionally_ambiguous_candidate_counts": ambiguous_counts,
    }


def evaluate_gates(
    evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, bool]:
    gates = config["baselineGates"]
    combined = evaluation["baseline_metrics"]["exact_parser_plus_version_space"]
    oracle = evaluation["baseline_metrics"]["oracle_hidden_contract"]
    return {
        "record_count": combined["record_count"] == gates["requiredRecordCount"],
        "baseline_names": set(evaluation["baseline_metrics"])
        == set(gates["requiredBaselineNames"]),
        "oracle_exact_version_space": oracle["exact_version_space_accuracy"]
        == gates["requiredOracleExactVersionSpaceAccuracy"],
        "oracle_evidence_status": oracle["evidence_status_accuracy"]
        == gates["requiredOracleEvidenceStatusAccuracy"],
        "combined_exact_version_space": combined["exact_version_space_accuracy"]
        == gates["requiredExactCombinedVersionSpaceAccuracy"],
        "combined_evidence_status": combined["evidence_status_accuracy"]
        == gates["requiredExactCombinedEvidenceStatusAccuracy"],
        "combined_target_retention": combined["noncontradictory_target_retention"]
        == gates["requiredExactCombinedTargetRetention"],
        "combined_sufficient_candidate": combined["sufficient_exact_candidate_accuracy"]
        == gates["requiredExactCombinedSufficientCandidateAccuracy"],
        "combined_sufficient_expressibility": combined[
            "sufficient_expressibility_class_accuracy"
        ]
        == gates["requiredExactCombinedSufficientExpressibilityAccuracy"],
        "combined_ambiguity_recall": combined["ambiguity_recall"]
        == gates["requiredExactCombinedAmbiguityRecall"],
        "combined_contradiction_recall": combined["contradiction_recall"]
        == gates["requiredExactCombinedContradictionRecall"],
        "zero_false_provisional_creation": combined[
            "false_provisional_creation_rate"
        ]
        == gates["requiredExactCombinedFalseProvisionalCreationRate"],
        "zero_false_resolution": combined["false_resolution_rate"]
        == gates["requiredExactCombinedFalseResolutionRate"],
        "combined_renaming_invariance": combined["renaming_invariance"]
        == gates["requiredExactCombinedRenamingInvariance"],
        "zero_model_eligible_residual": evaluation["model_eligible_residual_count"]
        == gates["requiredModelEligibleResidualCount"],
        "ambiguous_record_count": evaluation["intentionally_ambiguous_record_count"]
        == gates["requiredIntentionallyAmbiguousRecordCount"],
        "ambiguous_candidate_count": set(
            evaluation["intentionally_ambiguous_candidate_counts"]
        )
        == {gates["requiredCandidatesPerAmbiguousRecord"]},
        "zero_disallowed_access": all(
            access[key] <= gates[maximum]
            for key, maximum in {
                "evaluation_record_count": "maximumEvaluationRecordCount",
                "manual_judgment_count": "maximumManualJudgmentCount",
                "model_load_count": "maximumModelLoadCount",
                "model_generation_count": "maximumModelGenerationCount",
                "API_call_count": "maximumAPICallCount",
                "training_run_count": "maximumTrainingRunCount",
                "ontology_registration_count": "maximumOntologyRegistrationCount",
                "real_service_call_count": "maximumRealServiceCallCount",
                "external_side_effect_count": "maximumExternalSideEffectCount",
                "actual_execution_count": "maximumActualExecutionCount",
            }.items()
        ),
    }


__all__ = [
    "BASELINE_NAMES",
    "build_predictions",
    "canonical_registered_definitions",
    "evaluate_gates",
    "evaluate_predictions",
    "evidence_status",
    "predict_record",
]
