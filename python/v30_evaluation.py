"""Pure metric, gate, and decision utilities for V30."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from v30_language import canonical_prediction


def mean(values: Sequence[bool | float]) -> float:
    return float(np.mean(values)) if values else 0.0


def top_values(
    logits: dict[str, float], options: Sequence[dict[str, str]], count: int,
) -> list[str]:
    order = {option["token"]: index for index, option in enumerate(options)}
    ranked = sorted(
        options, key=lambda option: (-float(logits[option["token"]]), order[option["token"]])
    )
    return [option["value"] for option in ranked[:count]]


def primary_rows(
    records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    prediction_lookup = {row["id"]: row for row in predictions}
    result = []
    for record in records:
        prediction = prediction_lookup[record["id"]]
        target = record["target"]
        fields = prediction["selected_fields"]
        predicted_fact = canonical_prediction(fields, config)
        expected_argument_2 = (
            target["arguments"][1] if target["predicate_kind"] == "relation"
            else config["methods"]["primary"]["argument2NotApplicableValue"]
        )
        predicate_ok = fields["predicate"] == target["predicate"]
        argument_1_ok = fields["argument_1"] == target["arguments"][0]
        argument_2_ok = fields["argument_2"] == expected_argument_2
        relation_order_ok = (
            argument_1_ok and argument_2_ok if target["predicate_kind"] == "relation" else True
        )
        truth_ok = fields["truth_status"] == target["truth_status"]
        exact = (
            predicted_fact["predicate"] == target["predicate"]
            and predicted_fact["arguments"] == target["arguments"]
            and predicted_fact["truth_status"] == target["truth_status"]
        )
        gold_fields = {
            "predicate": target["predicate"],
            "argument_1": target["arguments"][0],
            "argument_2": expected_argument_2,
            "truth_status": target["truth_status"],
        }
        top2 = {
            field: gold in top_values(
                prediction["field_logits"][field], prediction["field_options"][field], 2
            )
            for field, gold in gold_fields.items()
        }
        result.append({
            "id": record["id"], "split": record["split"], "scene_id": record["scene_id"],
            "predicate_kind": target["predicate_kind"], "predicate": target["predicate"],
            "truth_status": target["truth_status"],
            "semantic_operator": record["oracle_metadata"]["semantic_operator"],
            "surface_family": record["oracle_metadata"]["surface_family"],
            "surface_name": record["oracle_metadata"]["surface_name"],
            "sentence_length_stratum": record["oracle_metadata"]["sentence_length_stratum"],
            "scene_variant": record["oracle_metadata"]["scene_variant"],
            "entity_count": record["oracle_metadata"]["entity_count"],
            "predicate_correct": predicate_ok, "argument_1_correct": argument_1_ok,
            "argument_2_correct": argument_2_ok,
            "relation_argument_order_correct": relation_order_ok,
            "truth_status_correct": truth_ok, "exact_signed_fact": exact,
            "target_retained_top2": all(top2.values()), "field_top2": top2,
            "predicted_fact": predicted_fact,
            "pairs": record["oracle_metadata"]["pairs"],
        })
    if len(prediction_lookup) != len(records):
        raise ValueError("V30 primary predictions do not cover exactly the corpus")
    return result


def grouped_accuracy(
    rows: Sequence[dict[str, Any]], field: str, metric: str,
) -> dict[str, dict[str, Any]]:
    return {
        str(value): {"records": len(selected), metric: mean([row[metric] for row in selected])}
        for value in sorted({row[field] for row in rows}, key=str)
        for selected in [[row for row in rows if row[field] == value]]
    }


def pair_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for pair in row["pairs"]:
            groups[(pair["kind"], pair["id"])].append(row)
    result = {}
    for kind in sorted({key[0] for key in groups}):
        selected = [members for (current, _), members in groups.items() if current == kind]
        if any(len(members) != 2 for members in selected):
            raise ValueError(f"V30 evaluation pair population malformed for {kind}")
        result[kind] = {
            "pairs": len(selected),
            "pair_exact": mean([
                all(row["exact_signed_fact"] for row in members) for members in selected
            ]),
        }
    return result


def primary_summary(
    records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    scored = primary_rows(records, predictions, config)
    by_split = {}
    for split in config["splits"]:
        rows = [row for row in scored if row["split"] == split]
        relations = [row for row in rows if row["predicate_kind"] == "relation"]
        scenes = defaultdict(list)
        for row in rows:
            scenes[row["scene_id"]].append(row)
        by_split[split] = {
            "records": len(rows), "scenes": len(scenes),
            "predicate_accuracy": mean([row["predicate_correct"] for row in rows]),
            "argument_1_accuracy": mean([row["argument_1_correct"] for row in rows]),
            "argument_2_accuracy": mean([row["argument_2_correct"] for row in rows]),
            "relation_argument_order_accuracy": mean([
                row["relation_argument_order_correct"] for row in relations
            ]),
            "truth_status_accuracy": mean([row["truth_status_correct"] for row in rows]),
            "exact_signed_fact_accuracy": mean([row["exact_signed_fact"] for row in rows]),
            "exact_scene_accuracy": mean([
                all(row["exact_signed_fact"] for row in members) for members in scenes.values()
            ]),
            "target_retention_top2": mean([row["target_retained_top2"] for row in rows]),
            "by_semantic_operator": grouped_accuracy(
                rows, "semantic_operator", "exact_signed_fact"
            ),
            "truth_by_semantic_operator": grouped_accuracy(
                rows, "semantic_operator", "truth_status_correct"
            ),
            "by_surface_family": grouped_accuracy(rows, "surface_family", "exact_signed_fact"),
            "truth_by_surface_family": grouped_accuracy(
                rows, "surface_family", "truth_status_correct"
            ),
            "by_truth_status": grouped_accuracy(rows, "truth_status", "truth_status_correct"),
            "by_predicate_kind": grouped_accuracy(rows, "predicate_kind", "exact_signed_fact"),
            "by_predicate": grouped_accuracy(rows, "predicate", "exact_signed_fact"),
            "by_entity_count": grouped_accuracy(rows, "entity_count", "exact_signed_fact"),
            "by_sentence_length": grouped_accuracy(
                rows, "sentence_length_stratum", "exact_signed_fact"
            ),
            "by_scene_variant": grouped_accuracy(rows, "scene_variant", "exact_signed_fact"),
            "controlled_pairs": pair_summary(rows),
        }
    evaluation = by_split["language_evaluation"]
    gates = config["gates"]["languageEvaluation"]
    checks = {
        "predicate_accuracy": evaluation["predicate_accuracy"] >= gates["minimumPredicateAccuracy"],
        "argument_1_accuracy": evaluation["argument_1_accuracy"] >= gates["minimumArgument1Accuracy"],
        "relation_argument_order_accuracy": (
            evaluation["relation_argument_order_accuracy"]
            >= gates["minimumRelationArgumentOrderAccuracy"]
        ),
        "truth_status_accuracy": (
            evaluation["truth_status_accuracy"] >= gates["minimumTruthStatusAccuracy"]
        ),
        "exact_signed_fact_accuracy": (
            evaluation["exact_signed_fact_accuracy"] >= gates["minimumExactSignedFactAccuracy"]
        ),
        "exact_scene_accuracy": evaluation["exact_scene_accuracy"] >= gates["minimumExactSceneAccuracy"],
        "worst_semantic_operator_exact": min(
            row["exact_signed_fact"] for row in evaluation["by_semantic_operator"].values()
        ) >= gates["minimumWorstSemanticOperatorExact"],
        "worst_surface_family_exact": min(
            row["exact_signed_fact"] for row in evaluation["by_surface_family"].values()
        ) >= gates["minimumWorstSurfaceFamilyExact"],
        "worst_truth_status_accuracy": min(
            row["truth_status_correct"] for row in evaluation["by_truth_status"].values()
        ) >= gates["minimumWorstTruthStatusAccuracy"],
        "distractor_pair_exact": (
            evaluation["controlled_pairs"]["distractor"]["pair_exact"]
            >= gates["minimumDistractorPairExact"]
        ),
        "inverse_pair_exact": (
            evaluation["controlled_pairs"]["inverse"]["pair_exact"]
            >= gates["minimumInversePairExact"]
        ),
        "argument_reversal_pair_exact": (
            evaluation["controlled_pairs"]["argument_reversal"]["pair_exact"]
            >= gates["minimumArgumentReversalPairExact"]
        ),
        "affirmative_negated_pair_exact": (
            evaluation["controlled_pairs"]["affirmative_negated"]["pair_exact"]
            >= gates["minimumAffirmativeNegatedPairExact"]
        ),
        "false_unknown_pair_exact": (
            evaluation["controlled_pairs"]["false_unknown"]["pair_exact"]
            >= gates["minimumFalseUnknownPairExact"]
        ),
    }
    return {"by_split": by_split, "checks": checks, "passed": all(checks.values())}


def truth_summary(
    records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    lookup = {row["id"]: row for row in predictions}
    scored = []
    for record in records:
        prediction = lookup[record["id"]]
        gold = record["target"]["truth_status"]
        options = prediction["options"]
        top2 = gold in top_values(prediction["logits"], options, 2)
        scored.append({
            "id": record["id"], "split": record["split"],
            "truth_status": gold,
            "semantic_operator": record["oracle_metadata"]["semantic_operator"],
            "surface_family": record["oracle_metadata"]["surface_family"],
            "correct": prediction["predicted_truth_status"] == gold,
            "target_retained_top2": top2,
        })
    if len(lookup) != len(records):
        raise ValueError("V30 truth predictions do not cover exactly the corpus")
    by_split = {}
    for split in config["splits"]:
        rows = [row for row in scored if row["split"] == split]
        by_split[split] = {
            "records": len(rows), "truth_status_accuracy": mean([row["correct"] for row in rows]),
            "truth_target_retention_top2": mean([
                row["target_retained_top2"] for row in rows
            ]),
            "by_truth_status": grouped_accuracy(rows, "truth_status", "correct"),
            "by_semantic_operator": grouped_accuracy(rows, "semantic_operator", "correct"),
            "by_surface_family": grouped_accuracy(rows, "surface_family", "correct"),
        }
    return {"by_split": by_split}


def lora_eligibility(
    primary: dict[str, Any], nli: dict[str, Any] | None,
    structural_audit_passed: bool, config: dict[str, Any],
) -> dict[str, Any]:
    rules = config["gates"]["loraEligibility"]
    evaluation = primary["by_split"]["language_evaluation"]
    if nli is None:
        checks = {
            "structural_audit_passed": structural_audit_passed,
            "primary_failed": not primary["passed"],
            "candidate_nli_ran": False,
            "candidate_nli_failed": False,
            "shared_supported_operator_failure": False,
            "oracle_atom_truth_not_retained_top2": False,
            "no_evaluation_selection": True,
        }
        return {"eligible": False, "checks": checks, "shared_failing_operators": []}
    nli_eval = nli["by_split"]["language_evaluation"]
    threshold = rules["surfaceFamilyTruthFailureThreshold"]
    required = rules["minimumRequiredTruthAccuracy"]
    shared = []
    for operator in config["semanticOperators"]:
        primary_families = {
            family for family, row in evaluation["truth_by_surface_family"].items()
            if family.startswith(operator + ".") and row["truth_status_correct"] < threshold
        }
        nli_families = {
            family for family, row in nli_eval["by_surface_family"].items()
            if family.startswith(operator + ".") and row["correct"] < threshold
        }
        if len(primary_families & nli_families) >= rules["minimumFailingEvaluationSurfaceFamiliesPerOperator"]:
            shared.append(operator)
    checks = {
        "structural_audit_passed": structural_audit_passed,
        "primary_failed": not primary["passed"],
        "candidate_nli_ran": True,
        "candidate_nli_failed": nli_eval["truth_status_accuracy"] < required,
        "shared_supported_operator_failure": (
            len(shared) >= rules["minimumSharedFailingSupportedOperators"]
        ),
        "oracle_atom_truth_not_retained_top2": (
            nli_eval["truth_target_retention_top2"]
            < rules["maximumAcceptableOracleAtomTruthTop2Recall"]
        ),
        "no_evaluation_selection": True,
    }
    return {"eligible": all(checks.values()), "checks": checks, "shared_failing_operators": shared}
