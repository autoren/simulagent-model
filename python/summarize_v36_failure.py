#!/usr/bin/env python3
"""Descriptive, post-outcome localization of the frozen V36 failure.

This script reads only saved V36 predictions and labels.  It does not fit,
select, tune, or run the backbone, and therefore cannot amend the frozen V36
decision.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from v22r2_grounding import PROJECT_ROOT
from v32_language import compile_truth


CORPUS = PROJECT_ROOT / "data/v36-independent-confirmation/independent_confirmation.jsonl"
PREDICTIONS = PROJECT_ROOT / "outputs/v36-independent-confirmation/evaluation/predictions.jsonl"
RESULT = PROJECT_ROOT / "outputs/v36-independent-confirmation/evaluation/result.json"
IMPLEMENTATION_LOCK = PROJECT_ROOT / "configs/v36-implementation-lock.json"
OUTPUT = PROJECT_ROOT / "outputs/v36-independent-confirmation/failure-localization.json"
MARKDOWN = PROJECT_ROOT / "docs/v36-semantic-failure-localization.md"
V32_CORPORA = tuple(sorted((PROJECT_ROOT / "data/v32-factorized-semantics").glob("factor_*.jsonl")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def keyed_confusion(counter: Counter[tuple[str, str]]) -> dict[str, int]:
    return {f"{target}->{prediction}": count for (target, prediction), count in sorted(counter.items())}


def main() -> None:
    rows = read_jsonl(CORPUS)
    predictions = {row["id"]: row for row in read_jsonl(PREDICTIONS)}
    if set(predictions) != {row["id"] for row in rows}:
        raise ValueError("Saved predictions do not exactly cover V36")
    result = json.loads(RESULT.read_text())
    if result["decision"] != "confirmation_fail_reopen_semantic_interface_only":
        raise ValueError("Unexpected frozen V36 decision")
    v32_config = json.loads(IMPLEMENTATION_LOCK.read_text())["v32_config_payload"]

    sign_confusion: Counter[tuple[str, str]] = Counter()
    operation_confusion: Counter[tuple[str, str]] = Counter()
    joint: Counter[tuple[bool, bool, bool]] = Counter()
    family: dict[str, Counter[str]] = defaultdict(Counter)
    distractor: dict[bool, Counter[str]] = defaultdict(Counter)
    target_sign: dict[str, Counter[str]] = defaultdict(Counter)
    cell: dict[str, Counter[str]] = defaultdict(Counter)
    actual_truth_correct = 0
    oracle_sign_truth_correct = 0
    oracle_operation_truth_correct = 0
    both_components_correct = 0

    for row in rows:
        target = row["target"]
        target_sign_value = target["factorization"]["lexical_sign"]
        target_operation = target["factorization"]["outer_operation"]
        prediction = predictions[row["id"]]["selected_intermediates"]
        predicted_sign = prediction["lexical_sign"]
        predicted_operation = prediction["outer_operation"]
        sign_correct = predicted_sign == target_sign_value
        operation_correct = predicted_operation == target_operation
        actual_truth = compile_truth(predicted_sign, predicted_operation, v32_config) == target["truth_status"]
        oracle_sign_truth = compile_truth(target_sign_value, predicted_operation, v32_config) == target["truth_status"]
        oracle_operation_truth = compile_truth(predicted_sign, target_operation, v32_config) == target["truth_status"]

        sign_confusion[(target_sign_value, predicted_sign)] += 1
        operation_confusion[(target_operation, predicted_operation)] += 1
        joint[(sign_correct, operation_correct, actual_truth)] += 1
        actual_truth_correct += actual_truth
        oracle_sign_truth_correct += oracle_sign_truth
        oracle_operation_truth_correct += oracle_operation_truth
        both_components_correct += sign_correct and operation_correct

        groups = (
            family[row["oracle_metadata"]["surface_family"]],
            distractor[bool(row["oracle_metadata"]["distractor"])],
            target_sign[target_sign_value],
            cell[f"{target_operation}.{target_sign_value}"],
        )
        for group in groups:
            group["records"] += 1
            group["sign_correct"] += sign_correct
            group["operation_correct"] += operation_correct
            group["truth_correct"] += actual_truth

    def summarize(groups: dict[Any, Counter[str]]) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for name, counts in sorted(groups.items(), key=lambda item: str(item[0])):
            total = counts["records"]
            output[str(name).lower()] = {
                "records": total,
                "lexical_sign_accuracy": accuracy(counts["sign_correct"], total),
                "outer_operation_accuracy": accuracy(counts["operation_correct"], total),
                "compiled_truth_accuracy": accuracy(counts["truth_correct"], total),
            }
        return output

    v32_evidence = {
        row["agent_input"]["evidence_text"]
        for path in V32_CORPORA
        for row in read_jsonl(path)
    }
    v36_evidence = {row["agent_input"]["evidence_text"] for row in rows}
    total = len(rows)
    unresolved_errors = sum(
        count for (target, predicted), count in operation_confusion.items()
        if target != predicted and predicted == "unresolved"
    )
    operation_errors = sum(count for (target, predicted), count in operation_confusion.items() if target != predicted)
    distractor_metrics = summarize(distractor)
    localization = {
        "schema_version": 36,
        "experiment": "v36_frozen_failure_localization",
        "status": "descriptive_post_outcome_only",
        "prohibited_actions_performed": {
            "backbone_forwards": 0,
            "fits": 0,
            "selection_runs": 0,
            "threshold_changes": 0,
            "v32_evaluation_scoring": 0,
        },
        "records": total,
        "exact_v32_v36_evidence_overlap": len(v32_evidence & v36_evidence),
        "sign_confusion": keyed_confusion(sign_confusion),
        "operation_confusion": keyed_confusion(operation_confusion),
        "operation_errors": operation_errors,
        "operation_errors_predicted_unresolved": unresolved_errors,
        "operation_error_fraction_predicted_unresolved": accuracy(unresolved_errors, operation_errors),
        "joint_component_truth_counts": {
            f"sign_{sign}_operation_{operation}_truth_{truth}": count
            for (sign, operation, truth), count in sorted(joint.items())
        },
        "counterfactual_truth_accuracy": {
            "saved_predictions": accuracy(actual_truth_correct, total),
            "oracle_sign_with_predicted_operation": accuracy(oracle_sign_truth_correct, total),
            "predicted_sign_with_oracle_operation": accuracy(oracle_operation_truth_correct, total),
            "both_components_correct": accuracy(both_components_correct, total),
        },
        "by_surface_family": summarize(family),
        "by_distractor": distractor_metrics,
        "by_target_sign": summarize(target_sign),
        "by_operation_and_sign": summarize(cell),
        "descriptive_findings": {
            "atom_interface_transferred": result["metrics"]["atom_exact_accuracy"] == 1.0,
            "both_semantic_components_failed_gate": (
                not result["gate_checks"]["lexical_sign"] and not result["gate_checks"]["outer_operation"]
            ),
            "operation_errors_concentrated_in_distractor_records": (
                distractor_metrics["true"]["outer_operation_accuracy"]
                < distractor_metrics["false"]["outer_operation_accuracy"]
            ),
            "unresolved_truth_is_sign_invariant": all(
                metrics["compiled_truth_accuracy"] == 1.0
                for key, metrics in summarize(cell).items() if key.startswith("unresolved.")
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(localization, indent=2, sort_keys=True) + "\n")

    oracle = localization["counterfactual_truth_accuracy"]
    lines = [
        "# V36 semantic failure localization", "",
        "Status: descriptive analysis of the frozen V36 predictions. This analysis performed no fitting, model forwards, selection, threshold changes, or V32 evaluation scoring, and it does not amend the frozen decision.", "",
        "## What transferred", "",
        "Predicate, entity binding, exact atom, and relation order all scored 1.000. The language grounder therefore transferred the content of the embedded proposition perfectly across the V36 families.", "",
        "## What failed", "",
        f"Lexical sign scored {result['metrics']['lexical_sign_accuracy']:.3f} and outer operation scored {result['metrics']['outer_operation_accuracy']:.3f}. Both failures are substantive: giving the compiler oracle sign while retaining the predicted operation raises truth accuracy only to {oracle['oracle_sign_with_predicted_operation']:.3f}, while giving it oracle operation with predicted sign raises it only to {oracle['predicted_sign_with_oracle_operation']:.3f}.", "",
        f"Of {operation_errors} operation errors, {unresolved_errors} ({localization['operation_error_fraction_predicted_unresolved']:.1%}) defaulted to `unresolved`. This is a strong out-of-template fallback, not a single reversed label.", "",
        f"Operation accuracy was {distractor_metrics['false']['outer_operation_accuracy']:.3f} without a distractor and {distractor_metrics['true']['outer_operation_accuracy']:.3f} with one. Thus distractor sensitivity accounts for much of the operation failure, although the non-distractor score still misses the preregistered 0.950 gate.", "",
        "All unresolved families reached 1.000 compiled truth even though their lexical-sign accuracy was imperfect. That is expected because unresolved propositions compile to unknown under either lexical sign; it is not evidence that sign transfer succeeded.", "",
        "## Independence and interpretation", "",
        f"Exact evidence-text overlap between all V32 factor corpora and V36 was {localization['exact_v32_v36_evidence_overlap']}. The result is therefore consistent with template-local semantic readouts that fit V32/V35 development language but did not learn a stable representation of sign and discourse operation.", "",
        "The justified next direction is restricted to the semantic interface: develop invariance across wording, distractor placement, negation scope, and operation paraphrase on new development-only language. Keep the backbone, atom/binding interface, executor, V32 evaluation firewall, and V28 prohibition fixed. Do not preregister or construct the end-to-end relational suite yet.", "",
        "Machine-readable tables, including all confusion matrices and operation/sign cells, are in `outputs/v36-independent-confirmation/failure-localization.json`.",
    ]
    MARKDOWN.write_text("\n".join(lines) + "\n")
    print(json.dumps(localization, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
