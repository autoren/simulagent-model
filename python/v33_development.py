"""Fit/calibration-only metrics, assembly, selection, and qualification for V33."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from v32_language import compile_truth
from v32_structured_model import select_predictions


OUTPUT_KEYS = ("predicate", "argument1", "argument2", "truth", "lexical_sign", "outer_operation")


def mean(values: Sequence[bool | float]) -> float:
    return float(np.mean(values)) if values else 0.0


def combine_outputs(
    atom: tuple[np.ndarray, ...], truth: tuple[np.ndarray, ...],
    sign: tuple[np.ndarray, ...], operation: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Assemble one modular parser while retaining a capacity-matched output contract."""
    return atom[0], atom[1], atom[2], truth[3], sign[4], operation[5]


def score_development(
    records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    lookup = {row["id"]: row for row in predictions}
    if set(lookup) != {row["id"] for row in records}:
        raise ValueError("V33 predictions do not exactly cover the requested development population")
    rows, scenes = [], defaultdict(list)
    for record in records:
        target, prediction = record["target"], lookup[record["id"]]
        selected, intermediate = prediction["selected_fields"], prediction["selected_intermediates"]
        expected2 = target["arguments"][1] if target["predicate_kind"] == "relation" else "N/A"
        predicate = selected["predicate"] == target["predicate"]
        argument1 = selected["argument_1"] == target["arguments"][0]
        argument2 = selected["argument_2"] == expected2
        atom = predicate and argument1 and argument2
        sign = intermediate["lexical_sign"] == target["factorization"]["lexical_sign"]
        operation = intermediate["outer_operation"] == target["factorization"]["outer_operation"]
        direct_truth = intermediate["direct_truth_status"] == target["truth_status"]
        compiled_value = compile_truth(
            intermediate["lexical_sign"], intermediate["outer_operation"], config
        )
        compiled_truth = compiled_value == target["truth_status"]
        selected_truth = selected["truth_status"] == target["truth_status"]
        row = {
            "id": record["id"], "scene_id": record["scene_id"],
            "predicate_correct": predicate, "argument1_correct": argument1,
            "argument2_correct": argument2, "atom_exact": atom,
            "relation_order_correct": argument1 and argument2,
            "lexical_sign_correct": sign, "outer_operation_correct": operation,
            "both_components_correct": sign and operation,
            "direct_truth_correct": direct_truth, "compiled_truth_correct": compiled_truth,
            "selected_truth_correct": selected_truth,
            "direct_exact_fact": atom and direct_truth,
            "compiled_exact_fact": atom and compiled_truth,
            "selected_exact_fact": atom and selected_truth,
            "predicate_kind": target["predicate_kind"],
            "surface_family": record["oracle_metadata"]["surface_family"],
        }
        rows.append(row)
        scenes[row["scene_id"]].append(row)
    relations = [row for row in rows if row["predicate_kind"] == "relation"]
    families = {
        family: {
            "records": len(selected),
            "atom_exact_accuracy": mean([row["atom_exact"] for row in selected]),
            "compiled_exact_fact_accuracy": mean([row["compiled_exact_fact"] for row in selected]),
        }
        for family in sorted({row["surface_family"] for row in rows})
        for selected in [[row for row in rows if row["surface_family"] == family]]
    }
    return {
        "records": len(rows), "scenes": len(scenes),
        "predicate_accuracy": mean([row["predicate_correct"] for row in rows]),
        "argument1_accuracy": mean([row["argument1_correct"] for row in rows]),
        "argument2_accuracy": mean([row["argument2_correct"] for row in rows]),
        "relation_order_accuracy": mean([row["relation_order_correct"] for row in relations]),
        "atom_exact_accuracy": mean([row["atom_exact"] for row in rows]),
        "lexical_sign_accuracy": mean([row["lexical_sign_correct"] for row in rows]),
        "outer_operation_accuracy": mean([row["outer_operation_correct"] for row in rows]),
        "both_components_accuracy": mean([row["both_components_correct"] for row in rows]),
        "direct_truth_accuracy": mean([row["direct_truth_correct"] for row in rows]),
        "compiled_truth_accuracy": mean([row["compiled_truth_correct"] for row in rows]),
        "selected_truth_accuracy": mean([row["selected_truth_correct"] for row in rows]),
        "direct_exact_fact_accuracy": mean([row["direct_exact_fact"] for row in rows]),
        "compiled_exact_fact_accuracy": mean([row["compiled_exact_fact"] for row in rows]),
        "selected_exact_fact_accuracy": mean([row["selected_exact_fact"] for row in rows]),
        "compiled_exact_scene_accuracy": mean([
            all(row["compiled_exact_fact"] for row in selected) for selected in scenes.values()
        ]),
        "by_surface_family": families,
    }


def component_mean(metrics: dict[str, Any]) -> float:
    return mean([
        metrics["atom_exact_accuracy"], metrics["lexical_sign_accuracy"],
        metrics["outer_operation_accuracy"], metrics["compiled_truth_accuracy"],
    ])


def select_search_checkpoint(
    objective: str, reports: Sequence[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    primary = config["search"]["selectionPrimaryMetric"][objective]
    ordered = sorted(reports, key=lambda row: (
        -row["calibration"][primary], -row["fit"][primary],
        -component_mean(row["calibration"]), row["epoch"], row["learning_rate"],
    ))
    return ordered[0]


def qualification_checks(
    fit: dict[str, Any], calibration: dict[str, Any], config: dict[str, Any],
) -> dict[str, bool]:
    fit_gates, calibration_gates = (
        config["qualification"]["perSeedFit"], config["qualification"]["perSeedCalibration"]
    )
    return {
        "fit_atom_exact": fit["atom_exact_accuracy"] >= fit_gates["minimumAtomExactAccuracy"],
        "fit_lexical_sign": fit["lexical_sign_accuracy"] >= fit_gates["minimumLexicalSignAccuracy"],
        "fit_outer_operation": fit["outer_operation_accuracy"] >= fit_gates["minimumOuterOperationAccuracy"],
        "fit_compiled_truth": fit["compiled_truth_accuracy"] >= fit_gates["minimumCompiledTruthAccuracy"],
        "fit_compiled_exact_fact": fit["compiled_exact_fact_accuracy"] >= fit_gates["minimumCompiledExactFactAccuracy"],
        "calibration_atom_exact": calibration["atom_exact_accuracy"] >= calibration_gates["minimumAtomExactAccuracy"],
        "calibration_lexical_sign": calibration["lexical_sign_accuracy"] >= calibration_gates["minimumLexicalSignAccuracy"],
        "calibration_outer_operation": calibration["outer_operation_accuracy"] >= calibration_gates["minimumOuterOperationAccuracy"],
        "calibration_compiled_truth": calibration["compiled_truth_accuracy"] >= calibration_gates["minimumCompiledTruthAccuracy"],
        "calibration_compiled_exact_fact": calibration["compiled_exact_fact_accuracy"] >= calibration_gates["minimumCompiledExactFactAccuracy"],
        "calibration_relation_order": calibration["relation_order_accuracy"] >= calibration_gates["minimumRelationOrderAccuracy"],
    }


def system_qualification(
    seeds: dict[str, dict[str, dict[str, Any]]], config: dict[str, Any],
) -> dict[str, Any]:
    results = {}
    for seed, values in seeds.items():
        checks = qualification_checks(values["fit"], values["calibration"], config)
        results[seed] = {**values, "checks": checks, "passed": all(checks.values())}
    required = config["qualification"]["requiredPassingSeeds"]
    metrics = (
        "atom_exact_accuracy", "lexical_sign_accuracy", "outer_operation_accuracy",
        "compiled_truth_accuracy", "compiled_exact_fact_accuracy", "relation_order_accuracy",
    )
    return {
        "seeds": results,
        "passing_seeds": sum(row["passed"] for row in results.values()),
        "required_passing_seeds": required,
        "fit_mean": {key: mean([row["fit"][key] for row in results.values()]) for key in metrics},
        "calibration_mean": {key: mean([row["calibration"][key] for row in results.values()]) for key in metrics},
        "passed": sum(row["passed"] for row in results.values()) >= required,
    }


def select_qualified_system(systems: dict[str, dict[str, Any]], config: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    joint, independent = systems["jointCompiled"], systems["independentCompiled"]
    delta = (
        independent["calibration_mean"]["compiled_exact_fact_accuracy"]
        - joint["calibration_mean"]["compiled_exact_fact_accuracy"]
    )
    minimum = config["qualification"]["selection"]["minimumIndependentMinusJointCalibrationExactFactForModularSelection"]
    if joint["passed"] and independent["passed"]:
        selected = "independentCompiled" if delta >= minimum else "jointCompiled"
        reason = "modular_material_calibration_advantage" if selected == "independentCompiled" else "prefer_simpler_joint_no_material_modular_advantage"
    elif joint["passed"]:
        selected, reason = "jointCompiled", "only_joint_qualified"
    elif independent["passed"]:
        selected, reason = "independentCompiled", "only_independent_qualified_negative_multitask_interference"
    else:
        selected, reason = None, "no_interface_development_qualified"
    return selected, {
        "selected_system": selected, "reason": reason,
        "independent_minus_joint_calibration_compiled_exact_fact": delta,
        "minimum_modular_advantage": minimum,
    }


def decode_outputs(
    rows: Sequence[dict[str, Any]], outputs: tuple[np.ndarray, ...],
    config: dict[str, Any], compiled: bool,
) -> list[dict[str, Any]]:
    return select_predictions(
        rows, outputs, config,
        "fixed_registered_truth_compiler" if compiled else "direct_truth_head",
    )
