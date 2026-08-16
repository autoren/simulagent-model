"""V35 atom prompt, fixed projection, binding decode, and assembly metrics."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v30_language import ontology_description, predicate_specs
from v32_language import compile_truth


def atom_prompt_layout(row: dict[str, Any], config: dict[str, Any]) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    entities = row["agent_input"]["entities"]
    parts = ["Typed entities:\n"]
    for index, entity in enumerate(entities, start=1):
        parts.append(f"Entity {index}: {entity['id']} ({entity['entity_type']})\n")
    parts.extend([
        "Declared predicate ontology:\n", ontology_description(config["v32_config"]),
        "\nEvidence statement: ",
    ])
    evidence_start = len("".join(parts))
    evidence = row["agent_input"]["evidence_text"]
    parts.append(evidence)
    classes = config["atomInterface"]["predicateClasses"]
    labels = config["atomInterface"]["predicateLabelTokens"]
    options = "\n".join(f"{label}: {predicate}" for label, predicate in zip(labels, classes, strict=True))
    parts.extend([
        "\nIdentify the declared predicate and its canonical typed arguments. For relations, "
        "canonical order is predicate(source, target), including under inverse or passive wording.\n",
        f"Predicate options:\n{options}\nCanonical atom representation:",
    ])
    content = "".join(parts)
    spans: dict[str, list[tuple[int, int]]] = {}
    for entity in entities:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(entity['id'])}(?![A-Za-z0-9_])")
        spans[entity["id"]] = [
            (evidence_start + match.start(), evidence_start + match.end())
            for match in pattern.finditer(evidence)
        ]
    return content, spans


def make_ridge(alpha: float):
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(
            alpha=float(alpha), class_weight="balanced", solver="lsqr", tol=1e-4,
        ),
    )


def fixed_gaussian_projection(features: np.ndarray, output_dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(
        0.0, 1.0 / np.sqrt(output_dims), size=(features.shape[1], output_dims)
    ).astype(np.float32)
    return np.asarray(features, dtype=np.float32) @ matrix


def select_report(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return sorted(reports, key=lambda row: (
        -row["mean_group_cv_primary_accuracy"],
        -row["minimum_group_cv_primary_accuracy"], -row["alpha"],
    ))[0]


def select_new_predicate_method(methods: dict[str, dict[str, Any]]) -> str:
    eligible = ("atomHiddenRidge", "nativePredicateLogitRidge")
    return sorted(eligible, key=lambda name: (
        -methods[name]["selected_cv"]["mean_group_cv_primary_accuracy"],
        -methods[name]["selected_cv"]["minimum_group_cv_primary_accuracy"],
        -methods[name]["selected_cv"]["alpha"], name,
    ))[0]


def select_binding_method(methods: dict[str, dict[str, Any]]) -> str:
    return sorted(methods, key=lambda name: (
        -methods[name]["selected_cv"]["mean_group_cv_primary_accuracy"],
        -methods[name]["selected_cv"]["minimum_group_cv_primary_accuracy"],
        -methods[name]["selected_cv"]["alpha"], name,
    ))[0]


def build_entity_examples(
    rows: Sequence[dict[str, Any]], entity_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, targets, row_indices, entity_indices = [], [], [], []
    for row_index, row in enumerate(rows):
        entities = [entity["id"] for entity in row["agent_input"]["entities"]]
        arguments = row["target"]["arguments"]
        for entity_index, entity in enumerate(entities):
            role = 1 if entity == arguments[0] else (2 if len(arguments) > 1 and entity == arguments[1] else 0)
            features.append(entity_features[row_index, entity_index])
            targets.append(role); row_indices.append(row_index); entity_indices.append(entity_index)
    return (
        np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64),
        np.asarray(row_indices, dtype=np.int64), np.asarray(entity_indices, dtype=np.int64),
    )


def decode_bindings(
    rows: Sequence[dict[str, Any]], row_indices: Sequence[int],
    predicate_indices: Sequence[int], role_scores: np.ndarray,
    example_rows: np.ndarray, config: dict[str, Any],
) -> list[tuple[int, int | None]]:
    predicates = config["atomInterface"]["predicateClasses"]
    specs = predicate_specs(config["v32_config"])
    decoded = []
    for row_index, predicate_index in zip(row_indices, predicate_indices, strict=True):
        row = rows[int(row_index)]
        positions = np.flatnonzero(example_rows == row_index)
        entities = row["agent_input"]["entities"]
        spec = specs[predicates[int(predicate_index)]]
        if spec["kind"] == "unary":
            valid = [index for index, entity in enumerate(entities) if entity["entity_type"] == spec["entityType"]]
            decoded.append((max(valid, key=lambda index: role_scores[positions[index], 1]), None))
        else:
            sources = [index for index, entity in enumerate(entities) if entity["entity_type"] == spec["sourceType"]]
            targets = [index for index, entity in enumerate(entities) if entity["entity_type"] == spec["targetType"]]
            pairs = [(source, target) for source in sources for target in targets if source != target]
            decoded.append(max(
                pairs,
                key=lambda pair: role_scores[positions[pair[0]], 1] + role_scores[positions[pair[1]], 2],
            ))
    return decoded


def score_assembly(
    rows: Sequence[dict[str, Any]], row_indices: Sequence[int],
    predicate_indices: Sequence[int], bindings: Sequence[tuple[int, int | None]],
    sign_indices: Sequence[int], operation_indices: Sequence[int], config: dict[str, Any],
) -> dict[str, float]:
    predicates = config["atomInterface"]["predicateClasses"]
    signs = config["v32_config"]["sharedHead"]["lexicalSignClasses"]
    operations = config["v32_config"]["sharedHead"]["outerOperationClasses"]
    values = {key: [] for key in (
        "predicate", "argument1", "argument2", "atom", "relation_order", "lexical_sign",
        "outer_operation", "compiled_truth", "compiled_exact_fact",
    )}
    for row_index, predicate_index, binding, sign_index, operation_index in zip(
        row_indices, predicate_indices, bindings, sign_indices, operation_indices, strict=True,
    ):
        row = rows[int(row_index)]
        target = row["target"]
        entities = [entity["id"] for entity in row["agent_input"]["entities"]]
        argument1, argument2 = binding
        predicate_ok = predicates[int(predicate_index)] == target["predicate"]
        argument1_ok = entities[argument1] == target["arguments"][0]
        argument2_ok = (
            argument2 is None and len(target["arguments"]) == 1
        ) or (
            argument2 is not None and len(target["arguments"]) == 2
            and entities[argument2] == target["arguments"][1]
        )
        atom_ok = predicate_ok and argument1_ok and argument2_ok
        sign = signs[int(sign_index)]; operation = operations[int(operation_index)]
        sign_ok = sign == target["factorization"]["lexical_sign"]
        operation_ok = operation == target["factorization"]["outer_operation"]
        truth_ok = compile_truth(sign, operation, config["v32_config"]) == target["truth_status"]
        values["predicate"].append(predicate_ok); values["argument1"].append(argument1_ok)
        values["argument2"].append(argument2_ok); values["atom"].append(atom_ok)
        if target["predicate_kind"] == "relation":
            values["relation_order"].append(argument1_ok and argument2_ok)
        values["lexical_sign"].append(sign_ok); values["outer_operation"].append(operation_ok)
        values["compiled_truth"].append(truth_ok); values["compiled_exact_fact"].append(atom_ok and truth_ok)
    return {f"{key}_accuracy": float(np.mean(value)) for key, value in values.items()}


def qualification(modular: dict[str, float], legacy: dict[str, float], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["qualification"]
    gain = modular["compiled_exact_fact_accuracy"] - legacy["compiled_exact_fact_accuracy"]
    checks = {
        "predicate": modular["predicate_accuracy"] >= gates["minimumCalibrationPredicateAccuracy"],
        "atom": modular["atom_accuracy"] >= gates["minimumCalibrationAtomExactAccuracy"],
        "relation_order": modular["relation_order_accuracy"] >= gates["minimumCalibrationRelationOrderAccuracy"],
        "lexical_sign": modular["lexical_sign_accuracy"] >= gates["minimumCalibrationLexicalSignAccuracy"],
        "outer_operation": modular["outer_operation_accuracy"] >= gates["minimumCalibrationOuterOperationAccuracy"],
        "compiled_truth": modular["compiled_truth_accuracy"] >= gates["minimumCalibrationCompiledTruthAccuracy"],
        "compiled_exact_fact": modular["compiled_exact_fact_accuracy"] >= gates["minimumCalibrationCompiledExactFactAccuracy"],
        "gain_over_legacy": gain >= gates["minimumCalibrationExactFactGainOverLegacyAssembly"],
    }
    return {"passed": all(checks.values()), "checks": checks, "calibration_exact_fact_gain_over_legacy": gain}
