"""Fixed V36 readout fitting, serialization, and inference helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from v35_binding import make_ridge


COMPONENTS = ("predicate", "binding", "lexical_sign", "outer_operation")


def projection_matrix(input_dims: int, output_dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(
        0.0, 1.0 / np.sqrt(output_dims), size=(input_dims, output_dims)
    ).astype(np.float32)


def fit_component(features: np.ndarray, targets: np.ndarray, alpha: float) -> tuple[Any, dict[str, np.ndarray]]:
    model = make_ridge(alpha)
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64))
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    arrays = {
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "coef": np.asarray(classifier.coef_, dtype=np.float64),
        "intercept": np.asarray(classifier.intercept_, dtype=np.float64),
        "classes": np.asarray(classifier.classes_, dtype=np.int64),
    }
    return model, arrays


def pack_component(target: dict[str, np.ndarray], name: str, arrays: dict[str, np.ndarray]) -> None:
    for key, value in arrays.items():
        target[f"{name}__{key}"] = value


def unpack_component(source: Any, name: str) -> dict[str, np.ndarray]:
    return {key: np.asarray(source[f"{name}__{key}"]) for key in ("mean", "scale", "coef", "intercept", "classes")}


def decision_function(features: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    standardized = (
        np.asarray(features, dtype=np.float64) - parameters["mean"]
    ) / parameters["scale"]
    return standardized @ parameters["coef"].T + parameters["intercept"]


def predict_component(features: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    scores = decision_function(features, parameters)
    classes = parameters["classes"]
    if len(classes) == 2:
        binary_scores = scores if scores.ndim == 1 else scores[:, 0]
        return np.where(binary_scores > 0.0, classes[1], classes[0]).astype(np.int64)
    return classes[np.argmax(scores, axis=1)].astype(np.int64)


def parameter_shapes(parameters: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in parameters.items()}
