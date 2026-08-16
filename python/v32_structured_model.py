"""Shared six-output frozen semantic parser and deterministic V32 decoder."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from extract_v10_features_mlx import chat_prompt
from extract_v22r2_relational_features_mlx import token_spans
from v30_language import predicate_specs
from v32_language import compile_truth, representation_prompt_layout


class FactorizedPointerHead(nn.Module):
    """One capacity-matched head; objectives decide which outputs receive loss."""

    def __init__(
        self, input_dims: int, width: int, predicates: int, truths: int,
        lexical_signs: int, outer_operations: int,
    ):
        super().__init__()
        self.clause_projection = nn.Linear(input_dims, width)
        self.entity_projection = nn.Linear(input_dims, width)
        self.clause_norm = nn.LayerNorm(width)
        self.entity_norm = nn.LayerNorm(width)
        self.predicate_head = nn.Linear(width, predicates)
        self.truth_head = nn.Linear(width, truths)
        self.lexical_sign_head = nn.Linear(width, lexical_signs)
        self.outer_operation_head = nn.Linear(width, outer_operations)
        self.role_interaction = nn.Linear(width * 4, width)
        self.argument1_head = nn.Linear(width, 1)
        self.argument2_head = nn.Linear(width, 1)
        self.na_embedding = mx.random.normal(shape=(width,)) * 0.02

    def __call__(
        self, clause_features: mx.array, entity_features: mx.array,
        entity_mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        clause = self.clause_norm(nn.gelu(self.clause_projection(clause_features)))
        entities = self.entity_norm(nn.gelu(self.entity_projection(entity_features)))
        expanded = mx.broadcast_to(clause[:, None, :], entities.shape)
        interaction = nn.gelu(self.role_interaction(mx.concatenate([
            expanded, entities, mx.abs(expanded - entities), expanded * entities,
        ], axis=-1)))
        argument1 = mx.squeeze(self.argument1_head(interaction), axis=-1)
        na = mx.broadcast_to(
            self.na_embedding[None, None, :], (len(clause), 1, self.na_embedding.shape[0])
        )
        expanded_na = mx.broadcast_to(clause[:, None, :], na.shape)
        na_interaction = nn.gelu(self.role_interaction(mx.concatenate([
            expanded_na, na, mx.abs(expanded_na - na), expanded_na * na,
        ], axis=-1)))
        argument2_entities = mx.squeeze(self.argument2_head(interaction), axis=-1)
        argument2_na = mx.squeeze(self.argument2_head(na_interaction), axis=(1, 2))
        negative = mx.array(-1e9, dtype=argument1.dtype)
        argument1 = mx.where(entity_mask, argument1, negative)
        argument2_entities = mx.where(entity_mask, argument2_entities, negative)
        argument2 = mx.concatenate([argument2_entities, argument2_na[:, None]], axis=1)
        return (
            self.predicate_head(clause), argument1, argument2, self.truth_head(clause),
            self.lexical_sign_head(clause), self.outer_operation_head(clause),
        )


def make_head(config: dict[str, Any]) -> FactorizedPointerHead:
    head = config["sharedHead"]
    return FactorizedPointerHead(
        config["model"]["hiddenSize"], head["width"], len(head["predicateClasses"]),
        len(head["truthClasses"]), len(head["lexicalSignClasses"]),
        len(head["outerOperationClasses"]),
    )


def prompt_tokens_and_entity_spans(
    row: dict[str, Any], config: dict[str, Any], tokenizer: Any,
) -> tuple[list[int], list[list[int]], str]:
    content, character_spans = representation_prompt_layout(row, config)
    prompt = chat_prompt(content, config["model"]["systemPrompt"], tokenizer)
    flat = {
        f"{entity}|{index}": span
        for entity, spans in character_spans.items() for index, span in enumerate(spans)
    }
    tokens, mapped = token_spans(prompt, content, flat, tokenizer)
    entity_spans = []
    for entity in row["agent_input"]["entities"]:
        indices = sorted({
            token for index in range(len(character_spans[entity["id"]]))
            for token in mapped[f"{entity['id']}|{index}"]
        })
        entity_spans.append(indices)
    return tokens, entity_spans, content


def features_from_hidden(
    hidden: mx.array, entity_spans: Sequence[Sequence[int]], max_entities: int,
) -> tuple[mx.array, mx.array, mx.array]:
    hidden32 = hidden.astype(mx.float32)
    clause = hidden32[-1]
    values = [mx.mean(hidden32[mx.array(span)], axis=0) for span in entity_spans]
    padding = max_entities - len(values)
    if padding < 0:
        raise ValueError("V32 record exceeds maximum entity count")
    values.extend(mx.zeros_like(clause) for _ in range(padding))
    return clause, mx.stack(values), mx.array([True] * len(entity_spans) + [False] * padding)


def target_arrays(row: dict[str, Any], config: dict[str, Any]) -> dict[str, int]:
    target = row["target"]
    entities = [entity["id"] for entity in row["agent_input"]["entities"]]
    maximum = max(config["construction"]["entityCounts"])
    factor = target["factorization"]
    head = config["sharedHead"]
    return {
        "predicate": head["predicateClasses"].index(target["predicate"]),
        "argument1": entities.index(target["arguments"][0]),
        "argument2": entities.index(target["arguments"][1]) if target["predicate_kind"] == "relation" else maximum,
        "truth": head["truthClasses"].index(target["truth_status"]),
        "lexicalSign": head["lexicalSignClasses"].index(factor["lexical_sign"]),
        "outerOperation": head["outerOperationClasses"].index(factor["outer_operation"]),
    }


def type_masks(
    rows: Sequence[dict[str, Any]], predicate_indices: Sequence[int],
    argument1_indices: Sequence[int], config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(config["construction"]["entityCounts"])
    predicates, specs = config["sharedHead"]["predicateClasses"], predicate_specs(config)
    mask1 = np.zeros((len(rows), maximum), dtype=bool)
    mask2 = np.zeros((len(rows), maximum + 1), dtype=bool)
    for row_index, (row, predicate_index, argument1) in enumerate(zip(
        rows, predicate_indices, argument1_indices, strict=True
    )):
        spec = specs[predicates[int(predicate_index)]]
        if spec["kind"] == "unary":
            required1 = spec["entityType"]
            mask2[row_index, maximum] = True
        else:
            required1 = spec["sourceType"]
            for index, entity in enumerate(row["agent_input"]["entities"]):
                if entity["entity_type"] == spec["targetType"] and index != int(argument1):
                    mask2[row_index, index] = True
        for index, entity in enumerate(row["agent_input"]["entities"]):
            if entity["entity_type"] == required1:
                mask1[row_index, index] = True
    return mask1, mask2


def apply_type_masks(
    argument1: np.ndarray, argument2: np.ndarray, rows: Sequence[dict[str, Any]],
    predicate_indices: np.ndarray, config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask1, _ = type_masks(rows, predicate_indices, np.zeros(len(rows), dtype=int), config)
    masked1 = np.where(mask1, argument1, -np.inf)
    selected1 = np.argmax(masked1, axis=1)
    _, mask2 = type_masks(rows, predicate_indices, selected1, config)
    return masked1, np.where(mask2, argument2, -np.inf), selected1


def select_predictions(
    rows: Sequence[dict[str, Any]], outputs: tuple[np.ndarray, ...],
    config: dict[str, Any], truth_decoding: str,
) -> list[dict[str, Any]]:
    predicate, argument1, argument2, truth, sign, operation = outputs
    predicate_indices = np.argmax(predicate, axis=1)
    _, masked2, argument1_indices = apply_type_masks(argument1, argument2, rows, predicate_indices, config)
    argument2_indices = np.argmax(masked2, axis=1)
    truth_indices, sign_indices, operation_indices = map(
        lambda values: np.argmax(values, axis=1), (truth, sign, operation)
    )
    head, maximum = config["sharedHead"], max(config["construction"]["entityCounts"])
    result = []
    for index, row in enumerate(rows):
        entities = [entity["id"] for entity in row["agent_input"]["entities"]]
        lexical_sign = head["lexicalSignClasses"][int(sign_indices[index])]
        outer_operation = head["outerOperationClasses"][int(operation_indices[index])]
        direct_truth = head["truthClasses"][int(truth_indices[index])]
        selected_truth = (
            direct_truth if truth_decoding == "direct_truth_head"
            else compile_truth(lexical_sign, outer_operation, config)
        )
        result.append({
            "id": row["id"], "scene_id": row["scene_id"], "split": row["split"],
            "truth_decoding": truth_decoding,
            "selected_fields": {
                "predicate": head["predicateClasses"][int(predicate_indices[index])],
                "argument_1": entities[int(argument1_indices[index])],
                "argument_2": "N/A" if argument2_indices[index] == maximum else entities[int(argument2_indices[index])],
                "truth_status": selected_truth,
            },
            "selected_intermediates": {
                "lexical_sign": lexical_sign, "outer_operation": outer_operation,
                "direct_truth_status": direct_truth,
            },
            "logits": {
                "predicate": predicate[index].tolist(), "argument1": argument1[index].tolist(),
                "argument2": argument2[index].tolist(), "truth": truth[index].tolist(),
                "lexical_sign": sign[index].tolist(), "outer_operation": operation[index].tolist(),
            },
        })
    return result


def class_weights(rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, np.ndarray]:
    targets = [target_arrays(row, config) for row in rows]
    classes = {
        "predicate": len(config["sharedHead"]["predicateClasses"]),
        "truth": len(config["sharedHead"]["truthClasses"]),
        "lexicalSign": len(config["sharedHead"]["lexicalSignClasses"]),
        "outerOperation": len(config["sharedHead"]["outerOperationClasses"]),
    }
    result = {}
    for key, count in classes.items():
        observed = np.bincount([row[key] for row in targets], minlength=count).astype(np.float32)
        result[key] = (observed.sum() / (count * observed)).astype(np.float32)
    return result


def loss_from_outputs(
    outputs: tuple[mx.array, ...], rows: Sequence[dict[str, Any]], config: dict[str, Any],
    class_weight_values: dict[str, mx.array], objective: Sequence[str],
) -> tuple[mx.array, dict[str, mx.array]]:
    predicate, argument1, argument2, truth, sign, operation = outputs
    targets = [target_arrays(row, config) for row in rows]
    arrays = {key: mx.array([row[key] for row in targets]) for key in targets[0]}
    mask1, mask2 = type_masks(
        rows, np.asarray([row["predicate"] for row in targets]),
        np.asarray([row["argument1"] for row in targets]), config,
    )
    negative = mx.array(-1e9, dtype=argument1.dtype)
    argument1, argument2 = mx.where(mx.array(mask1), argument1, negative), mx.where(mx.array(mask2), argument2, negative)
    raw = {
        "predicate": mx.mean(nn.losses.cross_entropy(predicate, arrays["predicate"], reduction="none") * class_weight_values["predicate"][arrays["predicate"]]),
        "argument1": mx.mean(nn.losses.cross_entropy(argument1, arrays["argument1"], reduction="none")),
        "argument2": mx.mean(nn.losses.cross_entropy(argument2, arrays["argument2"], reduction="none")),
        "truth": mx.mean(nn.losses.cross_entropy(truth, arrays["truth"], reduction="none") * class_weight_values["truth"][arrays["truth"]]),
        "lexicalSign": mx.mean(nn.losses.cross_entropy(sign, arrays["lexicalSign"], reduction="none") * class_weight_values["lexicalSign"][arrays["lexicalSign"]]),
        "outerOperation": mx.mean(nn.losses.cross_entropy(operation, arrays["outerOperation"], reduction="none") * class_weight_values["outerOperation"][arrays["outerOperation"]]),
    }
    weights = config["sharedHead"]["lossWeights"]
    total = sum(float(weights[key]) * raw[key] for key in objective)
    return total, raw


def make_loss(
    model: FactorizedPointerHead, rows: Sequence[dict[str, Any]], config: dict[str, Any],
    class_weight_values: dict[str, mx.array], objective: Sequence[str],
) -> Any:
    def loss_fn(
        clause: mx.array, entities: mx.array, mask: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        return loss_from_outputs(model(clause, entities, mask), rows, config, class_weight_values, objective)
    return loss_fn
