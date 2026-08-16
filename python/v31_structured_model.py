"""Shared structured pointer head and feature utilities for V31."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from extract_v10_features_mlx import chat_prompt
from extract_v22r2_relational_features_mlx import token_spans
from v30_language import predicate_specs
from v31_language import representation_prompt_layout


ENTITY_TYPE_ORDER = ("unit", "hub")


class StructuredPointerHead(nn.Module):
    def __init__(self, input_dims: int, width: int, predicates: int, truths: int):
        super().__init__()
        self.clause_projection = nn.Linear(input_dims, width)
        self.entity_projection = nn.Linear(input_dims, width)
        self.clause_norm = nn.LayerNorm(width)
        self.entity_norm = nn.LayerNorm(width)
        self.predicate_head = nn.Linear(width, predicates)
        self.truth_head = nn.Linear(width, truths)
        self.role_interaction = nn.Linear(width * 4, width)
        self.argument1_head = nn.Linear(width, 1)
        self.argument2_head = nn.Linear(width, 1)
        self.na_embedding = mx.random.normal(shape=(width,)) * 0.02

    def __call__(
        self, clause_features: mx.array, entity_features: mx.array,
        entity_mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        clause = self.clause_norm(nn.gelu(self.clause_projection(clause_features)))
        entities = self.entity_norm(nn.gelu(self.entity_projection(entity_features)))
        expanded = mx.broadcast_to(clause[:, None, :], entities.shape)
        interaction = nn.gelu(self.role_interaction(mx.concatenate([
            expanded, entities, mx.abs(expanded - entities), expanded * entities,
        ], axis=-1)))
        argument1 = mx.squeeze(self.argument1_head(interaction), axis=-1)
        na = mx.broadcast_to(self.na_embedding[None, None, :], (len(clause), 1, self.na_embedding.shape[0]))
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
        return self.predicate_head(clause), argument1, argument2, self.truth_head(clause)


class AdaptedStructuredGrounder(nn.Module):
    """Register one language-model core and the identical V31 pointer head."""

    def __init__(self, backbone: nn.Module, head: StructuredPointerHead, max_entities: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.max_entities = max_entities

    def __call__(
        self, tokens: mx.array, entity_spans: Sequence[Sequence[int]],
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        hidden = self.backbone(tokens)[0]
        clause, entities, mask = features_from_hidden(
            hidden, entity_spans, self.max_entities
        )
        return self.head(clause[None, :], entities[None, :, :], mask[None, :])


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
            token
            for index in range(len(character_spans[entity["id"]]))
            for token in mapped[f"{entity['id']}|{index}"]
        })
        entity_spans.append(indices)
    return tokens, entity_spans, content


def features_from_hidden(
    hidden: mx.array, entity_spans: Sequence[Sequence[int]], max_entities: int,
) -> tuple[mx.array, mx.array, mx.array]:
    hidden32 = hidden.astype(mx.float32)
    clause = hidden32[-1]
    entity_values = [mx.mean(hidden32[mx.array(span)], axis=0) for span in entity_spans]
    padding = max_entities - len(entity_values)
    if padding < 0:
        raise ValueError("V31 record exceeds maximum entity count")
    if padding:
        entity_values.extend(mx.zeros_like(clause) for _ in range(padding))
    entities = mx.stack(entity_values)
    mask = mx.array([True] * len(entity_spans) + [False] * padding)
    return clause, entities, mask


def target_arrays(row: dict[str, Any], config: dict[str, Any]) -> dict[str, int]:
    target = row["target"]
    entities = [entity["id"] for entity in row["agent_input"]["entities"]]
    max_entities = max(config["construction"]["entityCounts"])
    return {
        "predicate": config["sharedStructuredHead"]["predicateClasses"].index(target["predicate"]),
        "argument1": entities.index(target["arguments"][0]),
        "argument2": (
            entities.index(target["arguments"][1])
            if target["predicate_kind"] == "relation" else max_entities
        ),
        "truth": config["sharedStructuredHead"]["truthClasses"].index(target["truth_status"]),
    }


def type_masks(
    rows: Sequence[dict[str, Any]], predicate_indices: Sequence[int],
    argument1_indices: Sequence[int], config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    max_entities = max(config["construction"]["entityCounts"])
    predicates = config["sharedStructuredHead"]["predicateClasses"]
    specs = predicate_specs(config)
    mask1 = np.zeros((len(rows), max_entities), dtype=bool)
    mask2 = np.zeros((len(rows), max_entities + 1), dtype=bool)
    for row_index, (row, predicate_index, argument1) in enumerate(zip(
        rows, predicate_indices, argument1_indices, strict=True
    )):
        predicate = predicates[int(predicate_index)]
        spec = specs[predicate]
        entities = row["agent_input"]["entities"]
        if spec["kind"] == "unary":
            required1 = spec["entityType"]
            mask2[row_index, max_entities] = True
        else:
            required1 = spec["sourceType"]
            for index, entity in enumerate(entities):
                if entity["entity_type"] == spec["targetType"] and index != int(argument1):
                    mask2[row_index, index] = True
        for index, entity in enumerate(entities):
            if entity["entity_type"] == required1:
                mask1[row_index, index] = True
    return mask1, mask2


def apply_type_masks(
    argument1: np.ndarray, argument2: np.ndarray, rows: Sequence[dict[str, Any]],
    predicate_indices: np.ndarray, config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    provisional_mask1, _ = type_masks(rows, predicate_indices, np.zeros(len(rows), dtype=int), config)
    masked1 = np.where(provisional_mask1, argument1, -np.inf)
    selected1 = np.argmax(masked1, axis=1)
    _, mask2 = type_masks(rows, predicate_indices, selected1, config)
    masked2 = np.where(mask2, argument2, -np.inf)
    return masked1, masked2, selected1


def select_predictions(
    rows: Sequence[dict[str, Any]], outputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    predicate_logits, argument1_logits, argument2_logits, truth_logits = outputs
    predicate_indices = np.argmax(predicate_logits, axis=1)
    masked1, masked2, argument1_indices = apply_type_masks(
        argument1_logits, argument2_logits, rows, predicate_indices, config
    )
    argument2_indices = np.argmax(masked2, axis=1)
    truth_indices = np.argmax(truth_logits, axis=1)
    max_entities = max(config["construction"]["entityCounts"])
    predictions = []
    for index, row in enumerate(rows):
        entities = [entity["id"] for entity in row["agent_input"]["entities"]]
        predicate = config["sharedStructuredHead"]["predicateClasses"][int(predicate_indices[index])]
        argument2 = "N/A" if argument2_indices[index] == max_entities else entities[int(argument2_indices[index])]
        predictions.append({
            "id": row["id"], "scene_id": row["scene_id"], "split": row["split"],
            "selected_fields": {
                "predicate": predicate,
                "argument_1": entities[int(argument1_indices[index])],
                "argument_2": argument2,
                "truth_status": config["sharedStructuredHead"]["truthClasses"][int(truth_indices[index])],
            },
            "logits": {
                "predicate": predicate_logits[index].tolist(),
                "argument1": argument1_logits[index].tolist(),
                "argument2": argument2_logits[index].tolist(),
                "truth": truth_logits[index].tolist(),
            },
        })
    return predictions


def class_weights(rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    targets = [target_arrays(row, config) for row in rows]
    result = []
    for key, classes in (
        ("predicate", len(config["sharedStructuredHead"]["predicateClasses"])),
        ("truth", len(config["sharedStructuredHead"]["truthClasses"])),
    ):
        counts = np.bincount([row[key] for row in targets], minlength=classes).astype(np.float32)
        result.append((counts.sum() / (classes * counts)).astype(np.float32))
    return result[0], result[1]


def loss_from_outputs(
    outputs: tuple[mx.array, mx.array, mx.array, mx.array],
    rows: Sequence[dict[str, Any]], config: dict[str, Any],
    predicate_weights: mx.array, truth_weights: mx.array,
) -> tuple[mx.array, dict[str, mx.array]]:
    predicate, argument1, argument2, truth = outputs
    targets = [target_arrays(row, config) for row in rows]
    predicate_target = mx.array([row["predicate"] for row in targets])
    argument1_target = mx.array([row["argument1"] for row in targets])
    argument2_target = mx.array([row["argument2"] for row in targets])
    truth_target = mx.array([row["truth"] for row in targets])
    mask1_np, mask2_np = type_masks(
        rows, np.asarray([row["predicate"] for row in targets]),
        np.asarray([row["argument1"] for row in targets]), config,
    )
    negative = mx.array(-1e9, dtype=argument1.dtype)
    argument1 = mx.where(mx.array(mask1_np), argument1, negative)
    argument2 = mx.where(mx.array(mask2_np), argument2, negative)
    predicate_losses = nn.losses.cross_entropy(predicate, predicate_target, reduction="none")
    truth_losses = nn.losses.cross_entropy(truth, truth_target, reduction="none")
    parts = {
        "predicate": mx.mean(predicate_losses * predicate_weights[predicate_target]),
        "argument1": mx.mean(nn.losses.cross_entropy(argument1, argument1_target, reduction="none")),
        "argument2": mx.mean(nn.losses.cross_entropy(argument2, argument2_target, reduction="none")),
        "truth": mx.mean(truth_losses * truth_weights[truth_target]),
    }
    weights = config["sharedStructuredHead"]["loss"]
    total = sum(float(weights[key]) * value for key, value in parts.items())
    return total, parts


def make_loss(
    rows: Sequence[dict[str, Any]], config: dict[str, Any],
    predicate_weights: mx.array, truth_weights: mx.array,
) -> Any:
    def loss_fn(
        model: StructuredPointerHead, clause: mx.array, entities: mx.array, entity_mask: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        return loss_from_outputs(
            model(clause, entities, entity_mask), rows, config,
            predicate_weights, truth_weights,
        )

    return loss_fn
