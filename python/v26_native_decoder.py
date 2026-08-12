"""Prompt and label utilities for V26 native truth decoding."""

from __future__ import annotations

from typing import Any, Sequence


def decoder_prompt(row: dict[str, Any]) -> str:
    public = row["agent_input"]
    entities = ", ".join(
        f"{entity['id']} ({entity['entity_type']})" for entity in public["entities"]
    )
    binding = public["action"]["binding"]
    return (
        f"Typed entities: {entities}.\n"
        f"Action binding: actor={binding['actor']}, target={binding['target']}.\n"
        f"Evidence statement: {public['evidence_text']}\n"
        f"Candidate fact: {public['candidate_statement']}\n"
        "Classification:"
    )


def select_label(logits: Sequence[float], labels: Sequence[dict[str, str]]) -> dict[str, str]:
    if len(logits) != len(labels):
        raise ValueError("V26 logits and labels must have equal lengths")
    index = max(range(len(logits)), key=lambda value: (float(logits[value]), -value))
    return labels[index]
