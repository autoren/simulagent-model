"""Prompt construction for V25 explicit truth-assessment hypotheses."""

from __future__ import annotations

from typing import Any


def truth_prompt_layout(row: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    public = row["agent_input"]
    entities = ", ".join(
        f"{entity['id']} ({entity['entity_type']})" for entity in public["entities"]
    )
    binding = public["action"]["binding"]
    prefix = (
        f"Typed entities: {entities}.\n"
        f"Action binding: actor={binding['actor']}, target={binding['target']}.\n"
        f"Evidence statement: {public['evidence_text']}\n"
        f"Candidate fact: {public['candidate_statement']}\n"
        "Assessment hypothesis: "
    )
    statement = public["assessment_statement"]
    return prefix + statement, (len(prefix), len(prefix) + len(statement))


def truth_prompt_text(row: dict[str, Any]) -> str:
    return truth_prompt_layout(row)[0]
