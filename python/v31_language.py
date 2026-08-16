"""Language templates, prompts, and canonical utilities for V31."""

from __future__ import annotations

import re
from typing import Any, Sequence

from v30_language import (
    DISTRACTOR, ENTITY_ALIASES, LABEL_TOKENS, TRUTH_VALUES, atom_key, canonical_json,
    deterministic_shuffle, fact_phrases, field_options, log_softmax, opaque_id,
    ontology_description, positive_candidate_statement, predicate_specs, select_option,
    sha256_text,
)


SURFACE_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "affirmative_gold": {
        "fit_a": ("Inspection records {gold} as the current condition.", "short"),
        "fit_b": ("The latest observation establishes the following fact: {gold}.", "medium"),
        "fit_c": ("What the instruments presently indicate is that {gold}.", "medium"),
        "fit_d": ("For the state now under review, the accurate description is: {gold}.", "long"),
        "fit_e": ("A direct reading of the system confirms {gold}.", "medium"),
        "cal_a": ("The verified condition, according to the inspection log, is that {gold}.", "long"),
        "eval_a": ("The fact supported by the current observation is that {gold}.", "medium"),
        "eval_b": ("On inspection, the system proves to have this condition: {gold}.", "long"),
        "eval_c": ("The most recent state report records that {gold}.", "medium"),
        "eval_d": ("The description that agrees with the evidence is simply this: {gold}.", "long"),
        "eval_e": ("At present, observation verifies {gold}.", "short"),
    },
    "negated_opposite": {
        "fit_a": ("The observation contradicts the assertion that {opposite}.", "medium"),
        "fit_b": ("It would misstate the evidence to claim that {opposite}.", "medium"),
        "fit_c": ("The proposition that {opposite} does not hold.", "medium"),
        "fit_d": ("Inspection gives us reason to reject, rather than accept, the claim that {opposite}.", "long"),
        "fit_e": ("The current state cannot accurately be described by saying {opposite}.", "long"),
        "cal_a": ("The report saying that {opposite} is inconsistent with what was observed.", "long"),
        "eval_a": ("Evidence rules against the description that {opposite}.", "medium"),
        "eval_b": ("Treat the assertion that {opposite} as incorrect for the current state.", "long"),
        "eval_c": ("The inspection does not support—and in fact rejects—the claim that {opposite}.", "long"),
        "eval_d": ("Saying that {opposite} would give the reverse of the observed condition.", "long"),
        "eval_e": ("What is observed makes the statement that {opposite} false.", "medium"),
    },
    "contrastive_both": {
        "fit_a": ("The evidence indicates {gold}, whereas {opposite} is inaccurate.", "medium"),
        "fit_b": ("Choose the description {gold} over the contrary description {opposite}.", "medium"),
        "fit_c": ("Inspection supports {gold}; by contrast, it does not support {opposite}.", "long"),
        "fit_d": ("Although the alternative says {opposite}, the observed condition is that {gold}.", "long"),
        "fit_e": ("It is {gold}, and not—as another account suggests—{opposite}.", "long"),
        "cal_a": ("Of the opposed descriptions, evidence favors {gold} and rejects {opposite}.", "long"),
        "eval_a": ("Rather than {opposite}, the state recorded by inspection is that {gold}.", "long"),
        "eval_b": ("The valid side of the contrast is {gold}; the invalid side is {opposite}.", "long"),
        "eval_c": ("Observation distinguishes the two accounts: {gold} is right, while {opposite} is wrong.", "long"),
        "eval_d": ("Where one might report {opposite}, the evidence instead demonstrates {gold}.", "long"),
        "eval_e": ("The system exhibits {gold}, not {opposite}.", "short"),
    },
    "double_negation": {
        "fit_a": ("It is incorrect to deny that {gold}.", "short"),
        "fit_b": ("The statement that {gold} cannot accurately be called false.", "medium"),
        "fit_c": ("No valid reading would reject the proposition that {gold}.", "medium"),
        "fit_d": ("A denial of the claim that {gold} would contradict the inspection.", "long"),
        "fit_e": ("It is not wrong to report that {gold}.", "short"),
        "cal_a": ("The evidence leaves no basis for calling the proposition that {gold} untrue.", "long"),
        "eval_a": ("The claim that {gold} is not one that inspection permits us to deny.", "long"),
        "eval_b": ("It would itself be false to reject the statement that {gold}.", "medium"),
        "eval_c": ("No accurate interpretation can treat the proposition that {gold} as incorrect.", "long"),
        "eval_d": ("The evidence does not allow the assertion that {gold} to be negated.", "long"),
        "eval_e": ("Calling it untrue that {gold} would be a mistake.", "medium"),
    },
    "explicit_unknown": {
        "fit_a": ("Inspection provides no determination of whether {positive}.", "medium"),
        "fit_b": ("The present record leaves open the question of whether {positive}.", "medium"),
        "fit_c": ("Neither confirmation nor denial is available for the claim that {positive}.", "long"),
        "fit_d": ("Available observations are insufficient to establish or refute whether {positive}.", "long"),
        "fit_e": ("The status of whether {positive} has not been resolved.", "medium"),
        "cal_a": ("Inspection leaves both possibilities open as to whether {positive}.", "long"),
        "eval_a": ("The record cannot tell us whether {positive}.", "short"),
        "eval_b": ("It remains unsettled, on the available evidence, whether {positive}.", "medium"),
        "eval_c": ("Observation establishes neither that {positive} nor that this is false.", "long"),
        "eval_d": ("No definite truth value can be assigned to the proposition that {positive}.", "long"),
        "eval_e": ("The evidence is compatible with either answer to whether {positive}.", "long"),
    },
}

V31_DISTRACTOR = "An unrelated service entry concerns inventory timing and carries no state information."


def construction_hash(operator: str, surface_name: str) -> str:
    template = SURFACE_TEMPLATES[operator][surface_name][0]
    normalized = re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return sha256_text(f"{operator}|{normalized}")


def render_evidence(
    predicate: str, arguments: Sequence[str], truth_status: str, semantic_operator: str,
    surface_name: str, orientation: str, distractor: bool, config: dict[str, Any],
) -> tuple[str, str]:
    positive, negative = fact_phrases(predicate, arguments, orientation, config)
    if semantic_operator == "explicit_unknown":
        if truth_status != "unknown":
            raise ValueError("Explicit-unknown surfaces require unknown truth status")
        gold, opposite = positive, negative
    else:
        if truth_status not in ("true", "false"):
            raise ValueError("Known-value surfaces cannot carry unknown truth status")
        gold = positive if truth_status == "true" else negative
        opposite = negative if truth_status == "true" else positive
    template, length = SURFACE_TEMPLATES[semantic_operator][surface_name]
    text = template.format(
        positive=positive, negative=negative, gold=gold, opposite=opposite,
        gold_cap=gold[:1].upper() + gold[1:],
    )
    if distractor:
        text = f"{V31_DISTRACTOR} {text}"
    return text, length


def representation_prompt_layout(row: dict[str, Any], config: dict[str, Any]) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    entities = row["agent_input"]["entities"]
    parts = ["Typed entities:\n"]
    for index, entity in enumerate(entities, start=1):
        parts.append(f"Entity {index}: {entity['id']} ({entity['entity_type']})\n")
    parts.append("Declared predicate ontology:\n")
    parts.append(ontology_description(config))
    parts.append("\nEvidence statement: ")
    evidence_start = sum(len(value) for value in parts)
    evidence = row["agent_input"]["evidence_text"]
    parts.append(evidence)
    parts.append("\nCanonical signed-fact representation:")
    content = "".join(parts)
    spans: dict[str, list[tuple[int, int]]] = {entity["id"]: [] for entity in entities}
    for entity in entities:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(entity['id'])}(?![A-Za-z0-9_])")
        spans[entity["id"]] = [
            (match.start(), match.end()) for match in pattern.finditer(content)
        ]
        if not spans[entity["id"]]:
            raise RuntimeError(f"Entity absent from V31 representation prompt: {entity['id']}")
    return content, spans


def zero_shot_field_prompt(
    row: dict[str, Any], field: str, config: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    if field == "predicate":
        values = config["sharedStructuredHead"]["predicateClasses"]
    elif field == "argument_1":
        values = [entity["id"] for entity in row["agent_input"]["entities"]]
    elif field == "argument_2":
        values = [entity["id"] for entity in row["agent_input"]["entities"]] + ["N/A"]
    elif field == "truth_status":
        values = config["sharedStructuredHead"]["truthClasses"]
    else:
        raise ValueError(f"Unknown V31 zero-shot field: {field}")
    if len(values) > len(LABEL_TOKENS):
        raise ValueError("V31 field exceeds the locked label inventory")
    options = [
        {"token": LABEL_TOKENS[index], "value": str(value)}
        for index, value in enumerate(values)
    ]
    questions = {
        "predicate": "Which declared predicate does the evidence describe?",
        "argument_1": "Which entity is canonical argument 1: the unary entity or directed source?",
        "argument_2": "Which entity is canonical argument 2: the directed target, or N/A for a unary fact?",
        "truth_status": "Is the canonical fact known true, known false, or unresolved (unknown)?",
    }
    option_text = "\n".join(f"{row['token']}: {row['value']}" for row in options)
    content, _ = representation_prompt_layout(row, config)
    return (
        f"{content}\nCanonical relation order is predicate(source, target), including for inverse wording.\n"
        f"Question: {questions[field]}\nOptions:\n{option_text}\nAnswer:",
        options,
    )
