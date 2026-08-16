"""Deterministic corpus and prompt utilities for the V30 signed-fact study."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Iterable, Sequence


ENTITY_ALIASES = (
    "noru", "vela", "tavi", "soren", "mira", "keto", "luma", "pavo",
    "runi", "dexa", "zori", "bela", "cavo", "fira", "ganu", "hena",
    "jora", "lito", "mavo", "pira", "ravo", "sela", "tora", "vani",
)
LABEL_TOKENS = ("A", "B", "C", "D", "E", "F")
TRUTH_VALUES = {"true": [True], "false": [False], "unknown": [False, True]}


SURFACE_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "affirmative_gold": {
        "fit_a": ("The inspection confirms that {gold}.", "short"),
        "fit_b": ("According to the current reading, {gold}.", "short"),
        "fit_c": ("The condition recorded for the system is that {gold}.", "medium"),
        "cal_a": ("The observed configuration can be summarized as follows: {gold}.", "medium"),
        "eval_a": ("The inspection's finding can be stated plainly: {gold}.", "medium"),
        "eval_b": ("As for the system's present configuration, {gold}.", "medium"),
        "eval_c": ("The condition borne out by the latest check is that {gold}.", "long"),
    },
    "negated_opposite": {
        "fit_a": ("It is not the case that {opposite}.", "short"),
        "fit_b": ("The claim that {opposite} is false.", "short"),
        "fit_c": ("One must reject the statement that {opposite}.", "medium"),
        "cal_a": ("It would be incorrect to describe the current state by saying that {opposite}.", "long"),
        "eval_a": ("Far from being accurate, the report that {opposite} must be rejected.", "long"),
        "eval_b": ("There is no truth to the assertion that {opposite}.", "medium"),
        "eval_c": ("Whatever else is uncertain, it would be wrong to maintain that {opposite}.", "long"),
    },
    "contrastive_both": {
        "fit_a": ("{gold_cap}, rather than {opposite}.", "short"),
        "fit_b": ("The correct description is that {gold}; not that {opposite}.", "medium"),
        "fit_c": ("Of the two descriptions, {gold} is accurate and {opposite} is not.", "medium"),
        "cal_a": ("The reading supports {gold}, while ruling out the alternative that {opposite}.", "long"),
        "eval_a": ("Not {opposite}; what the inspection establishes instead is that {gold}.", "medium"),
        "eval_b": ("Between the competing reports, the one saying {gold} is correct, not the one saying {opposite}.", "long"),
        "eval_c": ("Although one report says {opposite}, the valid account is that {gold}.", "long"),
    },
    "double_negation": {
        "fit_a": ("It is not untrue that {gold}.", "short"),
        "fit_b": ("One cannot truthfully deny that {gold}.", "medium"),
        "fit_c": ("The proposition that {gold} is not false.", "medium"),
        "cal_a": ("Rejecting the statement that {gold} would itself be an error.", "medium"),
        "eval_a": ("It would be false to call the claim that {gold} untrue.", "long"),
        "eval_b": ("No accurate account can deny that {gold}.", "medium"),
        "eval_c": ("The denial of the proposition that {gold} does not withstand inspection.", "long"),
    },
    "explicit_unknown": {
        "fit_a": ("It is undetermined whether {positive}.", "short"),
        "fit_b": ("Current evidence leaves unresolved whether {positive}.", "short"),
        "fit_c": ("The available record does not settle whether {positive}.", "medium"),
        "cal_a": ("No conclusion can presently be drawn about whether {positive}.", "medium"),
        "eval_a": ("Whether {positive} cannot be established from the inspection.", "medium"),
        "eval_b": ("The evidence settles neither the claim that {positive} nor its denial.", "long"),
        "eval_c": ("There is insufficient information to decide whether or not {positive}.", "long"),
    },
}

DISTRACTOR = "A separate maintenance note concerns scheduling only."


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def opaque_id(kind: str, token: str, length: int = 20) -> str:
    return f"{kind}_{sha256_text(token)[:length]}"


def deterministic_shuffle(values: Iterable[Any], token: str) -> list[Any]:
    result = list(values)
    random.Random(int(sha256_text(token)[:16], 16)).shuffle(result)
    return result


def predicate_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ontology = config["ontology"]
    return {
        row["id"]: {**row, "kind": "unary"}
        for row in ontology["unaryPredicates"]
    } | {
        row["id"]: {**row, "kind": "relation"}
        for row in ontology["relations"]
    }


def atom_key(predicate: str, arguments: Sequence[str], config: dict[str, Any]) -> str:
    kind = predicate_specs(config)[predicate]["kind"]
    prefix = "u" if kind == "unary" else "r"
    return ":".join((prefix, predicate, *arguments))


def fact_phrases(
    predicate: str, arguments: Sequence[str], orientation: str, config: dict[str, Any],
) -> tuple[str, str]:
    spec = predicate_specs(config)[predicate]
    if spec["kind"] == "unary":
        values = {"entity": arguments[0]}
        return spec["trueForm"].format(**values), spec["falseForm"].format(**values)
    values = {"source": arguments[0], "target": arguments[1]}
    prefix = "direct" if orientation == "direct" else "inverse"
    return (
        spec[f"{prefix}TrueForm"].format(**values),
        spec[f"{prefix}FalseForm"].format(**values),
    )


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
        text = f"{DISTRACTOR} {text}"
    return text, length


def positive_candidate_statement(
    predicate: str, arguments: Sequence[str], config: dict[str, Any],
) -> str:
    positive, _ = fact_phrases(predicate, arguments, "direct", config)
    return positive + "."


def ontology_description(config: dict[str, Any]) -> str:
    lines = []
    for row in config["ontology"]["unaryPredicates"]:
        lines.append(
            f"- {row['id']}(entity): Boolean property for an entity of type {row['entityType']}."
        )
    for row in config["ontology"]["relations"]:
        lines.append(
            f"- {row['id']}(source, target): directed Boolean relation from "
            f"{row['sourceType']} to {row['targetType']}."
        )
    return "\n".join(lines)


def entity_description(row: dict[str, Any]) -> str:
    return ", ".join(
        f"{entity['id']} ({entity['entity_type']})" for entity in row["agent_input"]["entities"]
    )


def field_options(
    row: dict[str, Any], field: str, config: dict[str, Any],
) -> list[dict[str, str]]:
    if field == "predicate":
        values = config["methods"]["primary"]["predicateOrder"]
    elif field == "argument_1":
        values = [entity["id"] for entity in row["agent_input"]["entities"]]
    elif field == "argument_2":
        values = [entity["id"] for entity in row["agent_input"]["entities"]]
        values.append(config["methods"]["primary"]["argument2NotApplicableValue"])
    elif field == "truth_status":
        values = config["methods"]["primary"]["truthOrder"]
    else:
        raise ValueError(f"Unknown V30 field {field}")
    if len(values) > len(LABEL_TOKENS):
        raise ValueError("V30 field exceeds registered label inventory")
    return [
        {"token": LABEL_TOKENS[index], "value": str(value)}
        for index, value in enumerate(values)
    ]


def primary_field_prompt(
    row: dict[str, Any], field: str, config: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    options = field_options(row, field, config)
    question = {
        "predicate": "Which declared predicate does the evidence describe?",
        "argument_1": (
            "Which entity is canonical argument 1 (the entity for a unary predicate, or the "
            "directed source for a relation)?"
        ),
        "argument_2": (
            "Which entity is canonical argument 2 (the directed target), or N/A if the fact is unary?"
        ),
        "truth_status": (
            "What is the fact's epistemic truth status: true, false, or unknown? Unknown means the "
            "evidence leaves both truth values possible; it is not the same as false."
        ),
    }[field]
    rendered_options = "\n".join(
        f"{option['token']}: {option['value']}" for option in options
    )
    content = (
        f"Typed entities: {entity_description(row)}.\n"
        f"Declared predicate ontology:\n{ontology_description(config)}\n"
        f"Evidence statement: {row['agent_input']['evidence_text']}\n"
        "Canonical relation order is predicate(source, target), even when the sentence uses "
        "inverse or passive wording.\n"
        f"Question: {question}\nOptions:\n{rendered_options}\nAnswer:"
    )
    return content, options


def v26_baseline_prompt(row: dict[str, Any]) -> str:
    return (
        f"Typed entities: {entity_description(row)}.\n"
        f"Evidence statement: {row['agent_input']['evidence_text']}\n"
        f"Candidate fact: {row['target']['candidate_statement']}\n"
        "Classification:"
    )


def candidate_nli_prompt(row: dict[str, Any]) -> str:
    return (
        f"Typed entities: {entity_description(row)}.\n"
        f"Evidence statement: {row['agent_input']['evidence_text']}\n"
        f"Candidate atom: {row['target']['candidate_statement']}\n"
        "Interpret the complete evidence, including lexical opposites, negation scope, double "
        "negation, inverse relation wording, and uncertainty.\n"
        "A: the candidate atom is known true.\n"
        "B: the candidate atom is known false.\n"
        "C: the candidate atom is unresolved; both true and false remain possible.\n"
        "Answer:"
    )


def select_option(logits: Sequence[float], options: Sequence[dict[str, str]]) -> dict[str, str]:
    if len(logits) != len(options) or not options:
        raise ValueError("V30 logits and options must have equal nonzero lengths")
    index = max(range(len(logits)), key=lambda value: (float(logits[value]), -value))
    return options[index]


def log_softmax(values: Sequence[float]) -> list[float]:
    maximum = max(float(value) for value in values)
    denominator = maximum + math.log(sum(math.exp(float(value) - maximum) for value in values))
    return [float(value) - denominator for value in values]


def canonical_prediction(field_values: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    predicate = field_values["predicate"]
    kind = predicate_specs(config)[predicate]["kind"]
    arguments = [field_values["argument_1"]]
    if kind == "relation":
        arguments.append(field_values["argument_2"])
    return {
        "predicate": predicate,
        "arguments": arguments,
        "truth_status": field_values["truth_status"],
        "atom": atom_key(predicate, arguments, config),
    }


def valid_public_candidate_facts(
    entities: Sequence[dict[str, str]], config: dict[str, Any],
) -> list[dict[str, Any]]:
    by_type: dict[str, list[str]] = {}
    for entity in entities:
        by_type.setdefault(entity["entity_type"], []).append(entity["id"])
    result = []
    for spec in config["ontology"]["unaryPredicates"]:
        for entity in by_type.get(spec["entityType"], []):
            arguments = [entity]
            result.append({
                "predicate": spec["id"], "arguments": arguments,
                "atom": atom_key(spec["id"], arguments, config),
                "statement": positive_candidate_statement(spec["id"], arguments, config),
            })
    for spec in config["ontology"]["relations"]:
        for source in by_type.get(spec["sourceType"], []):
            for target in by_type.get(spec["targetType"], []):
                if source == target:
                    continue
                arguments = [source, target]
                result.append({
                    "predicate": spec["id"], "arguments": arguments,
                    "atom": atom_key(spec["id"], arguments, config),
                    "statement": positive_candidate_statement(spec["id"], arguments, config),
                })
    return result


def parse_public_candidate_statements(
    candidates: Sequence[dict[str, str]], entities: Sequence[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    possible = valid_public_candidate_facts(entities, config)
    by_statement = {row["statement"]: row for row in possible}
    if len(by_statement) != len(possible):
        raise ValueError("V30 public candidate grammar is not injective")
    result = {}
    for candidate in candidates:
        if candidate["statement"] not in by_statement:
            raise ValueError(f"Unparseable public candidate statement: {candidate['statement']}")
        result[candidate["id"]] = by_statement[candidate["statement"]]
    return result
