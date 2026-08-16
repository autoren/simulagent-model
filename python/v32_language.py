"""Formal semantics, fresh surfaces, and prompts for V32."""

from __future__ import annotations

import re
from typing import Any, Sequence

from v30_language import fact_phrases, ontology_description, sha256_text


SURFACE_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "assert": {
        "fit_a": ("The state ledger explicitly enters the literal: {literal}.", "medium"),
        "fit_b": ("Use this literal as the observed description—{literal}.", "medium"),
        "fit_c": ("The literal warranted by inspection reads: {literal}.", "medium"),
        "fit_d": ("For the present configuration, record without qualification that {literal}.", "long"),
        "cal_a": ("The inspection log directly endorses the literal saying {literal}.", "long"),
        "para_a": ("The current-state entry is the following literal: {literal}.", "medium"),
        "para_b": ("Take the observation at face value when it reports that {literal}.", "long"),
        "para_c": ("The literal description supported here is that {literal}.", "medium"),
        "comp_a": ("The baseline entry directly states the negative literal: {literal}.", "medium"),
        "comp_b": ("Inspection straightforwardly records this literal—{literal}.", "medium"),
        "comp_c": ("No outer qualification applies to the report that {literal}.", "long")
    },
    "deny": {
        "fit_a": ("The evidence invalidates the literal claiming that {literal}.", "medium"),
        "fit_b": ("Reject as an incorrect account the statement that {literal}.", "medium"),
        "fit_c": ("Inspection licenses a denial of the literal: {literal}.", "medium"),
        "fit_d": ("The current record is incompatible with accepting the proposition that {literal}.", "long"),
        "cal_a": ("Treat the literal that {literal} as contradicted by the observation.", "long"),
        "para_a": ("The state rules out, rather than supports, the assertion that {literal}.", "long"),
        "para_b": ("What must be denied is the literal saying that {literal}.", "medium"),
        "para_c": ("The report that {literal} is the one the evidence rejects.", "medium"),
        "comp_a": ("It is the negative literal itself that observation tells us to reject: {literal}.", "long"),
        "comp_b": ("Do not accept the stated negative condition that {literal}.", "medium"),
        "comp_c": ("Evidence contradicts the negative-form proposition according to which {literal}.", "long")
    },
    "double_deny": {
        "fit_a": ("A denial of the literal that {literal} is itself disallowed.", "medium"),
        "fit_b": ("It would be incorrect to reject the proposition saying {literal}.", "medium"),
        "fit_c": ("The record gives no license to deny the literal: {literal}.", "medium"),
        "fit_d": ("Inspection contradicts anyone who would repudiate the statement that {literal}.", "long"),
        "cal_a": ("Calling the literal that {literal} unacceptable would itself conflict with the evidence.", "long"),
        "para_a": ("The proposition that {literal} is not one that may accurately be rejected.", "long"),
        "para_b": ("It is the denial—not the literal—of {literal} that is mistaken.", "medium"),
        "para_c": ("No correct reading permits us to negate the report that {literal}.", "long"),
        "comp_a": ("It would be wrong to deny the negative-form literal that {literal}.", "medium"),
        "comp_b": ("The rejection of this negative literal is what fails: {literal}.", "medium"),
        "comp_c": ("Observation blocks a denial of the negative proposition according to which {literal}.", "long")
    },
    "contrast_select": {
        "fit_a": ("Between the alternatives, select {literal} and discard {opposite}.", "medium"),
        "fit_b": ("The supported side of the contrast is {literal}; the rejected side is {opposite}.", "long"),
        "fit_c": ("Evidence chooses the literal {literal} rather than its alternative, {opposite}.", "long"),
        "fit_d": ("Of the opposed descriptions—{literal} versus {opposite}—the former matches inspection.", "long"),
        "cal_a": ("The first literal, {literal}, is selected over the contrary literal, {opposite}.", "long"),
        "para_a": ("Prefer the account that {literal}, not the opposing account that {opposite}.", "long"),
        "para_b": ("The contrast resolves in favor of {literal} and against {opposite}.", "medium"),
        "para_c": ("Observation identifies {literal} as accurate instead of {opposite}.", "medium"),
        "comp_a": ("The negative-form side wins the contrast: {literal}, rather than {opposite}.", "long"),
        "comp_b": ("Choose the negative literal {literal} over the positive alternative {opposite}.", "long"),
        "comp_c": ("Evidence favors the former negative description—{literal}—not {opposite}.", "long")
    },
    "unresolved": {
        "fit_a": ("The record cannot decide whether the literal {literal} obtains.", "medium"),
        "fit_b": ("Both acceptance and rejection remain open for the proposition that {literal}.", "long"),
        "fit_c": ("Inspection leaves the literal that {literal} without a resolved status.", "long"),
        "fit_d": ("There is insufficient evidence either to endorse or to deny the statement that {literal}.", "long"),
        "cal_a": ("No determination is available for the literal proposition that {literal}.", "medium"),
        "para_a": ("The truth of the literal that {literal} remains unsettled by observation.", "long"),
        "para_b": ("Evidence leaves us unable to accept or reject the report that {literal}.", "long"),
        "para_c": ("The literal {literal} is presently unresolved in either direction.", "medium"),
        "comp_a": ("Whether the negative-form literal {literal} holds is not determined.", "medium"),
        "comp_b": ("The negative proposition that {literal} remains open, not established or refuted.", "long"),
        "comp_c": ("Observation assigns no resolved status to the negative literal {literal}.", "long")
    }
}

DISTRACTOR = "A separate logistics memorandum discusses replacement-part delivery and contains no state claim."


def compile_truth(sign: str, operation: str, config: dict[str, Any]) -> str:
    try:
        return config["factorization"]["truthCompiler"][operation][sign]
    except KeyError as error:
        raise ValueError(f"Unsupported V32 compiler cell: {operation}/{sign}") from error


def construction_hash(operation: str, surface_name: str) -> str:
    template = SURFACE_TEMPLATES[operation][surface_name][0]
    normalized = re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return sha256_text(f"{operation}|{normalized}")


def render_evidence(
    predicate: str, arguments: Sequence[str], lexical_sign: str, operation: str,
    surface_name: str, orientation: str, distractor: bool, config: dict[str, Any],
) -> tuple[str, str]:
    positive, negative = fact_phrases(predicate, arguments, orientation, config)
    literal = positive if lexical_sign == "positive" else negative
    opposite = negative if lexical_sign == "positive" else positive
    template, length = SURFACE_TEMPLATES[operation][surface_name]
    text = template.format(literal=literal, opposite=opposite)
    if distractor:
        text = f"{DISTRACTOR} {text}"
    return text, length


def representation_prompt_layout(
    row: dict[str, Any], config: dict[str, Any],
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    entities = row["agent_input"]["entities"]
    parts = ["Typed entities:\n"]
    for index, entity in enumerate(entities, start=1):
        parts.append(f"Entity {index}: {entity['id']} ({entity['entity_type']})\n")
    parts.append("Declared predicate ontology:\n")
    parts.append(ontology_description(config))
    parts.append("\nEvidence statement: ")
    parts.append(row["agent_input"]["evidence_text"])
    parts.append("\nCompositional canonical signed-fact representation:")
    content = "".join(parts)
    spans = {}
    for entity in entities:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(entity['id'])}(?![A-Za-z0-9_])")
        spans[entity["id"]] = [(match.start(), match.end()) for match in pattern.finditer(content)]
        if not spans[entity["id"]]:
            raise RuntimeError(f"V32 entity absent from prompt: {entity['id']}")
    return content, spans
