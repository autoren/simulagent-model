# V191 fresh language-to-menu population plan

## Purpose

V190 confirmed a narrow fixed typed hierarchy, but it did not test whether language can select or reduce that menu.
V191 creates the first fresh development population for that question. It is deliberately text-free: source language
is not extracted until the identities, exclusions, target coverage, and authority boundary are immutable.

## Freshness rule

Use only frozen SGD `dev` intent-introduction candidate metadata for the 14 V183 capability contracts. Exclude every
dialogue that supplied any V183 development or protected record, not merely the exact selected turn. This prevents a
new conversation prefix from overlapping language previously used by V185 or language still sealed in V184's
protected role.

Within each contract, order eligible candidate identifiers by a fixed salted hash and take six. Selected dialogues
must be globally unique. Add 14 synthetic missing-observation controls. The resulting development role must contain:

- 84 observed records, exactly six for each of 14 contracts;
- 36 known, 36 provisional, and 12 unsupported targets;
- 14 missing controls; and
- 98 total fixtures.

Selection may use only source partition, candidate identifier, service/intent definition, contract mapping, prior-use
identifiers, and the fixed salt. It may not use utterances, dialogue text, slots, semantic frames, predictions, scores,
or outcomes.

## Artifacts

V191 emits separate public identities and hidden targets. Public identities contain only opaque record ID, role, and
observation availability. Hidden targets retain the source candidate and full contract label for later automatic
scoring. Neither artifact contains language, slot values, frames, spans, or model output.

## Decision

A complete pass authorizes only a separately preregistered exact extraction of these 84 fresh conversation prefixes
into an unprotected development artifact. It does not authorize immediate extraction, interface scoring, a local/API
model, protected access, ontology registration or pruning, trusted mutation, action, or execution.
