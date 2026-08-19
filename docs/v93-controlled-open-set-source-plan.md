# V93 Controlled Open-Set Source Plan

## Purpose

V93 begins the controlled open-world program with source feasibility only. It pins one untouched
Schema-Guided Dialogue development shard and constructs a text-free index for five prospectively defined
classes. No utterance, prompt, candidate model output, probability, or downstream decision is inspected
or scored during this stage.

## Frozen classes

Each eligible service has at least three source-native typed intents. One structurally supported intent
is hidden by a SHA-256 rule before any language-derived statistic is used; every other intent remains in
the declared schema.

- `known_familiar`: a declared intent whose accumulated user history shares at least one normalized
  content token with its source intent name or description;
- `known_unfamiliar`: a declared intent with zero such overlap, creating an externally authored
  paraphrase-like open-set challenge without rewriting source language;
- `novel_valid`: the source annotation names the prospectively hidden, source-valid intent;
- `unsupported`: the source request belongs to a different eligible service and its source intent is
  absent from the target service's complete source schema;
- `insufficient_evidence`: the source annotation is `NONE`, retained separately from unsupported input.

Tokenization, stop words, hidden-intent selection, cross-service target selection, exclusions, minimum
counts, and hashes are frozen in code and configuration before the shard is downloaded. The inventory
may record identifiers, counts, class labels, and overlap counts, but it must not emit language, tokens,
slot values, prompts, or dialogue histories.

## Authorization boundary

Passing this stage authorizes only a later population, calibration/evaluation split, controls, prompt,
output contract, scoring, and gate preregistration. It does not authorize language extraction, manual
inspection, a model load, API use, training, a generated capability, posterior integration, planning,
or execution. Any failed source gate stops V93 without changing exclusions, thresholds, or class rules.
