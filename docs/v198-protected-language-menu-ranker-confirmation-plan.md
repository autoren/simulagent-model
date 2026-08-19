# V198 protected language-to-menu ranker confirmation plan

## Purpose

V198 applies the complete V195 model-plus-controller policy once to V197's independently isolated confirmation
language. It also applies the unchanged V194 `CHAR_LAST` ranker to the same records so incremental model value is
tested against a same-population control.

## Frozen policy

The model snapshot, prompt, finite menu, temperature, no-retry rule, low reasoning effort, 48-token reasoning cap,
mechanical thinking closure, reserved 64-token final phase, exact fail-closed parser, trusted top-3 answer, costs, and
qualification gates are inherited unchanged from V195.

The model must beat confirmation `CHAR_LAST` primary top-3 cost by at least `0.01`, while retaining the frozen
absolute primary and macro cost ceilings. Missing controls receive no model generation.

## Reporting and boundary

Report recall and cost under primary and balanced weights, class-conditioned top-3 recall, insufficiency, structural
validity, reasoning and final limit hits, exact retention, trusted completion, and development-to-confirmation deltas.
Raw model text is hashed but not persisted or manually inspected.

A failure is frozen without retry, reprompt, budget increase, another model, or API. A pass confirms only finite,
non-authoritative clarification-menu reduction and authorizes a separate robustness roadmap—not ontology changes,
services, side effects, action, or execution.

