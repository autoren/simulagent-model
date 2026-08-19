# V201 local menu-presentation robustness plan

## Question

Does the already confirmed local finite-menu ranker preserve its task value and semantic rankings when only menu
order and opaque identifiers change?

## Locked condition

V201 changes no model behavior. It inherits V195/V198 exactly: pinned `Qwen3.8-27B-4bit`, low reasoning effort,
48-token reasoning phase, mechanical `</think>` close, separately reserved 64-token final phase, temperature zero,
one sample, no retries, final-user-only prompt, exact three-ID parser, and fail-closed `INSUFFICIENT` fallback.

It evaluates 84 V192 development records under the two V199 variants, for 168 record-variant prompts and exactly two
generation phases per prompt. The 14 missing records receive no generation in either variant.

## Scoring

Every transformed option ID is mapped through its frozen per-record bijection before comparison with the canonical
V195 normalized proposal. Per variant, measure task recall and trusted-controller cost, top-1 contract agreement,
mean top-3 contract-set Jaccard, target-inclusion disagreement, structural validity, truncation, and incremental cost
over V200 `CHAR_LAST` on the identical variant.

The complete gates are inherited verbatim from V199. Each variant may lose at most 0.05 primary or macro recall and
add at most 0.02 primary cost; it needs at least 0.80 top-1 agreement, 0.80 mean top-3 Jaccard, no more than 5% target
hit disagreement, and at least 0.01 cost improvement over transformed `CHAR_LAST`. Structural validity must be at
least 0.98; final truncation and false terminal decisions zero; target retention and exact trusted completion one.

## Stop rules

Raw reasoning and final text are hashed but not persisted or inspected. Any malformed response fails closed. The run
is frozen positive or negative without prompt repair, retry, model selection, larger budget, or API fallback.
Protected language, training, ontology mutation, services, side effects, action, and execution remain closed. A pass
allows only a separately preregistered paired protected robustness design.

