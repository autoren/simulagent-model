# Prospective language pilot V1 — Phase 2 development result

## Verdict

The frozen Phase 2 condition completed all 16 prospective requests but **did not qualify** for automatic participant
exposure. This is a negative interface-realization result, not an assistant-understanding result.

| Measure | Result | Frozen gate |
|---|---:|---:|
| Completed requests | 16/16 | 16/16 |
| Structurally valid semantic proposals | 13/16 (81.25%) | at least 15/16 (93.75%) |
| Final continuations reaching token cap | 1/16 (6.25%) | 0/16 |
| Deterministic controller coverage | 16/16 | 16/16 |
| Maximum questions per clarification | 2 | at most 2 |
| Routes after safe fallback | 1 plan, 11 clarify, 4 defer | descriptive |
| Safe fallbacks | 3 | descriptive |
| Retries / API calls / service calls / actions | 0 / 0 / 0 / 0 | all zero |

The three invalid records failed for different reasons: one incompatible clarification payload, one non-JSON final
continuation that reached the 320-token cap, and one invalid missing-evidence code set. Every failure was converted to
the preregistered deterministic `DEFER`; none triggered a retry.

## Interpretation

The main finding is architectural:

> Bounded low-effort reasoning prevented an unbounded loop, but a single large seven-field JSON contract remained too
> brittle for reliable one-pass realization on natural prospective requests.

The condition was not unsafe. The deterministic controller maintained total coverage, limited clarification count,
and performed no action. The failure was that the model could not satisfy the richer semantic certificate often
enough to meet the frozen usability gate.

The run also replicated the earlier Qwen3.8 reasoning-budget concern in a narrower form. Every request consumed the
48-token reasoning allowance; one response then consumed the entire reserved 320-token final allowance without
producing valid JSON. A reasoning cap is therefore necessary but not sufficient. Future model interfaces should use
smaller independently validated fields or direct finite choices rather than asking one continuation to jointly
realize hypotheses, evidence typing, routing, questions, plans, and defer text.

## Exploratory interaction disposition

The failed gate must not be relabeled as a pass, and the same requests must not be retried or reprompted. However, 11
records already contain structurally valid controller-approved clarification outputs. A separate post-result protocol
review may expose only those fixed questions as an explicitly exploratory partial interaction if all of these remain
true:

- the original negative result and failed gates remain unchanged;
- no invalid record is repaired, regenerated, or converted into a clarification;
- only non-fallback `CLARIFY` records are shown;
- the exact question strings are hash-locked before presentation;
- participant answers are collected for all 11 before any terminal continuation;
- the salvage is not used as confirmation or as evidence of population reliability.

This disposition salvages prospective clarification-utility evidence while preserving the one-shot failure.
