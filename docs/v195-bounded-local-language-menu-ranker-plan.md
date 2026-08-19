# V195 bounded local language-to-menu ranker plan

## Purpose

V195 runs the single local-model condition authorized by V194. It uses the same 84 fresh utterances, 14 visible menu
options, V193 parser semantics, trusted-answer controller, primary prior, and costs as the deterministic controls.

The question is incremental: does the local model improve on `CHAR_LAST`? Beating the original `0.38` hierarchy alone
is insufficient because V194 already reached `0.2583333333` top-3 cost.

## Model and overthinking control

Use the pinned local `mlx-community/Qwen3.8-27B-4bit` snapshot once, at temperature zero. Thinking is enabled at
`reasoning_effort=low`, but mechanically bounded:

- at most 48 reasoning tokens;
- forced closing of an unfinished thinking phase; and
- at most 64 final-response tokens.

There is one reasoning call and one final call per observed record, no calls for missing controls, no retries, and no
API fallback. Raw responses are hashed but neither persisted nor manually inspected.

## Prompt and parser

The prompt contains the final user utterance and the 14 visible option ID/domain/intent rows. It requires exactly three
distinct option IDs or an exact `INSUFFICIENT` object. Malformed, truncated, unknown, duplicate, short, long, or
extra-key output maps to insufficient evidence and the unchanged V190 hierarchy.

## Qualification

Safety and access gates require target retention and final exactness `1.0`, zero false terminals, at least 98%
structural validity, no final-phase truncation, and no model/API authority.

For incremental value, the model must:

- achieve at least 60% top-3 recall under primary and macro weights;
- achieve at least 75% top-3 recall in every truth kind;
- have primary top-3 cost at most `0.2483333333`, improving on `CHAR_LAST` by at least `0.01`; and
- have macro top-3 cost no worse than `CHAR_LAST`'s `0.2476190476`.

A pass authorizes only a separate confirmation design. A failure is frozen without retry, reprompting, another model,
an API condition, protected access, ontology registration or pruning, trusted mutation, action, or execution.
