# V190: Protected multiway menu-compression confirmation

## Bottom line

V190 is a clean positive confirmation of finite hierarchical menu compression. The frozen `domain → intent concept` sequence reproduced its development cost exactly on the separately sealed protected identity distribution without reading protected utterance language or optimizing the policy.

## Confirmed result

- Protected bindings: 132.
- Observed targets: 120 across all 14 contracts.
- Missing controls: 12, all retained as insufficient at zero cost.
- Fixed-policy mean cost: `0.38`.
- Always-generic cost: `0.40`.
- Improvement: `0.02`.
- Mean categorical turns: `1.8`.
- Typed-only completion: `1.0`.
- Final exactness: `1.0`.
- Target retention: `1.0`.
- Absolute development/protected cost difference: `0.0`.
- Policy optimization count: 0.

All confirmation gates passed.

## Controls

| Policy | Mean cost | Mean turns | Typed completion |
|---|---:|---:|---:|
| Fixed domain → intent | 0.38 | 1.8 | 1.0 |
| Always generic | 0.40 | 0.0 | 0.0 |
| Flat 14-intent menu | 0.40 | 1.0 | 1.0 |
| Domain only, then generic | 0.62 | 1.0 | 0.2 |

The flat 14-intent menu costs four bit slots, exactly tying generic clarification. Domain alone is worse because it costs three bit slots and leaves most domains unresolved. The hierarchy wins because two singleton domains stop after the three-bit domain answer, while the remaining domains pay only the additional within-domain intent bits.

The fixed policy is also the best open-loop sequence observed in V189; no adaptive claim is supported. The gain comes from hierarchical coding and early stopping, not history-dependent selection among different next questions.

## What this establishes

Within the frozen 14-contract oracle-answer universe, a semantic hierarchy can reduce expected worst-case bit-slot cost below a flat exact answer. That mechanism transferred from development to a different protected contract-frequency distribution with identical aggregate cost.

## What this does not establish

V190 does not show that:

- users understand or can reliably answer the menus;
- raw utterances can be mapped to the right domain or intent;
- an LLM can answer or rank these questions reliably;
- the interface handles capabilities outside the 14-contract universe;
- multiway clarification is cognitively priced by bit slots in practice;
- adaptive planning adds value; or
- any capability may be registered, invoked, or executed.

Source annotations remained dataset-provided simulation oracles. Protected utterance language was never read. No model/API, training, service call, trusted-state mutation, side effect, action, or execution occurred.

## Decision

Freeze V190 as confirmed finite oracle-menu compression only. Do not reinterpret V189 as formally passed.

The next research branch should validate the missing observation/interface premise rather than add planner complexity. Two legitimate options remain:

1. an external human/UI burden and answer-reliability study for the fixed hierarchy; or
2. a fresh, non-authoritative shadow proposal study that asks whether language/model evidence can select the first menu or reduce it while preserving the full authoritative hypothesis set.

Because no human assistance is available, the practical next branch is the second option on a new development population, with model output limited to question/menu ranking and malformed or truncated output mapped to `INSUFFICIENT`. It must not reuse the closed V185 residual or open protected utterances.
