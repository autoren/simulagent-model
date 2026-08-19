# V221/V221r1 deterministic Mondo residual results

## Bottom line

The prospectively frozen V221 scientific design passed after one narrow implementation repair. At the accepted
candidate budget of eight, deterministic exact-family expansion retained the complete oracle version space on every
evaluation record. The model-eligible residual contained **zero evaluation groups**.

The frozen branch decision is:

> `DETERMINISTIC_SUFFICIENT_CLOSE_MODEL_ESCALATION`

An LLM should not be introduced for this retrospective ontology-release task. It has no missing oracle class to
recover and could only add cost or unsupported candidates.

This conclusion is deliberately narrow. It establishes deterministic sufficiency for reconstruction from published
Mondo releases. It does not establish how a new speaker intends an unfamiliar utterance, discover correct new
real-world meaning, or remove the need for independently sourced semantic evidence in B2c.

## Prospective design and repair

V221 froze, before development JSONL body access:

- a SHA-256-derived, group-disjoint split of 1,621 V220 development groups into 1,296 calibration groups and 325
  evaluation groups;
- the four-method deterministic portfolio;
- candidate budgets `k in {1, 4, 8, 16}` and accepted budget `k = 8`;
- exact version-space, decision, regret, and fail-closed metrics; and
- a model-eligibility gate requiring at least 12 evaluation groups with decision-relevant oracle classes still
  missing after all allowed deterministic controls.

The design lock is
`83fb85a7f202fbd75708424e3552460a47d68d8bf5ac6761adee95181bbf4c4e`.

The initial runner stopped during catalog construction because the V221 runtime config omitted the already-frozen
V220 `parserDesign` object required by the inherited asserted-state function. Development public and truth JSONLs had
each been loaded once, but no catalog, candidate observation, residual, summary, or scientific result was produced;
candidate-method evaluation count was zero.

V221r1 repaired only that interface defect by injecting the exact frozen V220 `parserDesign` into an in-memory copy of
the V221 config. The role split, catalog semantics, methods, budgets, controller, thresholds, residual rule, and input
hashes did not change. Its repair lock is
`7d43269b02330beeddbb1350c467097b710f725f611c8b15258e0de2b7696571`.

## Evaluation result

The repaired run produced 51,872 record-method-budget observations and all 64 preregistered aggregate metric cells.
There was no evaluation tuning.

At the primary condition, `M3_FINAL_FAIL_CLOSED` with `k = 8`:

| Metric | Evaluation result |
|---|---:|
| Oracle-class recall | 1.000000 |
| Full version-space retention | 1.000000 |
| Exact decision rate | 0.998462 |
| Mean decision regret | 0.000154 |
| Unsafe singleton-collapse rate | 0.000000 |
| Residual evaluation groups | 0 |

Candidate-class validity, candidate-budget compliance, atomic family expansion, and contradiction fail-closed
accuracy were all 1.0.

The evaluation curves show why model escalation is unwarranted:

| Method | Budget | Oracle recall | Full retention | Exact decision | Mean regret |
|---|---:|---:|---:|---:|---:|
| Normalized exact | 1 | 0.744487 | 0.496923 | 0.998462 | 0.000154 |
| Normalized exact | 8 | 0.767436 | 0.540000 | 0.998462 | 0.000154 |
| Exact family | 4 | 0.998462 | 0.998462 | 0.998462 | 0.000154 |
| Exact family | 8 | 1.000000 | 1.000000 | 0.998462 | 0.000154 |
| Hybrid retrieval | 8 | 1.000000 | 1.000000 | 0.998462 | 0.000154 |
| Final fail-closed | 8 | 1.000000 | 1.000000 | 0.998462 | 0.000154 |

Exact family expansion, rather than fuzzy retrieval, closed the residual. Once a public surface identified a stable
concept family, the deterministic system enumerated the relevant release-state classes without collapsing historical
ambiguity.

## The single conservative fallback

One of 650 primary evaluation records did not match the oracle decision exactly. It was a
`CURRENT_RELEASE_DECLARED / MAPPING_CHANGED` case. A normalized surface collision returned the target class plus one
additional class with an incompatible lifecycle interpretation. The fail-closed controller selected
`PRESERVE_VERSION_SPACE_OR_CLARIFY` instead of `RESOLVE_CURRENT_STATE`, incurring regret 0.1 for that record.

This was a conservative extra clarification, not an unsafe action. Oracle recall and full version-space retention
remained 1.0, and unsafe singleton collapse remained false. Averaged over both evidence modes, it accounts for the
0.998462 exact-decision rate and 0.000154 mean regret.

## Access and claim boundary

The repaired scientific run performed one catalog build and one development evaluation. It made zero network
requests, model loads, model generations, API calls, training runs, ontology registrations, trusted-state mutations,
service actions, external side effects, or executions.

V220 protected public and truth JSONL bodies were never loaded. They remained hash-only. Earlier V213, V216, and V218
data remained untouched under their existing boundaries.

The scientific conclusion is:

> Published, versioned ontology artifacts plus deterministic exact-family expansion are sufficient for this frozen
> retrospective Mondo reconstruction task at `k = 8`.

The conclusion is not:

> Deterministic ontology lookup solves unrestricted open-world language understanding or identifies a new speaker's
> intended concept.

Those claims require a different source of semantic evidence and a separately designed study.

