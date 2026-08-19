# Model-free uncertainty-aware reference architecture

## Status and claim boundary

This document consolidates mechanisms already frozen through V224. The companion harness is a software integration
check, not a new experiment or benchmark. Passing it shows that the selected modules can exchange exact symbolic
state while preserving their safety invariants. It does not demonstrate external natural-language understanding,
empirical semantic likelihoods, calibrated LLM uncertainty, human behavior, deployment, authority, action, or
execution.

The architecture is deliberately model-free. V198 permits an optional local model to reduce a finite clarification
menu, but that proposal provider is disabled in the reference harness and can never update belief weights directly.

## End-to-end control flow

```text
typed candidate/version space
        |
        v
exact belief + explicit OUTSIDE/OTHER state
        |
        v
value-of-information / certificate query policy
        |
        +---- insufficient, contradictory, or OTHER ----> preserve hypotheses + defer
        |
        v
triple observation + one-corruption-robust decode
        |
        v
unanimous trusted registered class?
        | no
        +-----------------------------------------------> defer
        | yes
        v
preview-only reversible sandbox
        |
        v
independent verification + terminal settlement
        |
        +---- mismatch -------------------------------> rollback + defer
        |
        v
simulated trusted completion
```

No arrow permits a candidate, model score, or unresolved hypothesis to register an ontology item or mutate a real
state. The sandbox is a local in-memory simulation with no service/tool target.

## Component contracts

| Component | Inputs | Outputs | Frozen evidence | Required invariant |
|---|---|---|---|---|
| Typed version space | finite candidate IDs, exact class-balanced prior | normalized surviving candidates | V165/V167/V214 | no similarity collapse; inconsistent mass becomes zero |
| Outside-semantic POMDP | state x codebook/outside belief, sensing/control actions | exact policy and posterior branches | V205/V209 | common support, explicit outside regime, safe defer, no horizon escape |
| Certificate policy | belief, available queries, costs, horizon | exact query/stop tree | V175/V177 | query only for lower expected routed risk; no false trusted route |
| Robust observation | three observations per selected bit | majority-decoded exact evidence | V179/V180/V182 | any one flip in a triple leaves decoded evidence unchanged |
| Conservative controller | surviving classes or OTHER | trusted alias/composition or defer | V190/V202 | only unanimous registered classes route; OTHER retains full space |
| Reversible sandbox | fixed-ontology proposal and exact preview token | verified simulated state or rollback | V168/V171 | preview nonmutation, atomic parity, independent verification, hash chain |
| Terminal settlement | completed repair or unfinished sensing | settled reward or safe defer | V205 | every repair settles; unfinished sensing cannot escape the horizon |

The exact source paths and outcome locks are frozen in
`configs/model-free-reference-architecture.json`.

## State interfaces

### Hypothesis state

```text
candidate_id
expressibility_class = alias | composition | provisional_primitive | outside
exact_weight
evidence_history
authority = provisional
```

Weights are rational or normalized floating-point values produced by an explicit generative channel. A proposal rank,
confidence, frequency, or model agreement value is not accepted as a weight.

### Controller decision

```text
QUERY(evidence_action)
TRUSTED_ROUTE(alias | composition)
OTHER_DEFER(residual_version_space)
```

There is no `BEST_GUESS_AND_COMMIT` decision. `OTHER_DEFER` is a state-preserving terminal/safe action, not an error
or a hidden assignment to the nearest catalog item.

### Sandbox transaction

```text
fixed trusted ontology proposal
-> nonmutating preview
-> preview token binds base + patch + expected result
-> atomic simulated commit
-> independent verification
-> retain or complete rollback
```

Provisional primitives and outside-semantic hypotheses cannot enter the sandbox.

## Reference harness

The harness uses only existing project-authored deterministic assets:

1. the V165 finite truth-table universe and V167 class-balanced exact belief;
2. the V175 exact certificate policy;
3. the V179 triple-repetition/majority decoder with one deliberately flipped raw inspection;
4. the V173 unanimous trusted route;
5. one existing V168 `valid_retain` sandbox fixture; and
6. the complete V205 terminally proper open-world oracle and audit.

Its three-candidate fixture contains one alias, one registered composition, and one provisional primitive. It is an
interface fixture, not a fresh population and not additional scientific evidence. The trusted branch must identify the
alias despite one corrupted raw inspection, while an uninterpretable `OTHER` observation must retain all candidates,
defer, and never enter the sandbox.

Run it with:

```bash
PYTHONPATH=python .venv/bin/python python/run_model_free_reference_architecture.py
PYTHONPATH=python .venv/bin/python -m unittest python/test_model_free_reference_architecture.py
```

## External semantic adapter remains absent

The architecture has a typed slot for an empirical observation channel, but the project does not currently possess a
valid implementation. V221 solves retrospective catalog reconstruction; V223 qualifies a workflow; V224 finds no
record-level four-way population. None supplies prospective speaker-intent likelihoods.

An external adapter may be added only after the stopping rule's source, independence, identifiability, provenance, and
deterministic-residual gates pass. Until then, the reference architecture ends at a verified mechanism stack and makes
no end-to-end language claim.
