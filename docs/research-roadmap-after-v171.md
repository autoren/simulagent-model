# Research roadmap after V171

## Evidence state

The program now has two independently confirmed mechanisms and one deliberately closed language branch.

1. **Track A — open-set language and models.** V164 closed the present local-model residual protocol negatively.
   Formatting improved, but novelty recall remained zero and regret worsened. V166 then found no residual at all
   in the exact finite DSL. No local or API model is currently justified.
2. **Track C — factored uncertainty and evidence gathering.** V165–V167 developed exact version-space inference
   and adaptive evidence gathering. V169 froze a nonoverlapping population before scoring, and V170 strongly
   confirmed the unchanged planner: all 58 eligible states had positive information value and strict adaptive
   improvement over the optimal fixed query pair; 49 had genuinely history-dependent second actions.
3. **Track B — reversible fixed-ontology control.** V168 developed exact typed preview, commit, verification,
   rollback, and provenance behavior. V171 freshly confirmed the unchanged contract on all 132 frozen stateful
   sequences spanning races, replay, four crash points, partial writes, repeated rollback, restart, tamper, and
   atomic multi-entity updates.

V170 and V171 satisfy the earlier roadmap's precondition for designing a controlled integration. They do not
authorize deployment, provisional registration, a model, a real service, or execution.

## Active Track D — trusted-only shadow integration

The next scientific question is whether the confirmed uncertainty-aware planner and confirmed reversible sandbox
compose safely when their authority boundaries are explicit.

```text
exact candidate version space
        -> policy may inspect or stop
        -> deterministic consensus gate
             -> unanimous alias/composition: trusted typed proposal
             -> mixed or provisional: defer
        -> V171 reversible simulation
        -> independent verification retains final authority
```

The planner's terminal class decision never authorizes a commit. The consensus gate may route a proposal only
when every surviving exact candidate agrees on one already trusted class. `provisional_primitive` is always
deferred and remains outside the sandbox. Schema checks, invariants, preview binding, state verification, rollback,
and provenance remain deterministic.

### Sequence

1. **V172 population:** exhaustively generate a new bounded constraint-state family that was not scored in V170.
   Freeze every state, exact version space, class coverage, eligible membership, and target identities before any
   integration policy is evaluated. Retain ineligible and no-opportunity states rather than filtering by outcome.
2. **V173 policy-routing study:** reuse the V167/V167r1 policy contracts and V171 sandbox contract unchanged.
   Execute each policy against frozen simulated target outcomes, apply the deterministic consensus gate, and
   compare safe trusted completion, deferral, query cost, exact class loss, false-trusted routing, sandbox final
   state, recovery, and provenance. Include no-query, forced-MAP, greedy, optimal open-loop, exact adaptive, random,
   and oracle controls. A negative or mixed result is retained without tuning.
3. **Confirmation or isolation:** if integration passes, freeze a new nonoverlapping integration confirmation
   before broadening the claim. If it fails, identify whether the incompatibility is policy stopping, consensus
   opportunity, transaction routing, or recovery, and keep the mechanisms isolated until a separately locked
   redesign.

## Dormant tracks

- **Language residual track:** may reopen only with a prospectively frozen, truth-free open-language interface
  that leaves a nonempty deterministic residual. Human labels are optional; ambiguity may be represented as a
  set-valued oracle or simulated from a declared grammar. A model remains a candidate proposer, never authority.
- **Capacity/model track:** local Qwen or an API comparator is relevant only after such a residual exists. Model
  family, quantization, reasoning budget, prompt, schema, and token-failure handling must be frozen separately.
- **Deployment track:** remains closed. Persistent databases, real concurrency, credentials, services, side
  effects, and execution require evidence and authority not present here.

## Standing boundaries

- Project-authored and procedural records are never called human-authored or external.
- Development evidence and fresh procedural confirmation remain distinct.
- No policy score may influence population membership.
- Learned confidence cannot authorize a trusted mutation.
- Permanent registration, real service calls, external side effects, and execution remain zero.
