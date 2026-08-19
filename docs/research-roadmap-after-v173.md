# Research roadmap after V173

## Updated evidence state

V170 and V171 remain strong isolated confirmations: adaptive evidence gathering improves the original
cost-sensitive class-decision objective, and the reversible fixed-ontology sandbox preserves safety and recovery
under fresh stateful histories.

V173 shows that the two mechanisms do **not** yet compose beneficially under a unanimous deterministic commit
certificate. Safety was exact, but every operational policy deferred on every target. The unchanged exact planner
spent 1.8063 queries on average and reduced recommendation risk to 1.2031, yet safe routed risk increased from 2.0
under no query to 2.1806. No V173 state improved over immediate deferral.

The non-operational oracle completed trusted work with probability two thirds, so the population contained real
trusted opportunity. The bottleneck was certification: two queries could improve a class decision but could not
eliminate every candidate from competing classes.

## Closed branch

Do not run a V173 confirmation. Do not reinterpret recommendation risk as routed-system benefit. Do not weaken
the unanimity gate, increase the V173 horizon, change its costs, or tune its query choices after observing V172.
V173 is frozen as a safe nonbeneficial boundary result.

## Active Track E — certification-aware planning

The next mechanism must optimize the actual authority boundary from the start:

```text
terminal version space
        -> unanimous trusted class: deterministic typed route
        -> otherwise: defer
```

The planner's objective is expected routed terminal loss plus query cost. A posterior mode or recommended class
has no commit meaning. The unanimous gate and V171 sandbox remain unchanged.

### Recommended sequence

1. **V174 certificate-depth feasibility census.** On a declared development population, compute exact minimal
   additional query certificates for alias, composition, and provisional targets, plus the best achievable trusted
   completion at each horizon from zero through all remaining valuations. This is structural feasibility, not a
   tuned policy comparison. Retain targets that cannot certify early.
2. **V175 certification-aware planner development.** Freeze a horizon and query cost prospectively using only the
   V174 structural census. Implement exact dynamic programming whose stop action is the deterministic consensus
   route or deferral. Compare against immediate deferral, V167 unchanged, greedy information gain, fixed open-loop,
   random, and a certification oracle. Report query use, trusted completion, routed risk, class-conditional behavior,
   and statewise dominance. Keep all safety gates noncompensatory.
3. **Fresh confirmation only if beneficial.** If V175 improves routed risk without weakening safety, create a new
   constraint family whose membership is frozen before scoring—preferably complete four-constraint states, which
   are exact-signature-disjoint from V172. Reuse the certification-aware policy unchanged. A mixed or negative
   result remains closed without tuning.

## Other tracks

- **Factored inference/planning (Track C):** retained as positive mechanism evidence for recommendation decisions;
  it is a baseline, not commit authority.
- **Reversible sandbox (Track B):** retained as positive safety evidence; no need to repeat it until the routing or
  persistence contract changes.
- **Open-language/model track (Track A):** remains dormant. No nonempty truth-free language residual has appeared,
  and V173's failure is exact certification geometry, not missing linguistic capability. A local or API LLM would
  not solve it and is not authorized.
- **Deployment:** remains closed. Persistent databases, arbitrary concurrency, credentials, services, effects,
  and execution are outside the evidence.

## Standing boundaries

- The certification gate is deterministic and cannot be bypassed by a planner, model, posterior, or hidden target.
- `provisional_primitive` never enters the trusted sandbox.
- Population construction and scoring remain separate.
- Project-authored procedural evidence is not external or human-authored.
- Registration, real services, real-state mutation, side effects, and execution remain zero.
