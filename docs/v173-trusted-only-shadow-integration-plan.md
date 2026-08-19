# V173 trusted-only shadow integration plan

## Question

V173 tests whether the unchanged uncertainty-aware policies and unchanged reversible sandbox compose usefully
when authority stays deterministic. It evaluates every V172 eligible state and target with exact class-balanced
weights. No V172 state or target may be removed after scoring.

## Authority boundary

A policy may choose up to two simulated Boolean inspections and may recommend a terminal class or deferral. Its
recommendation never authorizes a transaction. After the policy stops, a deterministic gate examines only the
exact surviving version space:

- unanimous `alias` routes a registered single-entity typed proposal;
- unanimous `composition` routes a registered atomic multi-entity typed proposal;
- a mixed class set defers; and
- unanimous `provisional_primitive` also defers and never enters the sandbox.

Every trusted route is previewed, validated, committed, independently verified, restarted, and provenance-checked
inside the V171 in-memory simulation. The hidden target is used only to generate inspection outcomes and score the
shadow study. The oracle comparator reveals the target before applying the same gate and is explicitly
non-operational.

## Controls and estimands

The V167 prior, loss matrix, query cost, two-query horizon, policy tie-breaks, exact adaptive planner, greedy
information-gain policy, optimal and uniformly averaged random open-loop pairs, no-query policies, and oracle are
unchanged. The corrected V167r1 history metric remains frozen but is not an integration authority.

Planner recommendation loss and safe routed-system loss are reported separately. This matters because the gate
can safely defer after a wrong forced-MAP recommendation; that prevents a bad commit but does not make the MAP
recommendation correct. The main operational estimands are routed loss plus inspection cost, trusted completion,
deferral, false trusted routing, sandbox exactness, and statewise comparison.

Safety gates are noncompensatory. Benefit and strong-comparison thresholds are separate: a safe but nonbeneficial
integration remains a negative boundary result, while a beneficial result need not be mislabeled unsafe merely
because another safe policy is better. No threshold failure permits tuning on V172.

## Claim boundary

All language, models, APIs, training, provisional registration, real trusted-state mutation, services, side
effects, and execution remain absent. V173 is finite-DSL, project-authored, local in-memory shadow evidence. Any
confirmation requires a new nonoverlapping population and lock.
