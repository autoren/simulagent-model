# V175 certification-aware planner development plan

## Mechanism

V175 is a new planner, not a V173 repair. Its maximum horizon is frozen at five because V174 found zero possible
trusted completion through horizon four and positive completion first at horizon five. Query cost remains 0.1.

At every belief, stopping has the actual routed-system meaning: a unanimously trusted version space routes with
loss zero; every other version space defers with loss two. Exact dynamic programming compares that stop risk with
the expected child risk plus query cost and prefers stopping on ties. A class recommendation, posterior mode, or
hidden target never authorizes a route.

## Development comparison

All 132 V172 development states and 4,224 targets are retained with frozen class-balanced weights. Controls are
immediate deferral, unchanged V167 exact adaptive planning at horizon two, greedy class information gain at
horizon five with consensus stopping, the exact best fixed query subset, an exact uniform average over every
query order with consensus stopping, and a non-operational target-informed minimal-certificate oracle.

Every trusted route passes through the same deterministic gate and V171 reversible simulation. Safety gates are
noncompensatory. Benefit and dominance are separate from safety, and a negative or mixed result cannot be tuned on
V172.

## Boundary

This is development evidence on a population already used to diagnose V173. If beneficial, the unchanged policy
must be tested on a separately frozen exact-signature-disjoint population before any broader claim. No language,
model, API, registration, real state, real service, side effect, or execution is allowed.
