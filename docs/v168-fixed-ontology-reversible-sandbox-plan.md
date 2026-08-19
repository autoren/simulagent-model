# V168 fixed-ontology reversible sandbox plan

## Purpose

V168 tests the end-to-end control plumbing that can be studied without solving open-language ontology induction.
Only three trusted `Device` entities and three registered mutable fields exist. Typed proposals are previewed on a
copy, checked against deterministic invariants, committed atomically to an in-memory store, independently
verified, and either retained or rolled back. There is no real service or tool target.

## Population

The one formal development census contains 132 project-authored fixtures: 12 each for valid retention, explicit
rollback, atomic multi-entity update, invariant violation, unauthorized field, stale revision, malformed type,
duplicate-field conflict, unknown entity, preview-token tampering, and post-commit corruption.

The injected corruption condition is deliberate: after a valid commit, the harness mutates an untargeted field.
The independent verifier must detect the mismatch and restore the complete pre-transaction snapshot. This tests
the recovery boundary without claiming protection from arbitrary host compromise.

## Gates

Every disposition and final state must be exact. Preview must never mutate state; committed state must equal the
accepted preview; multi-entity transactions must be atomic; rejected transactions must leave state unchanged;
explicit and automatic rollback must recover the complete prior state; invariants must hold at every retained
boundary; commit may mutate only proposed fields plus managed revisions; both injected fault types must be
detected; and the append-only provenance hash chain must verify.

No evaluation data, human judgment, model, API, training, provisional concept, real service, external side effect,
or real execution is allowed. Passing remains simulated development evidence and authorizes no deployment.
