# V171 stateful sandbox sequence confirmation plan

## Purpose

V171 asks whether V168's fixed-ontology transaction boundary survives fresh, multi-transaction histories rather
than isolated fixtures. The ontology, invariants, proposal validation, revision increments, preview binding,
atomic patch semantics, independent verification, and rollback contract remain unchanged. A simulation-only
durability adapter adds lifecycle status, a recovery journal, restart, and fail-closed provenance checks.

## Population

The formal population is the complete Cartesian product of 11 frozen scenario families and 12 variant indices,
for 132 sequences. No sequence may be selected, excluded, or tuned after outcomes are inspected. Each sequence
starts from a fresh V168-valid state and contains multiple protocol events. Together they cover stale-preview
races, replay across restart, four crash points, a partial multi-entity write, repeated rollback across restart,
multiple provenance-preserving restarts, provenance tampering, and a retained atomic multi-entity update.

Unverified prepared or applied work must recover by restoring its complete before-state. Verified work may be
finalized after restart only when the durable state exactly matches the preview-bound expected post-state.
Corrupted provenance must stop recovery and continuation without changing state. Every recoverable scenario must
also accept and retain a valid tail transaction after recovery.

## Gates and boundary

Every final state and expected disposition must match the frozen oracle. Safety, recovery, idempotence,
provenance, atomicity, post-recovery continuation, and invariant metrics are noncompensatory and must each be
100%. The fixed V168 source and locked artifacts may not be edited.

The store, crashes, persistence, and restarts are all local in-memory simulations. This is not a database,
concurrency, host-compromise, or deployment proof. No language data, model, API, training, provisional concept,
real service, side effect, or execution is allowed. A pass authorizes only a separate trusted-only shadow
integration design; it does not authorize the integration run itself.
