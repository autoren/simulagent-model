# V171 stateful sandbox sequence confirmation results

## Outcome

V171 passed every preregistered confirmation gate on the complete frozen population of 132 project-authored
stateful sequences: 11 scenario families crossed with 12 variant indices. Every expected disposition and oracle
final state was exact.

All scenario-specific metrics were 100%:

- stale previews lost revision races without changing the winner's state;
- committed transaction replays remained idempotently rejected across restart;
- crashes after preview, after snapshot, and after unverified apply restored the complete before-state;
- crashes after successful verification finalized the preview-bound post-state;
- partial multi-entity writes were fully rolled back;
- repeated rollback stayed idempotent across restart and never resurrected an old snapshot;
- provenance chains remained valid through multiple restarts;
- tampered provenance failed closed before recovery or continuation;
- valid multi-entity updates remained atomic; and
- every recoverable sequence accepted and retained a valid follow-up transaction.

Every externally visible boundary preserved the fixed V168 invariants. Retained mutations were limited to proposed
fields and managed revisions, and every provenance chain in the surviving harness remained valid.

## Interpretation

V168 established the transaction mechanism on isolated development fixtures. V171 confirms the same fixed
ontology, validation, revision, preview-binding, atomic-patch, verification, and rollback semantics under fresh
multi-transaction histories. Population membership was frozen as a complete Cartesian product before any formal
sequence outcome was opened; no sequence was selected or excluded using results.

The added lifecycle and recovery adapter provides a coherent fail-closed policy: prepared or applied but
unverified work rolls back, while verified work is finalized after restart only if durable state exactly matches
the expected post-state bound into the preview. Provenance is checked before recovery or continuation.

## Boundary and decision

This remains bounded in-memory simulation evidence. It is not validation of a persistent database, arbitrary
concurrency, filesystem durability, host compromise, deployment, or a real service. The records are procedural
and project-authored, not human-authored. No language data, model, API, training, provisional ontology concept,
real service call, side effect, or execution was used.

Freeze V171 as a positive fresh stateful sandbox confirmation. Together with V170, it satisfies the roadmap's
precondition for designing—but not yet running—a separately locked trusted-only shadow integration. Provisional
candidates must remain outside commit authority, and deterministic validation plus the independent verifier must
retain final authority.
