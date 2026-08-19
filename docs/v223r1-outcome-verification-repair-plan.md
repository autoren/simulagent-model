# V223r1 outcome-verification repair plan

## Failure boundary

The V223 metadata census completed and its metrics, result branch, manifest, evidence, and snapshots reconstructed
exactly. The first outcome-verifier invocation nevertheless failed because `docs/research-roadmap-after-v221r1.md`, a
hashed V223 design dependency, had been updated after the formal run but before verification. The failed audit recorded
exactly one false check: `design_lock_and_dependencies_are_exact`.

No task record, request body, model, training process, ontology mutation, service action, or execution was involved.
The roadmap was restored byte-for-byte to its locked hash. The V223 scientific artifacts were not changed.

## Repair

V223r1 is verification-only. It will:

1. preserve and hash the original failed V223 outcome audit;
2. prove that every V223 design dependency now matches the original lock;
3. reconstruct the exact existing V223 metrics, scientific audit, result, manifest, evidence, and snapshot hashes;
4. verify that the original failure contained no scientific or access-boundary failure; and
5. freeze the existing positive V223 outcome without retrieving or scoring a new census.

The repair cannot alter the source assessments, recommendation, gates, branch, decision, or successor authorization.

## Authorization boundary

One exact outcome reconstruction and freeze is permitted. No new metadata retrieval, task language, record endpoint,
model/API call, training run, registration, trusted-state mutation, service action, side effect, or execution is
authorized.

