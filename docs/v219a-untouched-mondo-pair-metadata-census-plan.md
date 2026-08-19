# V219A untouched Mondo pair metadata census plan

## Question

V218 found a strong Mondo population but failed because a repository `README.md` asset was assigned the wrong semantic
role. V219A asks a narrower question without opening another payload:

> Does the already-frozen official Mondo release metadata identify one completely untouched adjacent release pair with
> exact bounded assets, complete controls, and an actual release-body summary suitable for a new payload design?

This is a local metadata census. It is not an in-place V218 repair, payload run, deterministic evaluation, or model
experiment.

## Frozen evidence and prior exposure

The sole evidence source is the content-hashed Mondo GitHub releases API snapshot captured in V217A:

```text
outputs/v217a-independent-source-event-metadata-census/
  metadata-snapshots/f1e464db2cd1cdc3df7e.json
sha256: 2cdcb1f171bf912c57d1b5a48364da7afeb211cd8bd00af6e6a1b70209d63072
```

The snapshot and its release bodies were already inspected in V217A and V218 planning, so V219A makes no blind
discovery claim. It performs no network request and reads no ontology or control payload body. It also reads no V218
development or protected record.

## Untouched adjacency and selection

Releases are ordered by `published_at` descending, with tag name as a deterministic tie-breaker. Adjacency is computed
over that complete frozen order before exclusions. Only then are pairs containing either V218-opened release
(`v2026-07-06` or `v2026-08-04`) removed. This prevents exclusion from manufacturing a pair that was never adjacent.

Eligible pairs are ordered newest-newer-release first. At most the first eligible pair is selected. No scientific
metric, payload count, or downstream result can alter that priority.

## Required assets

Each pair must expose exactly eight prospectively defined roles in the frozen release objects:

- older and newer `mondo-base.obo`;
- newer changed-term and new-term diff TSVs;
- older and newer obsoletion-candidate TSVs; and
- older and newer source-version TSVs.

Every asset must have an HTTPS URL, positive integer byte count, and published SHA-256 digest. Each asset must be at
most 50,000,000 bytes and all eight together at most 99,000,000 bytes. No substitution or link expansion is allowed.

## Correct release-summary control

The release-summary control is the exact `body` field of the newer release object already present in the frozen API
snapshot. Its SHA-256 hash is stored in census evidence. A repository README is explicitly not accepted as release-
summary evidence.

The body must document all four categories using frozen patterns:

- addition;
- label or renaming;
- text definition;
- obsoletion with replacement.

These categories establish metadata feasibility only. They do not prove that a later parser or population will pass.

## Decision and boundaries

A positive result selects at most one pair and authorizes only a separately audited payload-design protocol. It does
not authorize payload retrieval, reuse of V218 records, deterministic evaluation, protected access, a local or API
model, training, registration, mutation, service action, or execution.

If no pair passes every exact asset, digest, byte, release-body, selection, integrity, and access gate, V219A freezes a
negative result and stops the fresh Mondo payload branch without tuning any pair, asset role, pattern, threshold, or
gate.

Historical release metadata can support retrospective reconstruction of published Mondo semantics. It cannot establish
the intended meaning of a new speaker or replace appropriate domain-expert evidence for strong ontology-acquisition
claims.
