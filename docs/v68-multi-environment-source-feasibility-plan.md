# V68 source-feasibility lock for multi-environment replication

## Purpose

This stage freezes the source universe before any new multi-environment policy, information-gain,
regret, or approximate-inference result is computed. The universe is all 14 Cassandra POMDP files
in POBAX commit `a5e1d62d14e4efe783885b9d4f19cffa2a568eec`. It is an infrastructure and
provenance audit, not the V68 replication evaluation.

The V62 parser intentionally supported only the full-matrix subset needed at that time. Four files
in the pinned source use sparse entries or Cassandra keywords. A new strict parser therefore covers
the exact grammar union present in the checkout while leaving the frozen V62 and V67 parsers
unchanged. Its arrays and metadata must match POBAX's own pinned `POMDPFile` parser for every file.
The reference parser class is extracted directly from the checkout so its optional Gym/JAX runtime
dependencies are not needed.

## Prospective split

The split is determined only by prior project exposure. `4x3_nonterminating`, `tiger-alt-start`,
`tmaze2`, and `tmaze5` were used in V62–V67 and are development-only. Untouched files remain
confirmatory candidates. The terminating 4x3 and fully observable T-maze variants are explicitly
marked structurally related; cheese, hallway, heaven-hell, network, and shuttle are marked novel.
No new policy outcome is consulted for this assignment.

Every source model must pass normalization and finiteness checks. The pinned `paint.POMDP` is
expected to fail observation normalization because its terminal row has mass two under POBAX's own
sequential wildcard semantics. It is excluded before any planning result and will not be silently
renormalized or repaired. All other valid untouched models remain registered. A later resource
deferral, if necessary, may use only prospectively frozen structural operation bounds.

## Two separate replication claims

Tier A will test one project-authored static command-channel uncertainty family across external
models with three, four, and five actions. The environmental arrays are external, but the uncertainty
family is not; that boundary remains explicit.

Tier B is frozen now as the matched external pair `cheese.95.POMDP` and
`cheese.95_nonterminating.POMDP`. The two source files share state, action, and observation labels
and supply different termination dynamics, so binary model identity is externally supplied. Tier B
will be designed and reported separately from Tier A.

## What this lock authorizes next

A passing source audit authorizes only exact infrastructure and outcome screening on the four
development models. Confirmatory models may be parsed, structurally validated, and assigned a
prospective operation bound, but no policy value, EIG, regret, or SMC2 score may be computed until a
full design, implementation audit, and immutable population protocol are locked.

The eventual design must keep horizons short enough for exact reference computation, retain the
full generated population, report full-suite and per-environment results, include a worst-environment
gate, and define any posterior-sensitive stratum using exact-oracle quantities before SMC2 scoring.
MAP, first-repeat, posterior-sampling, and open-loop controls are required. Tier A and Tier B cannot
be merged into one external-uncertainty claim.
