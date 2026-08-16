# V65r2 preregistration: exact-zero identity support repair

Date frozen: 2026-08-16

## Motivation and status

V65r1 remains a failed one-shot run. Its immutable evaluation terminated before writing any scored
result because an identity-conditioned SMC² filter raised when every particle received zero
likelihood. A post-failure exact support audit, which computed no candidate EIG, found one sealed
history that is possible under identity 0 and structurally impossible under identity 1. The joint
Bayesian posterior is nevertheless well-defined.

V65r2 is not a new benchmark or an independent replication. It is a targeted implementation-domain
repair using the same 48 histories in the same order, the same approximate and exact methods, and
the same gates.

## Sole inference repair

Before running an identity-conditioned particle filter, V65r2 computes Boolean support through the
pinned 11-state initial, transition, and observation arrays. The check is topological: an edge is
present exactly when its frozen probability is strictly positive. Throughout the registered theta
support `[0.6, 0.95]`, command-success and command-failure probabilities are both positive, so the
transition support does not depend on theta.

If a public history has zero Boolean support under one identity, that branch returns log evidence
`-Infinity`, no particles, and zero joint posterior mass. It is not an error. If both identities
have zero support, the public history is invalid and inference fails before normalization. If
Boolean support is positive but the finite particle approximation nevertheless becomes extinct,
V65r2 raises a distinct hard error; it must not disguise finite-particle collapse as certainty that
an identity is impossible.

All positive-mass atoms, inner filters, evidence estimates, PMMH moves, pooled repeat posteriors,
state diagnostics, and Rao–Blackwellized acquisition predictions remain unchanged from V65r1.

## Durable one-shot protocol

The evaluator must atomically create an attempt marker before any record is scored. An existing
attempt marker, result, or failure rejects every later invocation. On success it atomically writes
raw cells and the result. On any caught exception it atomically writes a terminal failure artifact
containing the exception, stage, progress counters, access counters, and frozen hashes, then exits
nonzero. The marker consumes the one-shot authorization even if no accuracy result is produced.

This protocol repairs V65r1's missing failure serialization; it does not authorize a V65r1 retry.

## Frozen evaluation

After separate implementation and evaluator locks, V65r2 may run once on the existing sealed
subset. Every V65r1 posterior, predictive, four-action EIG, selection, regret, scaling, control,
normalization, finite-value, access, and upstream-audit gate remains noncompensatory and unchanged.
The budgets remain `[31, 127, 509]`, the inner budget remains 127, and every budget retains three
independent repeats pooled before scoring.

A pass may authorize only preregistration of the external Bayes-adaptive reward-decision stage. A
failure or exception is terminal for V65r2 and does not unlock planning, formal verification,
human-data substitution, model access, or adapter training.
