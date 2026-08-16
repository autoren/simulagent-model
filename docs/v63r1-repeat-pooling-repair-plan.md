# V63r1 repeat-pooling measurement repair

V63 completed all records and failed only primary mean binned joint `(identity, theta)` TV:
`0.0730573` versus the frozen `0.06` maximum. The original outcome is immutable and remains a
failure.

The post-result localization found a scorer discrepancy against the frozen estimator being
transferred. V53r2's `pool_smc2_repeats` forms an equal-weight mixture of the three independent
posterior estimates, including their program/identity marginal, theta particles, binned joint
posterior, state posterior, predictive posterior, atoms, and log-mean evidence. It computes one
accuracy row per record and particle budget. V63 instead computed one row per repeat and averaged
the three error values. Because total variation is convex, average per-repeat histogram TV is at
least the TV of the pooled posterior and retains a three-times-smaller effective sample size in the
measurement.

V63r1 changes only that aggregation rule. For every one of the same 32 exact records and each
unchanged outer budget `31/127/509`, it reruns the same three deterministic repeat seeds, creates
one equal-weight posterior mixture, and computes one metric row. Log evidence is pooled with
log-mean-exp. Stream, ESS, ancestry, resampling, and rejuvenation diagnostics remain summaries over
the three independent unpooled runs.

The candidate inference source, population, truth sidecars, identities, theta prior, quadrature,
particle counts, proposal, seeds, bins, controls, gates, and hierarchy cannot change. The exact SBC,
scale, and pinned-runtime results are reused byte-for-byte from the failed V63 result and runtime
artifact; they are not rerun. Only exact-benchmark and exact-dependent gate values may be replaced.

A repaired pass is a measurement repair, not independent replication. It would authorize only the
same next step the original V63 design specified: preregistration of a separate external EIG model
with at least two informative nonterminal actions. Tiger itself remains disallowed as a substantive
active-design test. V58, reward planning, verification, model access, and adapter training remain
outside V63r1.
