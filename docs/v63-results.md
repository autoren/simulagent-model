# V63 external unknown-dynamics inference results

**Qualification:** FAIL
**Decision:** `repair_or_reject_v63_inference_before_active_selection`

V63 completed its single sealed evaluation over the pinned POBAX Tiger anchor plus the explicitly
project-authored persistent/alternating transition layer. The result does not claim that POBAX
supplies an unknown-dynamics benchmark.

## Passed evidence

- Exact identity TV: mean `0.00123500`, q95 `0.00399113` at 509 outer particles.
- Theta Wasserstein: mean `0.00342040`, q95 `0.00684971`.
- Current-side TV: mean `0.00182399`, q95 `0.00643313`.
- Next-observation TV: mean `0.00156405`, q95 `0.00463496`.
- Mean absolute log-evidence error: `0.0266678`.
- SBC minimum rank chi-square p-value: `0.00845262`.
- SBC maximum rank-bin z: `3.25`; maximum coverage z: `2.35151`.
- Exact, SBC, and scale normalization rates: `1.0`.
- Target-identity extinction and false-collapse rates: `0.0`.
- Unintended stream and resampling-fingerprint collision rates: `0.0`.
- Pinned POBAX runtime transition/observation array errors: `0.0`.
- Maximum runtime empirical-probability error: `0.0134375`.
- All seven controls were detected or dominated.

## Blocking gate

Primary mean binned joint `(identity, theta)` TV was `0.0730573`, above the frozen maximum `0.06`.
Its q95 value, `0.101778`, remained below the separate `0.18` gate. Every other registered gate
passed, but the gates are noncompensatory, so V63 does not qualify and does not authorize active
selection or planning.

## Post-result localization

The frozen V53r2 evaluator forms an equal-weight mixture of its three independent SMC² posterior
replicates and then computes one metric row per record and budget. V63 declared those same three
inherited repeats, but its scorer computed each repeat's metric separately and averaged the three
TV values. TV is convex, so average per-repeat TV retains the finite histogram noise that pooling
is intended to reduce and cannot be interpreted as the inherited pooled estimator's error.

This is a measurement-implementation discrepancy, not evidence that the failed value should be
silently replaced. Original V63 remains failed and immutable. A valid repair must be preregistered
and may change only the repeat aggregation rule to the V53r2 equal-weight posterior mixture. It
must reuse the exact population, candidate implementation, seeds, particle budgets, binning,
controls, thresholds, SBC, scale, and runtime outputs. Any repaired run is a measurement repair,
not independent replication.

V58 remains deferred; no human records were accessed or simulated, and no model or adapter was
run.
