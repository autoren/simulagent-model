# V69 development-only dominant-remapping preregistration

V68r2 showed that the nominal-dominant command-channel family was not sensitive enough for the intended Bayes-adaptive comparison. Across 59 sealed development records it produced no exact-BA/MAP root disagreement, no material MAP regret, and no material posterior-sampling regret. This was not a numerical or evaluation failure: the census, convergence, finiteness, source, and simpler-control gates passed.

V69 changes one structural assumption. A persistent hidden identity selects the forward or backward permutation of the already-frozen per-model action cycle, and θ in [0.6, 0.95] is the probability that the remapped action executes; the nominal command executes with probability 1−θ. Thus neither hidden identity preserves the nominal command as modal. This creates a genuine control-relevant system-identification problem while retaining the same state spaces, observations, rewards, action labels, two-identity prior, θ prior, horizon, exact inference machinery, and interpretation as an unknown action channel.

The census is reconstructed from the V69 mixture and retains every positive-probability depth-0 and depth-1 public history. No V68 posterior or history probability is reused. MAP and persistent posterior-sampling policies use the already-audited deterministic exact-zero off-support fallback from their first run, so policy totality is part of the design rather than a repair.

All 19 V68 gates and the normalization are unchanged. No confirmatory model may be parsed for outcome scoring. A passing development screen can authorize only a fresh confirmatory preregistration; a failure closes V69 without holdout access.
