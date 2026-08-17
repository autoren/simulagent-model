# V71 sensor-codebook development result

## Decision

V71 failed its prospectively frozen development gates and stops before protected confirmation. The complete 21-record census was evaluated once across `concert.POMDP`, `ejs1.POMDP`, and `manuel-hartman.2013-09-19.POMDP`. No record or model was selected, rejected, replaced, repaired, or renormalized, and the protected confirmation policy-value count remained zero.

## Result

All implementation-validity gates passed: source validation, belief normalization, finite metrics, full record completion, and point-model support were each `1.0`. Both sensor-codebook point models remained on-support on every evaluated branch, with zero fallback calls.

The scientific gates all failed:

- models with an exact-BA/MAP root-action disagreement: `0 / 3` (required `3`);
- models with material MAP regret: `0 / 3` (required `2`);
- models with material posterior-sampling regret: `0 / 3` (required `1`);
- maximum normalized MAP regret: `0.0` (required `0.01`).

Exact Bayes-adaptive, MAP, open-loop, and myopic values were identical throughout the census. Posterior-sampling discrepancies were bounded by `5.93e-17` in absolute normalized regret and are floating-point noise, far below the frozen `0.005` materiality threshold.

## Development-only diagnostic

At the root, the horizon-three exact planner chose the same source-order action as every relevant control. The exact action values were:

- `concert.POMDP`: `nothing`, with Q-values `[-10.0, -2.0, 0.0]`;
- `ejs1.POMDP`: `Manufacture`, with Q-values `[1.52621175, 1.27621175, 0.221525, -0.278475]`;
- `manuel-hartman.2013-09-19.POMDP`: `b`, with Q-values `[4.0, 30.0]`.

This post-outcome diagnostic is confined to the three development models. It indicates that the chosen source problems and horizon admit a common dominant action sequence, leaving no decision value for retaining uncertainty over observation-label semantics. It does not show that sensor-semantic uncertainty is generally irrelevant.

## Claim boundary and next direction

V71 is a clean negative boundary result. It removes V70's off-support fallback confound by construction, but the new family does not create policy separation on the frozen development set. The reliability, horizon, models, controls, normalization, and gates must not be tuned retrospectively, and the five protected models must not be opened.

Any successor study should be a new preregistration using fresh environments whose design explicitly contains an action-dependent sensing-versus-control tradeoff and delayed state-dependent reward. It should establish that structural property from source metadata before outcomes, while retaining the identical-support requirement. Until such a source set exists, this branch should defer rather than search the V71 protected set or relax V71's gates.
